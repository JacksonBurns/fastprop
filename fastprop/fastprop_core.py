"""
fastprop - molecular property prediction with simple descriptors

This is an attempt to do molecular property prediction using learnable linear
combinations of the mordred descriptors, rather than a more complex deep learning
architecture. The advantage of this approach would be that the model should
train much faster since the representation (graph->vector) is much simpler
than competing GCN and MPNN approaches. The difficulty will be in making the
representation sufficiently flexible s.t. the network can actually 'learn' a
good representation.

TODO:
 - add the 3D descriptors using a best-guess geometry (openbabel?) to try
   and improve the predictions
 - validate that the transformed input features are actually different from the regular input features
   (i.e. look at the output from the first interaction layer)
 - add a way to support reading mordred output from using it like this: python -m mordred example.smi -o example.csv
   (can be kind of a cop-out for speed considerations)
 - https://github.com/chriskiehl/Gooey
 - need a function to write the model to a checkpoint file, and then do so occasionally during training

To reduce total calculation burden, could interpolatively sample dataset to get some small percent,
and then see which among those are transformed out, and then calc just the remaining for rest of
dataset. Still should make calc'ing all an option.

hyperparameter optimization:
https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py
"""
import datetime
import logging
import os
import warnings
from pathlib import Path
from time import perf_counter
from typing import OrderedDict

import numpy as np
import pandas as pd
import psutil
import pytorch_lightning as pl
import torch
from astartes import train_val_test_split
from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as l2_error
from torch.utils.data import Dataset as TorchDataset

# choose the descriptor set absed on the args
from fastprop.utils import (
    ALL_2D,
    SUBSET_947,
    calculate_mordred_desciptors,
    load_cached_descs,
    load_from_csv,
    load_from_morded_csv,
    mordred_descriptors_from_strings,
)

from .defaults import _LOGGING_ARGS
from .preprocessing import preprocess

descriptors_lookup = dict(
    optimized=SUBSET_947,
    all=ALL_2D,
)

warnings.filterwarnings(action="ignore", message=".*does not have many workers which may be a bottleneck.*")

# device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 1

# 5e-4 generally a good setting, lowered for qm9
NUM_VALIDATION_CHECKS = 10


logging.basicConfig(**_LOGGING_ARGS)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _collate_fn(batch):
    descs = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return descs, labels


class ArbitraryDataset(TorchDataset):
    def __init__(self, data, targets):
        self.data = data
        self.length = len(targets)
        self.targets = targets

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


class ArbitraryDataModule(LightningDataModule):
    def __init__(self, cleaned_data, targets, batch_size, random_seed, train_size, val_size, test_size, sampler):
        super().__init__()
        self.data = cleaned_data
        self.targets = targets
        self.batch_size = batch_size
        self.train_size, self.val_size, self.test_size = train_size, val_size, test_size
        self.sampler = sampler
        self.train_idxs, self.val_idxs, self.test_idxs = None, None, None
        self.random_seed = random_seed

    def prepare_data(self):
        # cast input numpy data to a tensor
        if not isinstance(self.data, torch.Tensor):
            self.data = torch.tensor(self.data, dtype=torch.float32)
        if not isinstance(self.targets, torch.Tensor):
            self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def setup(self, stage=None):
        # partition data randomly
        (
            *_,
            self.train_idxs,
            self.val_idxs,
            self.test_idxs,
        ) = train_val_test_split(
            np.array(self.targets),
            train_size=self.train_size,
            val_size=self.val_size,
            test_size=self.test_size,
            sampler=self.sampler,
            random_state=self.random_seed,
            return_indices=True,
        )

    def _init_dataloader(self, shuffle, idxs):
        return torch.utils.data.DataLoader(
            ArbitraryDataset(
                [self.data[i] for i in idxs],
                [self.targets[i] for i in idxs],
            ),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            persistent_workers=True,
            collate_fn=_collate_fn if self.batch_size > 1 else None,
        )

    def train_dataloader(self):
        return self._init_dataloader(True, self.train_idxs)

    def val_dataloader(self):
        return self._init_dataloader(False, self.val_idxs)

    def test_dataloader(self):
        return self._init_dataloader(False, self.test_idxs)


class fastprop(pl.LightningModule):
    def __init__(self, number_features, target_scaler, num_epochs, hidden_size, learning_rate, fnn_layers, shh=False):
        super().__init__()
        # for saving human-readable accuracy metrics and predicting
        self.target_scaler = target_scaler

        # shh
        self.shh = shh

        # training configuration
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        # fully-connected nn
        layers = OrderedDict(
            [
                ("lin1", torch.nn.Linear(number_features, hidden_size)),
                ("act1", torch.nn.ReLU()),
            ]
        )
        for i in range(fnn_layers - 1):
            layers[f"lin{i+2}"] = torch.nn.Linear(hidden_size, hidden_size)
            layers[f"act{i+2}"] = torch.nn.ReLU()
        self.fnn = torch.nn.Sequential(layers)
        self.readout = torch.nn.Linear(hidden_size, self.target_scaler.n_features_in_)

    def forward(self, x):
        x = self.fnn.forward(x)
        x = self.readout(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        loss = self._machine_loss(batch, reduction="sum")
        self.log("train_mse_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, y, y_hat = self._machine_loss(batch, return_all=True)
        self.log("validation_mse_loss", loss, on_step=False, on_epoch=True)
        self._human_loss(y_hat.detach().cpu(), y.detach().cpu(), "validation")
        return loss

    def test_step(self, batch, batch_idx):
        loss, _, y, y_hat = self._machine_loss(batch, return_all=True)
        self.log("test_mse_loss", loss, on_step=False, on_epoch=True)
        self._human_loss(y_hat.detach().cpu(), y.detach().cpu(), "test")
        return loss

    def _machine_loss(self, batch, reduction="mean", return_all=False):
        x, y = batch
        y_hat = self.forward(x)
        loss = torch.nn.functional.mse_loss(y_hat, y, reduction=reduction)
        if not return_all:
            return loss
        else:
            return loss, x, y, y_hat

    def _human_loss(self, pred, real, name):
        # outputs the performance in more human-interpretable manner
        # expensive so we don't do this during training
        rescaled_pred = self.target_scaler.inverse_transform(pred)
        rescaled_truth = self.target_scaler.inverse_transform(real)
        # mean absolute percentage error
        # sklearn asks for an array of weights, but it actually just passes through to np.average which
        # accepts weights of the same shape as the inputs
        per_task_mape = mape(rescaled_truth, rescaled_pred, multioutput="raw_values", sample_weight=rescaled_truth)
        if len(self.target_scaler.feature_names_in_) == 1:
            self.log(f"unitful_{name}_wmape", np.mean(per_task_mape))
        else:
            self.log(f"unitful_{name}_mean_wmape", np.mean(per_task_mape))
            for target, value in zip(self.target_scaler.feature_names_in_, per_task_mape):
                self.log(f"unitful_{name}_wmape_output_{target}", value)

        # mean absolute error
        all_loss = torch.nn.functional.l1_loss(torch.tensor(rescaled_pred), torch.tensor(rescaled_truth), reduction="none")
        per_task_loss = all_loss.numpy().mean(axis=0)
        if len(self.target_scaler.feature_names_in_) == 1:
            self.log(f"unitful_{name}_l1", np.mean(per_task_loss))
        else:
            self.log(f"unitful_{name}_l1_avg", np.mean(per_task_loss))
            for target, value in zip(self.target_scaler.feature_names_in_, per_task_loss):
                self.log(f"unitful_{name}_l1_output_{target}", value)

        # rmse
        per_task_loss = l2_error(rescaled_truth, rescaled_pred, multioutput="raw_values", squared=False)
        if len(self.target_scaler.feature_names_in_) == 1:
            self.log(f"unitful_{name}_rmse", np.mean(per_task_loss))
        else:
            self.log(f"unitful_{name}_rmse_avg", np.mean(per_task_loss))
            for target, value in zip(self.target_scaler.feature_names_in_, per_task_loss):
                self.log(f"unitful_{name}_rmse_output_{target}", value)

    def on_train_epoch_end(self) -> None:
        if (self.trainer.current_epoch + 1) % (self.num_epochs // NUM_VALIDATION_CHECKS) == 0:
            if not self.shh:
                logger.info(f"Epoch [{self.trainer.current_epoch + 1}/{self.num_epochs}]")


def train_and_test(
    outdir,
    n_epochs,
    datamodule,
    model,
    verbose=True,
):
    csv_logger = CSVLogger(
        outdir,
        "csv_logs",
    )
    tensorboard_logger = TensorBoardLogger(
        outdir,
        "tensorboard_logs",
    )

    trainer = pl.Trainer(
        max_epochs=n_epochs,
        accelerator=DEVICE,
        enable_progress_bar=False,
        logger=[csv_logger, tensorboard_logger],
        log_every_n_steps=1,
        enable_checkpointing=True,
        check_val_every_n_epoch=n_epochs // NUM_VALIDATION_CHECKS,
    )

    t1_start = perf_counter()
    trainer.fit(
        model,
        datamodule,
    )
    t1_stop = perf_counter()
    if verbose:
        logger.info("Elapsed time during training: " + str(datetime.timedelta(seconds=t1_stop - t1_start)))
    test_results = trainer.test(model, datamodule, verbose=verbose)
    return test_results


def _get_descs(precomputed, input_file, output_directory, descriptors, enable_cache, mols):
    """Loads descriptors according to the user-specified configuration

    Args:
        precomputed (str): Use precomputed descriptors if str is.
        input_file (str): Filepath of input data.
        output_directory (str): Destination directory for caching.
        descriptors (list): List of strings of descriptors to calculate.
        enable_cache (bool): Allow/disallow caching mechanism.
        mols (list): RDKit molecules.
    """
    descs = None
    if precomputed:
        del mols
        logger.info(f"Loading precomputed descriptors from {precomputed}.")
        descs = load_from_morded_csv(precomputed)
    else:
        in_name = Path(input_file).stem
        # cached descriptors, which contains (1) cached (2) source filename (3) types of descriptors (4) timestamp when file was last touched
        cache_file = os.path.join(output_directory, "cached_" + in_name + "_" + descriptors + "_" + str(int(os.stat(input_file).st_ctime)) + ".csv")

        if os.path.exists(cache_file) and enable_cache:
            logger.info(f"Found cached descriptor data at {cache_file}, loading instead of recalculating.")
            descs = load_cached_descs(cache_file)
        else:
            d2c = mordred_descriptors_from_strings(descriptors_lookup[descriptors])
            # use all the cpus available
            logger.info("Calculating descriptors.")
            descs = calculate_mordred_desciptors(d2c, mols, psutil.cpu_count(logical=False), "fast")
            # cache these
            if enable_cache:
                d = pd.DataFrame(descs)
                d.to_csv(cache_file)
                logger.info(f"Cached descriptors to {cache_file}.")
    return descs


def train_fastprop(
    output_directory,
    input_file,
    smiles_column,
    target_columns,
    descriptors="optimized",
    enable_cache=True,
    precomputed=None,
    rescaling=True,
    zero_variance_drop=True,
    colinear_drop=False,
    fnn_layers=3,
    hidden_size=512,
    learning_rate=0.0001,
    batch_size=2048,
    number_epochs=1000,
    number_repeats=1,
    problem_type="regression",
    checkpoint=None,
    train_size=0.8,
    val_size=0.1,
    test_size=0.1,
    sampler="random",
    random_seed=0,
):
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
    torch.manual_seed(random_seed)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    targets, mols = load_from_csv(input_file, smiles_column, target_columns)
    descs = _get_descs(precomputed, input_file, output_directory, descriptors, enable_cache, mols)

    logger.info("Preprocessing data...")
    X, y, target_scaler = preprocess(descs, targets, rescaling, zero_variance_drop, colinear_drop)
    target_scaler.feature_names_in_ = target_columns
    logger.info("...done.")

    datamodule = ArbitraryDataModule(X, y, batch_size, random_seed, train_size, val_size, test_size, sampler)

    model = fastprop(X.shape[1], target_scaler, number_epochs, hidden_size, learning_rate, fnn_layers)

    all_results = []
    for _ in range(number_repeats):
        results = train_and_test(output_directory, number_epochs, datamodule, model)
        all_results.append(results[0])
        random_seed += 1
    # average the results
    results_df = pd.DataFrame.from_records(all_results)
    logger.info("Displaying results:\n%s", results_df.describe().transpose().to_string())
