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
from astartes.molecules import train_val_test_split_molecules
from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from scipy.stats import ttest_ind
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as l2_error
from torch.utils.data import Dataset as TorchDataset
from torchmetrics.functional.classification import multilabel_auroc, f1_score, auroc, multiclass_auroc, binary_accuracy

# choose the descriptor set absed on the args
from fastprop.utils import (
    ALL_2D,
    SUBSET_947,
    calculate_mordred_desciptors,
    load_from_csv,
    load_saved_desc,
    mordred_descriptors_from_strings,
)

from .defaults import init_logger
from .preprocessing import preprocess

descriptors_lookup = dict(
    optimized=SUBSET_947,
    all=ALL_2D,
)

logger = init_logger(__name__)

warnings.filterwarnings(action="ignore", message=".*does not have many workers which may be a bottleneck.*")

# device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 1


NUM_VALIDATION_CHECKS = 20


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
    def __init__(self, cleaned_data, targets, batch_size, random_seed, train_size, val_size, test_size, sampler, smiles=None):
        super().__init__()
        self.data = cleaned_data
        self.targets = targets
        self.batch_size = batch_size
        self.train_size, self.val_size, self.test_size = train_size, val_size, test_size
        self.sampler = sampler
        self.smiles = smiles
        self.train_idxs, self.val_idxs, self.test_idxs = None, None, None
        # either using folds or random sampling, both of these will be incremented by the training loop
        self.random_seed = random_seed
        self.fold_number = 0

    def prepare_data(self):
        # cast input numpy data to a tensor
        if not isinstance(self.data, torch.Tensor):
            self.data = torch.tensor(self.data, dtype=torch.float32)
        if not isinstance(self.targets, torch.Tensor):
            self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def setup(self, stage=None):
        # see `benchmarks/quantumscents/quantumscents.py` for an example of overriding this method
        # to implement multiple folds
        if self.sampler != "scaffold":
            (
                *_,
                self.train_idxs,
                self.val_idxs,
                self.test_idxs,
            ) = train_val_test_split(
                np.array(self.data),
                # flatten 1D targets
                np.array(self.targets).flatten() if self.targets.size()[1] == 1 else np.array(self.targets),
                train_size=self.train_size,
                val_size=self.val_size,
                test_size=self.test_size,
                sampler=self.sampler,
                random_state=self.random_seed,
                return_indices=True,
            )
        else:
            (
                *_,
                self.train_idxs,
                self.val_idxs,
                self.test_idxs,
            ) = train_val_test_split_molecules(
                self.smiles,
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
    def __init__(self, number_features, target_scaler, num_epochs, hidden_size, learning_rate, fnn_layers, problem_type, shh=False, num_classes=None):
        super().__init__()
        self.problem_type = problem_type
        self.training_metric, self.overfitting_metric = fastprop.get_metrics(problem_type)

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
        self.num_classes = num_classes
        readout_size = num_classes if num_classes else self.target_scaler.n_features_in_
        self.readout = torch.nn.Linear(hidden_size, readout_size)

    def get_metrics(problem_type):
        match problem_type:
            case "regression":
                return "mse", "rmse"
            case "multilabel":
                return "bce", "auroc"
            case "multiclass":
                return "kldiv", "auroc"
            case "binary":
                return "bce", "accuracy"
            case _:
                raise RuntimeError(f"Unsupported problem type '{problem_type}'!")

    def forward(self, x):
        x = self.fnn.forward(x)
        x = self.readout(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        loss = self._machine_loss(batch, reduction="mean")
        self.log(f"train_{self.training_metric}_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, y, y_hat = self._machine_loss(batch, return_all=True)
        self.log(f"validation_{self.training_metric}_loss", loss)
        self._human_loss(y_hat.detach().cpu(), y.detach().cpu(), "validation")
        return loss

    def test_step(self, batch, batch_idx):
        loss, _, y, y_hat = self._machine_loss(batch, return_all=True)
        self.log(f"test_{self.training_metric}_loss", loss)
        self._human_loss(y_hat.detach().cpu(), y.detach().cpu(), "test")
        return loss

    def _machine_loss(self, batch, reduction="mean", return_all=False):
        x, y = batch
        y_hat = self.forward(x)
        if self.problem_type == "regression":
            loss = torch.nn.functional.mse_loss(y_hat, y, reduction=reduction)
        elif self.problem_type in {"multilabel", "binary"}:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y, reduction=reduction)
        else:
            y_pred = torch.softmax(y_hat, dim=1)
            loss = torch.nn.functional.kl_div(y_pred.log(), y, reduction="batchmean")
        if not return_all:
            return loss
        else:
            return loss, x, y, y_hat

    def _human_loss(self, pred, real, name):
        n_tasks = len(self.target_scaler.feature_names_in_)
        # outputs the performance in more human-interpretable manner
        # expensive so we don't do this during training
        if self.problem_type == "regression":
            rescaled_pred = self.target_scaler.inverse_transform(pred)
            rescaled_truth = self.target_scaler.inverse_transform(real)
            # mean absolute percentage error
            # sklearn asks for an array of weights, but it actually just passes through to np.average which
            # accepts weights of the same shape as the inputs
            per_task_mape = mape(rescaled_truth, rescaled_pred, multioutput="raw_values", sample_weight=rescaled_truth)
            if n_tasks == 1:
                self.log(f"{name}_wmape", np.mean(per_task_mape))
            else:
                self.log(f"{name}_mean_wmape", np.mean(per_task_mape))
                for target, value in zip(self.target_scaler.feature_names_in_, per_task_mape):
                    self.log(f"{name}_wmape_output_{target}", value)

            # mean absolute error
            all_loss = torch.nn.functional.l1_loss(torch.tensor(rescaled_pred), torch.tensor(rescaled_truth), reduction="none")
            per_task_loss = all_loss.numpy().mean(axis=0)
            if n_tasks == 1:
                self.log(f"{name}_l1", np.mean(per_task_loss))
            else:
                self.log(f"{name}_l1_avg", np.mean(per_task_loss))
                for target, value in zip(self.target_scaler.feature_names_in_, per_task_loss):
                    self.log(f"{name}_l1_output_{target}", value)

            # rmse
            per_task_loss = l2_error(rescaled_truth, rescaled_pred, multioutput="raw_values", squared=False)
            if n_tasks == 1:
                self.log(f"{name}_rmse", np.mean(per_task_loss))
            else:
                self.log(f"{name}_rmse_avg", np.mean(per_task_loss))
                for target, value in zip(self.target_scaler.feature_names_in_, per_task_loss):
                    self.log(f"{name}_rmse_output_{target}", value)
        else:
            if self.problem_type == "multilabel":
                pred_prob = torch.sigmoid(pred)
                auroc_score = multilabel_auroc(pred_prob, real.int(), n_tasks, "macro")
                self.log(f"{name}_auroc", auroc_score)
            elif self.problem_type == "multiclass":
                pred_prob = torch.nn.functional.softmax(pred, dim=1)
                real_class = torch.tensor(self.target_scaler.inverse_transform(real.int()), dtype=torch.long).squeeze()
                score = multiclass_auroc(pred, real_class, self.num_classes)
                self.log(f"{name}_auroc", score)
            elif self.problem_type == "binary":
                pred_prob = torch.sigmoid(pred)
                acc = binary_accuracy(pred_prob, real)
                self.log(f"{name}_accuracy", acc)
                f1 = f1_score(pred_prob, real, "binary")
                self.log(f"{name}_f1", f1)
                auroc_score = auroc(pred_prob, real.int(), task="binary")
                self.log(f"{name}_auroc", auroc_score)

    def on_train_epoch_end(self) -> None:
        # print progress every NUM_VALIDATION_CHECKS epochs, unless the total number of epochs is tiny (<NUM_VALIDATION_CHECKS)
        if self.trainer.max_epochs > NUM_VALIDATION_CHECKS and (self.trainer.current_epoch + 1) % (self.num_epochs // NUM_VALIDATION_CHECKS) == 0:
            if not self.shh:
                logger.info(f"Epoch [{self.trainer.current_epoch + 1}/{self.num_epochs}]")


def train_and_test(
    outdir,
    n_epochs,
    datamodule,
    model,
    patience=5,
    verbose=True,
    no_logs=False,
    enable_checkpoints=True,
):
    if not no_logs:
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
        enable_model_summary=verbose,
        logger=False if no_logs else [csv_logger, tensorboard_logger],
        log_every_n_steps=0 if no_logs else 1,
        enable_checkpointing=enable_checkpoints,
        check_val_every_n_epoch=1,
        callbacks=[
            EarlyStopping(
                monitor=f"validation_{model.training_metric}_loss",
                mode="min",
                verbose=False,
                patience=patience,
            )
        ],
    )

    t1_start = perf_counter()
    trainer.fit(
        model,
        datamodule,
    )
    t1_stop = perf_counter()
    if verbose:
        logger.info("Elapsed time during training: " + str(datetime.timedelta(seconds=t1_stop - t1_start)))
    validation_results = trainer.validate(model, datamodule, verbose=verbose)
    test_results = trainer.test(model, datamodule, verbose=verbose)
    return test_results, validation_results


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
        descs = load_saved_desc(precomputed)
    else:
        in_name = Path(input_file).stem
        # cached descriptors, which contains (1) cached (2) source filename (3) types of descriptors (4) timestamp when file was last touched
        cache_file = os.path.join(output_directory, "cached_" + in_name + "_" + descriptors + "_" + str(int(os.stat(input_file).st_ctime)) + ".csv")

        if os.path.exists(cache_file) and enable_cache:
            logger.info(f"Found cached descriptor data at {cache_file}, loading instead of recalculating.")
            descs = load_saved_desc(cache_file)
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


def _training_loop(
    number_repeats,
    number_features,
    target_scaler,
    number_epochs,
    hidden_size,
    learning_rate,
    fnn_layers,
    output_directory,
    datamodule,
    patience,
    problem_type,
    num_classes,
    hopt=False,
):
    # hopt disables some printing from fastprop, as well as the training loop, disables logging, and disables writing checkpoints
    all_test_results, all_validation_results = [], []
    for i in range(number_repeats):
        # reinitialize model
        model = fastprop(
            number_features,
            target_scaler,
            number_epochs,
            hidden_size,
            learning_rate,
            fnn_layers,
            problem_type,
            shh=hopt,
            num_classes=num_classes,
        )
        logger.info(f"Training model {i+1} of {number_repeats}")
        test_results, validation_results = train_and_test(
            output_directory,
            number_epochs,
            datamodule,
            model,
            patience=patience,
            verbose=not hopt,
            no_logs=hopt,
            enable_checkpoints=not hopt,
        )
        all_test_results.append(test_results[0])
        all_validation_results.append(validation_results[0])
        # resample for repeat trials
        if i + 1 < number_repeats:
            datamodule.random_seed += 1
            datamodule.fold_number += 1
            datamodule.setup()
            # ensure that the model is re-initialized on the next iteration
            del model

    validation_results_df = pd.DataFrame.from_records(all_validation_results)
    logger.info("Displaying validation results:\n%s", validation_results_df.describe().transpose().to_string())
    test_results_df = pd.DataFrame.from_records(all_test_results)
    logger.info("Displaying testing results:\n%s", test_results_df.describe().transpose().to_string())
    if number_repeats > 1:
        # regression results are reported per-task, so we might need to average a differently-named
        # column
        n_tasks = len(model.target_scaler.feature_names_in_)
        addendum = "_avg" if (problem_type == "regression" and n_tasks > 1) else ""
        ttest_result = ttest_ind(
            test_results_df[f"test_{model.overfitting_metric}" + addendum].to_numpy(),
            validation_results_df[f"validation_{model.overfitting_metric}" + addendum].to_numpy(),
        )
        if (p := ttest_result.pvalue) < 0.05:
            logger.warn(
                "Detected possible over/underfitting! 2-sided T-test between validation and testing"
                f" {model.overfitting_metric} yielded {p=:.3f}<0.05. Consider changing patience."
            )
        else:
            logger.info(f"2-sided T-test between validation and testing {model.overfitting_metric} yielded p value of {p=:.3f}>0.05.")
    else:
        logger.info("fastprop is unable to generate statistics to check for overfitting, consider increasing 'num_repeats' to at least 2.")
    return test_results_df, validation_results_df


def train_fastprop(
    output_directory,
    input_file,
    smiles_column,
    target_columns,
    descriptors="optimized",
    enable_cache=True,
    precomputed=None,
    rescaling=True,
    zero_variance_drop=False,
    colinear_drop=False,
    fnn_layers=2,
    hidden_size=1800,
    learning_rate=0.0001,
    batch_size=2048,
    number_epochs=100,
    number_repeats=1,
    problem_type="regression",
    checkpoint=None,
    train_size=0.8,
    val_size=0.1,
    test_size=0.1,
    sampler="random",
    random_seed=0,
    patience=5,
):
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
    torch.manual_seed(random_seed)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    targets, mols, smiles = load_from_csv(input_file, smiles_column, target_columns)
    descs = _get_descs(precomputed, input_file, output_directory, descriptors, enable_cache, mols)

    logger.info("Preprocessing data...")
    X, y, target_scaler = preprocess(descs, targets, rescaling, zero_variance_drop, colinear_drop, problem_type=problem_type)
    target_scaler.feature_names_in_ = target_columns
    num_classes = y.shape[1] if problem_type == "multiclass" else None
    logger.info("...done.")

    datamodule = ArbitraryDataModule(X, y, batch_size, random_seed, train_size, val_size, test_size, sampler, smiles=smiles)
    number_features = X.shape[1]
    _training_loop(
        number_repeats,
        number_features,
        target_scaler,
        number_epochs,
        hidden_size,
        learning_rate,
        fnn_layers,
        output_directory,
        datamodule,
        patience,
        problem_type,
        num_classes,
    )
