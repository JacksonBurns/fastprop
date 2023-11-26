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

To reduce total calculation burden, could interpolatively sample dataset to get some small percent,
and then see which among those are transformed out, and then calc just the remaining for rest of
dataset. Still should make calc'ing all an option.
"""


# main driver function should accept args that align with those in the default training dict
# so that it can be called with train_fastprop(**args)

# need a function to write the model to a checkpoint file, as well as the preprocessing pipeline
# and the result scaler

# by default, also write the checkpoints at 4 intermediates
# during training

import os
import datetime
from pathlib import Path
import psutil
import warnings
from time import perf_counter

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from astartes import train_val_test_split

from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import Dataset as TorchDataset
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as l2_error


from fastprop.utils import load_from_morded_csv, load_cached_descs, load_from_csv
from .preprocessing import preprocess

from fastprop.utils import calculate_mordred_desciptors

# choose the descriptor set absed on the args
from fastprop.utils import SUBSET_947, ALL_2D
from fastprop.utils import mordred_descriptors_from_strings

descriptors_lookup = dict(
    optimized=SUBSET_947,
    all=ALL_2D,
)


torch.manual_seed(42)
warnings.filterwarnings(action="ignore", message=".*does not have many workers which may be a bottleneck.*")

# device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 1

# training configuration
TRAIN_SIZE = 0.6
VAL_SIZE = 0.2
TEST_SIZE = 1.0 - TRAIN_SIZE - VAL_SIZE
INIT_LEARNING_RATE = 5e-4

import logging

logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
logger = logging.getLogger("fastprop")
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
    def __init__(self, cleaned_data, targets, batch_size):
        super().__init__()
        self.data = cleaned_data
        self.targets = targets
        self.batch_size = batch_size
        self.train_idxs, self.val_idxs, self.test_idxs = None, None, None

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
            train_size=TRAIN_SIZE,
            val_size=VAL_SIZE,
            test_size=TEST_SIZE,
            sampler="random",
            random_state=42,
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
            collate_fn=_collate_fn if self.batch_size > 1 else None,
        )

    def train_dataloader(self):
        return self._init_dataloader(True, self.train_idxs)

    def val_dataloader(self):
        return self._init_dataloader(False, self.val_idxs)

    def test_dataloader(self):
        return self._init_dataloader(False, self.test_idxs)


class fastprop(pl.LightningModule):
    def __init__(self, number_features, target_scaler, num_epochs, hidden_size):
        super().__init__()
        # for saving human-readable accuracy metrics and predicting
        self.target_scaler = target_scaler
        self.num_epochs = num_epochs
        # growing interaction representation - break this into an encode function
        self.square_interactions = torch.nn.Linear(number_features, number_features, bias=False)
        self.dropout1 = torch.nn.Dropout(p=0.20)
        self.cubic_interactions = torch.nn.Linear(number_features, number_features, bias=False)
        self.dropout2 = torch.nn.Dropout(p=0.20)
        self.quartic_interactions = torch.nn.Linear(number_features, number_features, bias=False)
        self.dropout2 = torch.nn.Dropout(p=0.20)

        # fully-connected nn
        self.fc1 = torch.nn.Linear(number_features, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, hidden_size)
        self.readout = torch.nn.Linear(hidden_size, self.target_scaler.n_features_in_)
        self.act_fn1 = torch.nn.ReLU()
        self.act_fn2 = torch.nn.ReLU()
        self.act_fn3 = torch.nn.ReLU()

    def forward(self, x):
        
        # OPTION 1: directly send learned interactions to FNN
        # x = self.square_interactions(x)
        # x = self.cubic_interactions(x)
        # or do this more involved version:
        # x = self.dropout1(x)s
        # this is really more like quartic interactions (?)... can we concatenate
        # the input vector to the square interactions vector, and then learn
        # a transformation on that to get the cubic interactions
        # x = self.cubic_interactions(x)  # <-- !! performance is still great without this!
        # x = self.dropout2(x)  # <-- !! performance is still great without this!
        # make the number of these blocks a parameter, and then add n of them to a
        # nn.Sequential inside an encode() function (like how chemprop does it)
        # try adding more hidden layers
        # x = self.quartic_interactions(x)
        
        # OPTION 2: concatenate interaction layers, send to FNN
        # x is the zero-order 'interactions' (just the values themselves)
        # square_terms = self.square_interactions(x)
        # cube_terms = self.cubic_interactions(square_terms)
        # all_terms = torch.hstack([x, square_terms, cube_terms])
        
        # HAHA option 3 is to just send all the descriptors in directly, turns out
        # its really good anyway
        
        # FNN
        # x = self.fc1(all_terms)
        x = self.fc1(x)
        x = self.act_fn1(x)
        x = self.fc2(x)
        x = self.act_fn2(x)
        x = self.fc3(x)
        x = self.act_fn3(x)
        x = self.readout(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=INIT_LEARNING_RATE)
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = torch.nn.functional.mse_loss(y_hat, y, reduction='sum')
        self.log("train_mse_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("validation_mse_loss", loss, on_step=False, on_epoch=True)
        self._human_loss(y_hat.detach().cpu(), y.detach().cpu(), "validation")
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("test_mse_loss", loss, on_step=False, on_epoch=True)
        self._human_loss(y_hat.detach().cpu(), y.detach().cpu(), "test")
        return loss
    
    def _human_loss(self, pred, real, name):
        # outputs the performance in more human-interpretable manner
        # expensive so we don't do this during training
        rescaled_pred = self.target_scaler.inverse_transform(pred)
        rescaled_truth = self.target_scaler.inverse_transform(real)
        # mean absolute percentage error
        # sklearn asks for an array of weights, but it actually just passes through to np.average which
        # accepts weights of the same shape as the inputs
        per_task_mape = mape(rescaled_truth, rescaled_pred, multioutput="raw_values", sample_weight=rescaled_truth)
        self.log(f"unitful_{name}_mean_wmape", np.mean(per_task_mape))
        for target, value in zip(self.target_scaler.feature_names_in_, per_task_mape):
            self.log(f"unitful_{name}_wmape_output_{target}", value)

        # mean absolute error
        all_loss = torch.nn.functional.l1_loss(torch.tensor(rescaled_pred), torch.tensor(rescaled_truth), reduction='none')
        per_task_loss = all_loss.numpy().mean(axis=0)
        self.log(f"unitful_{name}_l1_avg", np.mean(per_task_loss))
        for target, value in zip(self.target_scaler.feature_names_in_, per_task_loss):
            self.log(f"unitful_{name}_l1_output_{target}", value)

        # rmse
        per_task_loss = l2_error(rescaled_truth, rescaled_pred, multioutput="raw_values")
        self.log(f"unitful_{name}_rmse_avg", np.mean(per_task_loss))
        for target, value in zip(self.target_scaler.feature_names_in_, per_task_loss):
            self.log(f"unitful_{name}_rmse_output_{target}", value)

    def on_train_epoch_end(self) -> None:
        if (self.trainer.current_epoch + 1) % (self.num_epochs // 10) == 0:
            print(f"Epoch [{self.trainer.current_epoch + 1}/{self.num_epochs}]")


def train_and_test(X, y, target_scaler, outdir, n_epochs, batch_size, hidden_size):
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
        check_val_every_n_epoch=n_epochs // 10,
    )

    model = fastprop(X.shape[1], target_scaler, n_epochs, hidden_size)
    # currently (early nov. '23) bugged
    # https://github.com/Lightning-AI/lightning/issues/17177
    # might need to use Python 3.10?
    compiled_model = None
    try:
        compiled_model = torch.compile(model)
    except Exception as e:
        print("Couldn't compiled: ", str(e))

    # if compiled_model:
    #     model = compiled_model

    datamodule = ArbitraryDataModule(
        X,
        y,
        batch_size
    )
    t1_start = perf_counter()
    trainer.fit(
        model,
        datamodule,
    )
    t1_stop = perf_counter()
    logger.info("Elapsed time during training: " + str(datetime.timedelta(seconds=t1_stop - t1_start)))
    trainer.test(
        model,
        datamodule,
    )


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
    interaction_layers=2,
    dropout_rate=0.2,
    fnn_layers=3,
    learning_rate=0.0001,
    batch_size=2048,
    number_epochs=1000,
    problem_type="regression",
    checkpoint=None,
):
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    targets, mols = load_from_csv(input_file, smiles_column, target_columns)
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
            descs = calculate_mordred_desciptors(d2c, mols, psutil.cpu_count(logical=False), "fast")
            # cache these
            if enable_cache:
                d = pd.DataFrame(descs)
                d.to_csv(cache_file)
                logger.info(f"Cached descriptors to {cache_file}.")

    logger.info("Preprocessing data...")
    X, y, target_scaler = preprocess(descs, targets, rescaling, zero_variance_drop, colinear_drop)
    target_scaler.feature_names_in_ = target_columns
    logger.info("...done.")

    train_and_test(X, y, target_scaler, output_directory, number_epochs, batch_size, 512)
