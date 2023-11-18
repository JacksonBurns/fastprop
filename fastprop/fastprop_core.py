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
import warnings
from time import perf_counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from astartes import train_val_test_split

from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import Dataset as TorchDataset
from torchmetrics.functional import mean_absolute_percentage_error as mape


torch.manual_seed(42)
warnings.filterwarnings(action="ignore", message=".*does not have many workers which may be a bottleneck.*")

# device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 1

# training configuration
BATCH_SIZE = 5096
TRAIN_SIZE = 0.6
VAL_SIZE = 0.2
TEST_SIZE = 1.0 - TRAIN_SIZE - VAL_SIZE
MAX_EPOCHS = 1000
INIT_LEARNING_RATE = 5e-5

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
    def __init__(self, cleaned_data, targets):
        super().__init__()
        self.data = cleaned_data
        self.targets = targets
        self.train_idxs, self.val_idxs, self.test_idxs = None, None, None

    def prepare_data(self):
        # cast input numpy data to a tensor
        if not isinstance(self.data, torch.Tensor):
            self.data = torch.tensor(self.data, dtype=torch.float32)
        if not isinstance(self.targets, torch.Tensor):
            self.targets = torch.tensor(self.targets.reshape(-1, 1), dtype=torch.float32)

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
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            collate_fn=_collate_fn if BATCH_SIZE > 1 else None,
        )

    def train_dataloader(self):
        return self._init_dataloader(True, self.train_idxs)

    def val_dataloader(self):
        return self._init_dataloader(False, self.val_idxs)

    def test_dataloader(self):
        return self._init_dataloader(False, self.test_idxs)


class fastprop(pl.LightningModule):
    def __init__(self, number_features, target_scaler):
        super().__init__()
        # for saving human-readable accuracy metrics and predicting
        self.target_scaler = target_scaler

        # growing interaction representation - break this into an encode function
        self.square_interactions = torch.nn.Linear(number_features, number_features, bias=False)
        self.dropout1 = torch.nn.Dropout(p=0.25)
        self.cubic_interactions = torch.nn.Linear(number_features, number_features, bias=False)
        self.dropout2 = torch.nn.Dropout(p=0.25)

        # fully-connected nn
        self.fc1 = torch.nn.Linear(number_features, 256)
        self.fc2 = torch.nn.Linear(256, 64)
        self.fc3 = torch.nn.Linear(64, 1)
        self.act_fn1 = torch.nn.Sigmoid()
        self.act_fn2 = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.square_interactions(x)
        x = self.dropout1(x)
        # this is really more like quartic interactions (?)... can we concatenate
        # the input vector to the square interactions vector, and then learn
        # a transformation on that to get the cubic interactions
        x = self.cubic_interactions(x)  # <-- !! performance is still great without this!
        x = self.dropout2(x)  # <-- !! performance is still great without this!
        # make the number of these blocks a parameter, and then add n of them to a
        # nn.Sequential inside an encode() function (like how chemprop does it)
        x = self.fc1(x)
        x = self.act_fn1(x)
        x = self.fc2(x)
        x = self.act_fn2(x)
        x = self.fc3(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=INIT_LEARNING_RATE)
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        # loss for the sgd
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)

        # loss for me TODO: TURN THIS OFF! EXPENSIVE AND SLOW TO CALL .cpu()
        rescaled_pred = self.target_scaler.inverse_transform(y_hat.detach().cpu())
        rescaled_truth = self.target_scaler.inverse_transform(y.detach().cpu())
        human_loss = mape(torch.tensor(rescaled_pred), torch.tensor(rescaled_truth))
        human_loss_2 = torch.nn.functional.l1_loss(torch.tensor(rescaled_pred), torch.tensor(rescaled_truth))
        self.log("unitful_training_mape", human_loss)
        self.log("unitful_training_l1", human_loss_2)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("validation_loss", loss, on_step=False, on_epoch=True)

        # loss for me TODO: TURN THIS OFF! EXPENSIVE AND SLOW TO CALL .cpu()
        rescaled_pred = self.target_scaler.inverse_transform(y_hat.detach().cpu())
        rescaled_truth = self.target_scaler.inverse_transform(y.detach().cpu())
        human_loss = mape(torch.tensor(rescaled_pred), torch.tensor(rescaled_truth))
        self.log("unitful_validation_mape", human_loss)
        human_loss_2 = torch.nn.functional.l1_loss(torch.tensor(rescaled_pred), torch.tensor(rescaled_truth))
        self.log("unitful_validation_l1", human_loss_2)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        # loss for me
        rescaled_pred = self.target_scaler.inverse_transform(y_hat.detach().cpu())
        rescaled_truth = self.target_scaler.inverse_transform(y.detach().cpu())
        human_loss = mape(torch.tensor(rescaled_pred), torch.tensor(rescaled_truth))
        self.log("unitful_test_mape", human_loss)
        human_loss_2 = torch.nn.functional.l1_loss(torch.tensor(rescaled_pred), torch.tensor(rescaled_truth))
        self.log("unitful_test_l1", human_loss_2)
        return loss

    def on_train_epoch_end(self) -> None:
        if (self.trainer.current_epoch + 1) % (MAX_EPOCHS // 10) == 0:
            print(f"Epoch [{self.trainer.current_epoch + 1}/{MAX_EPOCHS}]")


def train_and_test(X, y, target_scaler):
    csv_logger = CSVLogger(
        ".",
        "csv_logs_qm8",
    )
    tensorboard_logger = TensorBoardLogger(
        ".",
        "tensorboard_logs_qm8",
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=DEVICE,
        enable_progress_bar=True,
        logger=[csv_logger, tensorboard_logger],
        log_every_n_steps=1,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
    )

    model = fastprop(X.shape[1], target_scaler)
    # currently (early nov. '23) bugged
    # https://github.com/Lightning-AI/lightning/issues/17177
    # might need to use Python 3.10?
    compiled_model = None
    try:
        compiled_model = torch.compile(model)
    except Exception as e:
        print("Couldn't compiled: ", str(e))

    if compiled_model:
        model = compiled_model

    datamodule = ArbitraryDataModule(
        X,
        y,
    )
    t1_start = perf_counter()
    trainer.fit(
        model,
        datamodule,
    )
    t1_stop = perf_counter()
    print("Elapsed time during training:", t1_stop - t1_start)
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
    colinear_drop=True,
    interaction_layers=2,
    dropout_rate=0.2,
    fnn_layers=3,
    learning_rate=0.0001,
    batch_size=2048,
    problem_type="regression",
    checkpoint=None,
):
    # if cached
    # laod from cache
    from fastprop.utils import load_from_csv

    # put a logging statement before each call
    targets, mols = load_from_csv(input_file, smiles_column, target_columns)
    from fastprop.utils import calculate_mordred_desciptors

    # choose the descriptor set absed on the args
    from fastprop.utils import SUBSET_947
    from fastprop.utils import mordred_descriptors_from_strings

    d2c = mordred_descriptors_from_strings(SUBSET_947)

    descs = calculate_mordred_desciptors(d2c, mols, 4, "fast")
    # cache these
    from fastprop import preprocess

    X, y, target_scaler = preprocess(descs, targets)

    train_and_test(X, y, target_scaler)
