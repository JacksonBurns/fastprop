import datetime
import os
from time import perf_counter
from typing import OrderedDict, Literal

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch import distributed

from fastprop.defaults import init_logger
from fastprop.data import fastpropDataLoader, standard_scale, inverse_standard_scale
from fastprop.metrics import SCORE_LOOKUP

logger = init_logger(__name__)


class fastprop(pl.LightningModule):
    def __init__(
        self,
        input_size: int = 1613,
        hidden_size: int = 1800,
        readout_size: int = 1,
        num_tasks: int = 1,
        learning_rate: float = 0.001,
        fnn_layers: int = 2,
        problem_type: Literal["regression", "binary", "multiclass", "multilabel"] = "regression",
        target_names: list[str] = [],
        feature_means: torch.Tensor = None,
        feature_vars: torch.Tensor = None,
        target_means: torch.Tensor = None,
        target_vars: torch.Tensor = None,
    ):
        super().__init__()
        self.n_tasks = num_tasks
        self.register_buffer('feature_means', feature_means)
        self.register_buffer('feature_vars', feature_vars)
        self.register_buffer('target_means', target_means)
        self.register_buffer('target_vars', target_vars)
        self.problem_type = problem_type
        self.training_metric = fastprop.get_metric(problem_type)
        self.learning_rate = learning_rate
        self.target_names = target_names

        # fully-connected nn
        layers = OrderedDict(
            [
                ("lin1", torch.nn.Linear(input_size, hidden_size)),
                ("act1", torch.nn.ReLU()),
            ]
        )
        for i in range(fnn_layers - 1):
            layers[f"lin{i+2}"] = torch.nn.Linear(hidden_size, hidden_size)
            layers[f"act{i+2}"] = torch.nn.ReLU()
        self.fnn = torch.nn.Sequential(layers)
        self.readout = torch.nn.Linear(hidden_size, readout_size)

        self.save_hyperparameters()

    def configure_optimizers(self):
        return {"optimizer": torch.optim.Adam(self.parameters(), lr=self.learning_rate)}

    def setup(self, stage=None):
        if stage == "fit":
            if len(self.target_names) == 0:
                self.target_names = [f"task_{i}" for i in range(self.n_tasks)]

    def get_metric(problem_type: str):
        """Get the metric for training and early stopping based on the problem type.

        Args:
            problem_type (str): Regression, multilabel, multiclass, or binary.

        Raises:
            RuntimeError: Unsupported problem types

        Returns:
            str: names for the two metrics
        """
        if problem_type == "regression":
            return "mse"
        elif problem_type in {"multilabel", "binary"}:
            return "bce"
        elif problem_type == "multiclass":
            return "kldiv"
        else:
            raise RuntimeError(f"Unsupported problem type '{problem_type}'!")

    def forward(self, x):
        x = self.fnn.forward(x)
        x = self.readout(x)
        return x

    def log(self, name, value, **kwargs):
        return super().log(name, value, sync_dist=distributed.is_initialized(), **kwargs)

    def training_step(self, batch, batch_idx):
        loss, _ = self._machine_loss(batch)
        self.log(f"train_{self.training_metric}_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat = self._machine_loss(batch)
        self.log(f"validation_{self.training_metric}_loss", loss)
        self._human_loss(y_hat, batch, "validation")
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat = self._machine_loss(batch)
        self.log(f"test_{self.training_metric}_loss", loss)
        self._human_loss(y_hat, batch, "test")
        return loss

    def predict_step(self, descriptors: torch.Tensor):
        if self.feature_means is not None and self.feature_vars is not None:
            descriptors = standard_scale(descriptors, self.feature_means, self.feature_vars)
        with torch.inference_mode():
            logits = self.forward(descriptors)
        if self.problem_type == "regression":
            logits = inverse_standard_scale(logits, self.target_means, self.target_vars)
            return logits
        elif self.problem_type in {"multilabel", "binary"}:
            return torch.sigmoid(logits)
        elif self.problem_type == "multiclass":
            return torch.nn.functional.softmax(logits, dim=1)

    def _machine_loss(self, batch):
        # reports the scaled loss directly on the logits for computational efficiency
        x, y = batch
        y_hat = self.forward(x)
        if self.problem_type == "regression":
            loss = torch.nn.functional.mse_loss(y_hat, y, reduction="mean")
        elif self.problem_type in {"multilabel", "binary"}:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y, reduction="mean")
        else:
            y_pred = torch.softmax(y_hat, dim=1)
            loss = torch.nn.functional.kl_div(y_pred.log(), y.float(), reduction="batchmean")
        return loss, y_hat

    def _human_loss(self, pred, batch, name):
        truth = batch[1]
        if self.problem_type == "regression" and self.target_means is not None and self.target_vars is not None:
            pred = inverse_standard_scale(pred, self.target_means, self.target_vars)
            truth = inverse_standard_scale(truth, self.target_means, self.target_vars)
        # get probability from logits
        elif self.problem_type in {"binary", "multilabel"}:
            pred = torch.sigmoid(pred)
        else:
            pred = torch.nn.functional.softmax(pred, dim=1)

        if "multi" not in self.problem_type:
            for metric in SCORE_LOOKUP[self.problem_type]:
                self.log(f"{name}_{metric.__name__}", metric(truth, pred))
                if self.n_tasks > 1:
                    per_task_metric = metric(truth, pred, True)
                    for target, value in zip(self.target_names, per_task_metric):
                        self.log(f"{name}_{target}_{metric.__name__}", value)
        else:
            metric = SCORE_LOOKUP[self.problem_type]
            self.log(f"{name}_{metric.__name__}", metric(truth, pred, self.readout.out_features))


def train_and_test(
    output_directory: str,
    fastprop_model: fastprop,
    train_dataloader: fastpropDataLoader,
    val_dataloader: fastpropDataLoader,
    test_dataloader: fastpropDataLoader,
    number_epochs: int = 30,
    patience: int = 5,
):
    try:
        repetition_number = len(os.listdir(os.path.join(output_directory, "tensorboard_logs"))) + 1
    except FileNotFoundError:
        repetition_number = 1
    tensorboard_logger = TensorBoardLogger(
        output_directory,
        name="tensorboard_logs",
        version=f"repetition_{repetition_number}",
        default_hp_metric=False,
    )

    callbacks = [
        EarlyStopping(
            monitor=f"validation_{fastprop_model.training_metric}_loss",
            mode="min",
            verbose=False,
            patience=patience,
        ),
        ModelCheckpoint(
            monitor=f"validation_{fastprop_model.training_metric}_loss",
            dirpath=os.path.join(output_directory, "checkpoints"),
            filename=f"repetition-{repetition_number}" + "-{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
            mode="min",
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=number_epochs,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=tensorboard_logger,
        log_every_n_steps=1,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
    )

    t1_start = perf_counter()
    trainer.fit(fastprop_model, train_dataloader, val_dataloader)
    t1_stop = perf_counter()
    logger.info("Elapsed time during training: " + str(datetime.timedelta(seconds=t1_stop - t1_start)))
    validation_results = trainer.validate(fastprop_model, val_dataloader, verbose=False)
    test_results = trainer.test(fastprop_model, test_dataloader, verbose=False)
    validation_results_df = pd.DataFrame.from_records(validation_results, index=("value",))
    logger.info("Displaying validation results for repetition %d:\n%s", repetition_number, validation_results_df.transpose().to_string())
    test_results_df = pd.DataFrame.from_records(test_results, index=("value",))
    logger.info("Displaying validation results for repetition %d:\n%s", repetition_number, test_results_df.transpose().to_string())
    return test_results, validation_results
