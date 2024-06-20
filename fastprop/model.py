import datetime
import os
from time import perf_counter
from typing import List, Literal, Optional, OrderedDict, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch import distributed

from fastprop.data import fastpropDataLoader, inverse_standard_scale, standard_scale
from fastprop.defaults import init_logger
from fastprop.metrics import SCORE_LOOKUP

logger = init_logger(__name__)


class ClampN(torch.nn.Module):
    def __init__(self, n: float) -> None:
        super().__init__()
        self.n = n

    def forward(self, batch: torch.Tensor):
        return torch.clamp(batch, min=-self.n, max=self.n)

    def extra_repr(self) -> str:
        return f"n={self.n}"


class fastprop(pl.LightningModule):
    def __init__(
        self,
        input_size: int = 1613,
        hidden_size: int = 1800,
        readout_size: int = 1,
        num_tasks: int = 1,
        learning_rate: float = 0.001,
        fnn_layers: int = 2,
        clamp_input: bool = False,
        problem_type: Literal["regression", "binary", "multiclass", "multilabel"] = "regression",
        target_names: List[str] = [],
        feature_means: Optional[torch.Tensor] = None,
        feature_vars: Optional[torch.Tensor] = None,
        target_means: Optional[torch.Tensor] = None,
        target_vars: Optional[torch.Tensor] = None,
    ):
        """fastprop descriptor-based regression FNN.

        Args:
            input_size (int, optional): Number of features. Defaults to 1613.
            hidden_size (int, optional): Height of hidden layers. Defaults to 1800.
            readout_size (int, optional): Number of output neurons. Defaults to 1.
            num_tasks (int, optional): Number of distinct tasks. Defaults to 1.
            learning_rate (float, optional): Learning rate for SGD. Defaults to 0.001.
            fnn_layers (int, optional): Number of hidden layers. Defaults to 2.
            clamp_input (bool, optional): Clamp inputs to +/-3 to aid extrapolation. Defaults to False.
            problem_type (Literal["regression", "binary", "multiclass", "multilabel"], optional): Type of training task. Defaults to "regression".
            target_names (list[str], optional): Names for targets in dataset, blank for simple integer names. Defaults to [].
            feature_means (Optional[torch.Tensor], optional): Means for scaling features in regression. Defaults to None.
            feature_vars (Optional[torch.Tensor], optional): Variances for scaling features in regression. Defaults to None.
            target_means (Optional[torch.Tensor], optional): Means for scaling targets during inference. Defaults to None.
            target_vars (Optional[torch.Tensor], optional): Means for scaling targets during inference. Defaults to None.
        """
        super().__init__()
        self.n_tasks = num_tasks
        self.register_buffer("feature_means", feature_means)
        self.register_buffer("feature_vars", feature_vars)
        self.register_buffer("target_means", target_means)
        self.register_buffer("target_vars", target_vars)
        self.problem_type = problem_type
        self.training_metric = fastprop.get_metric(problem_type)
        self.learning_rate = learning_rate
        self.target_names = target_names

        # fully-connected nn
        layers = OrderedDict()
        if clamp_input:
            layers["clamp"] = ClampN(3)
        for i in range(fnn_layers):
            layers[f"lin{i+1}"] = torch.nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
            if fnn_layers == 1 or i < (fnn_layers - 1):  # no output activation, unless single layer
                layers[f"act{i+1}"] = torch.nn.ReLU()
        self.fnn = torch.nn.Sequential(layers)
        self.readout = torch.nn.Linear(hidden_size, readout_size)

        self.save_hyperparameters()

    def configure_optimizers(self):
        """See https://lightning.ai/docs/pytorch/stable/common/optimization.html

        Returns:
            dict: Optimizer name and instance.
        """
        return {"optimizer": torch.optim.Adam(self.parameters(), lr=self.learning_rate)}

    def setup(self, stage=None):
        """Fill the target names if none were provided

        Args:
            stage (str, optional): Step of pipeline. Defaults to None.
        """
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
        """Returns the logits (i.e. without activation or scaling) for a given batch of features.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Logits.
        """
        x = self.fnn.forward(x)
        x = self.readout(x)
        return x

    def log(self, name, value, **kwargs):
        """Wrap the parent PyTorch Lightning log function to automatically detect DDP."""
        return super().log(name, value, sync_dist=distributed.is_initialized(), **kwargs)

    def training_step(self, batch, batch_idx):
        loss, _ = self._machine_loss(batch)
        self.log(f"train_{self.training_metric}_scaled_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat = self._machine_loss(batch)
        self.log(f"validation_{self.training_metric}_scaled_loss", loss)
        self._human_loss(y_hat, batch, "validation")
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat = self._machine_loss(batch)
        self.log(f"test_{self.training_metric}_scaled_loss", loss)
        self._human_loss(y_hat, batch, "test")
        return loss

    def predict_step(self, batch: Tuple[torch.Tensor]):
        """Applies feature scaling and appropriate activation function to a Tensor of descriptors.

        Args:
            batch (tuple[torch.Tensor]): Unscaled descriptors.

        Returns:
            torch.Tensor: Predictions.
        """
        descriptors = batch[0]
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
        """Reports the loss without undoing scaling for greater efficiency."""
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
        """Reports human-interpretable loss metrics.

        Args:
            pred (torch.Tensor): Network logits.
            batch (tuple[torch.Tensor, torch.Tensor]): Inputs and targets
            name (str): Name under which to log the results.
        """
        truth = batch[1]
        if self.problem_type == "regression" and self.target_means is not None and self.target_vars is not None:
            pred = inverse_standard_scale(pred, self.target_means, self.target_vars)
            truth = inverse_standard_scale(truth, self.target_means, self.target_vars)
        # get probability from logits
        elif self.problem_type in {"binary", "multilabel"}:
            pred = torch.sigmoid(pred)
        else:
            pred = torch.nn.functional.softmax(pred, dim=1)

        for metric in SCORE_LOOKUP[self.problem_type]:
            self.log(f"{name}_{metric.__name__}", metric(truth, pred, self.readout.out_features))
            if "multi" not in self.problem_type and self.n_tasks > 1:
                per_task_metric = metric(truth, pred, None, True)
                for target, value in zip(self.target_names, per_task_metric):
                    self.log(f"{name}_{target}_{metric.__name__}", value)


def train_and_test(
    output_directory: str,
    fastprop_model: fastprop,
    train_dataloader: fastpropDataLoader,
    val_dataloader: fastpropDataLoader,
    test_dataloader: fastpropDataLoader,
    number_epochs: int = 30,
    patience: int = 5,
    quiet: bool = False,
    **trainer_kwargs,
):
    """Run a single train/validate and test iteration.

    Args:
        output_directory (str): Filepath to write logs and checkpoints.
        fastprop_model (fastprop): fastprop LightningModule instance.
        train_dataloader (fastpropDataLoader): Training data.
        val_dataloader (fastpropDataLoader): Validation data.
        test_dataloader (fastpropDataLoader): Testing data.
        number_epochs (int, optional): Maximum number of epochs for training. Defaults to 30.
        patience (int, optional): Number of epochs for early stopping. Defaults to 5.
        quiet (bool, optional): Set True to disable some printing. Default to False.
        trainer_kwargs (dict, optional): Additional arguments to pass the the pl.Trainer

    Returns:
        list[dict]: Lightning model output.
    """
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
            monitor=f"validation_{fastprop_model.training_metric}_scaled_loss",
            mode="min",
            verbose=False,
            patience=patience,
        ),
        ModelCheckpoint(
            monitor=f"validation_{fastprop_model.training_metric}_scaled_loss",
            dirpath=os.path.join(output_directory, "checkpoints"),
            filename=f"repetition-{repetition_number}" + "-{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
            mode="min",
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=number_epochs,
        enable_progress_bar=not quiet,
        enable_model_summary=not quiet,
        logger=tensorboard_logger,
        log_every_n_steps=1,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        **trainer_kwargs,
    )

    t1_start = perf_counter()
    trainer.fit(fastprop_model, train_dataloader, val_dataloader)
    t1_stop = perf_counter()
    logger.info("Elapsed time during training: " + str(datetime.timedelta(seconds=t1_stop - t1_start)))
    ckpt_path = trainer.checkpoint_callback.best_model_path
    logger.info(f"Reloading best model from checkpoint file: {ckpt_path}")
    fastprop_model = fastprop_model.__class__.load_from_checkpoint(ckpt_path)
    validation_results = trainer.validate(fastprop_model, val_dataloader, verbose=False)
    test_results = trainer.test(fastprop_model, test_dataloader, verbose=False)
    validation_results_df = pd.DataFrame.from_records(validation_results, index=("value",))
    logger.info("Displaying validation results for repetition %d:\n%s", repetition_number, validation_results_df.transpose().to_string())
    test_results_df = pd.DataFrame.from_records(test_results, index=("value",))
    logger.info("Displaying validation results for repetition %d:\n%s", repetition_number, test_results_df.transpose().to_string())
    return test_results, validation_results
