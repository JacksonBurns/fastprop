"""
fastprop - molecular property prediction with simple descriptors

This file contains the core data handling and model definition functionality
for the fastprop framework.
"""

import datetime
import os
import warnings
from time import perf_counter
from typing import OrderedDict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from astartes import train_val_test_split
from astartes.molecules import train_val_test_split_molecules
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from scipy.stats import ttest_ind
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as l2_error
from torch.utils.data import Dataset as TorchDataset
from torchmetrics.functional.classification import (
    auroc,
    binary_accuracy,
    f1_score,
    multiclass_accuracy,
    multiclass_auroc,
    multilabel_auroc,
)
from torchmetrics.functional.regression import r2_score

from fastprop.defaults import init_logger

logger = init_logger(__name__)

warnings.filterwarnings(action="ignore", message=".*does not have many workers which may be a bottleneck.*")

# device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 1
# controls level of output
NUM_VALIDATION_CHECKS = 20


class ArbitraryDataset(TorchDataset):
    """Basic PyTorch dataset class.

    Args:
        TorchDataset (pytorch.Dataset): PyTorch's dataset class.
    """

    def __init__(self, data, targets):
        self.data = data
        self.length = len(targets)
        self.targets = targets

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


class ArbitraryDataModule(LightningDataModule):
    """Basic loader for PyTorch Lightning data modules.

    Args:
        LightningDataModule (lightning.DataModule): Lightning's parent DataModule class.
    """

    def __init__(self, cleaned_data, targets, batch_size, random_seed, train_size, val_size, test_size, sampler, smiles=None):
        """Load an arbitrary set of descriptors into the format expected by PyTorch.

        Args:
            cleaned_data (numpy.ndarray or torch.Tensor): Descriptors, already subjected to preprocessing (scaling, imputation, etc.)
            targets (numpy.ndarray or torch.Tensor): Scaled targets in the same order as the descriptors.
            batch_size (int): Samples/per batch - for small feature sets like in fastprop, set as high as possible.
            random_seed (int): Seed for RNG.
            train_size (float): Fraction of data for training.
            val_size (float): Fraction of data for validation.
            test_size (float): Fraction of data for test.
            sampler (str): Type of sampler to use, see astartes for a list of implemented samplers.
            smiles (list[str], optional): SMILES strings corresponding to the molecules for use in some samplers. Defaults to None.
        """
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
        logger.info(f"Sampling dataset with {self.sampler} sampler.")
        # see `benchmarks/quantumscents/quantumscents.py` for an example of overriding this method
        # to implement multiple folds
        split_kwargs = dict(
            train_size=self.train_size,
            val_size=self.val_size,
            test_size=self.test_size,
            sampler=self.sampler,
            random_state=self.random_seed,
            return_indices=True,
        )
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
                **split_kwargs,
            )
        else:
            (
                *_,
                self.train_idxs,
                self.val_idxs,
                self.test_idxs,
            ) = train_val_test_split_molecules(self.smiles, **split_kwargs)

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
        )

    def train_dataloader(self):
        return self._init_dataloader(True, self.train_idxs)

    def val_dataloader(self):
        return self._init_dataloader(False, self.val_idxs)

    def test_dataloader(self):
        return self._init_dataloader(False, self.test_idxs)


class fastprop(pl.LightningModule):
    """Core fastprop LightningModule

    Args:
        pl (pl.LightningModule): Parent Lightning Module class.
    """

    def __init__(self, number_features, target_scaler, num_epochs, hidden_size, learning_rate, fnn_layers, problem_type, shh=False, num_classes=None):
        """Core fastprop model.

        Args:
            number_features (int): Number of features in the input layer.
            target_scaler (sklearn scaler): Scaler used on target variables, used for reporting metrics in human-scale.
            num_epochs (int): Maximum allowed number of training epochs.
            hidden_size (int): Number of neurons in the hidden layers.
            learning_rate (float): Learning rate.
            fnn_layers (int): Number of layers in the FNN.
            problem_type (str): Problem type, i.e. regression, multiclass, multilabel, or binary.
            shh (bool, optional): Reduces some logging if true. Defaults to False.
            num_classes (int, optional): Number of classes for multiclass classification. Defaults to None.
        """
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
        if problem_type == "regression":
            return "mse", "rmse"
        elif problem_type == "multilabel":
            return "bce", "auroc"
        elif problem_type == "multiclass":
            return "kldiv", "auroc"
        elif problem_type == "binary":
            return "bce", "accuracy"
        else:
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

    def predict_step(self, X):
        with torch.no_grad():
            logits = self.forward(X)
        if self.problem_type == "regression":
            return self.target_scaler.inverse_transform(logits.detach().cpu())
        elif self.problem_type in {"multilabel", "binary"}:
            return torch.sigmoid(logits).detach().cpu()
        elif self.problem_type == "multiclass":
            return torch.nn.functional.softmax(logits, dim=1).detach().cpu()

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
            # report r2
            if n_tasks == 1:
                r2 = r2_score(torch.tensor(rescaled_pred), torch.tensor(rescaled_truth))
                self.log(f"{name}_r2", r2)

            # mean absolute percentage error
            per_task_mape = mape(rescaled_truth, rescaled_pred, multioutput="raw_values")
            if n_tasks == 1:
                self.log(f"{name}_mape", np.mean(per_task_mape))
            else:
                self.log(f"{name}_mean_mape", np.mean(per_task_mape))
                for target, value in zip(self.target_scaler.feature_names_in_, per_task_mape):
                    self.log(f"{name}_mape_output_{target}", value)
            # same, but weighted
            # sklearn asks for an array of weights, but it actually just passes through to np.average which
            # accepts weights of the same shape as the inputs
            per_task_wmape = mape(rescaled_truth, rescaled_pred, multioutput="raw_values", sample_weight=rescaled_truth)
            if n_tasks == 1:
                self.log(f"{name}_wmape", np.mean(per_task_wmape))
            else:
                self.log(f"{name}_mean_wmape", np.mean(per_task_wmape))
                for target, value in zip(self.target_scaler.feature_names_in_, per_task_wmape):
                    self.log(f"{name}_wmape_output_{target}", value)

            # mean absolute error
            all_loss = torch.nn.functional.l1_loss(torch.tensor(rescaled_pred), torch.tensor(rescaled_truth), reduction="none")
            per_task_loss = all_loss.numpy().mean(axis=0)
            if n_tasks == 1:
                self.log(f"{name}_l1", np.mean(per_task_loss))
                self.log(f"{name}_mdae", np.median(all_loss))
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
                score = multiclass_accuracy(pred, real_class, self.num_classes)
                self.log(f"{name}_accuracy", score)
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
    datamodule,
    model,
    patience=5,
    verbose=True,
    no_logs=False,
    enable_checkpoints=True,
):
    """Run the lightning trainer loop on a given model.

    Args:
        outdir (str): Output directory for log files.
        datamodule (ArbitraryDataModule): Lightning-style datamodule.
        model (LightingModule): fastprop model architecture itself.
        patience (int, optional): Maximum number of epochs to wait before stopping early. Defaults to 5.
        verbose (bool, optional): Set to false for less output. Defaults to True.
        no_logs (bool, optional): Set to true to disable logs. Defaults to False.
        enable_checkpoints (bool, optional): Set to false to disable checkpoint writing. Defaults to True.

    Returns:
        list[dict{metric: score}]: Output of lightning model.test and model.validate
    """
    if not no_logs:
        try:
            repetition_number = len(os.listdir(os.path.join(outdir, "csv_logs"))) + 1
        except FileNotFoundError:
            repetition_number = 1
        csv_logger = CSVLogger(outdir, name="csv_logs", version=f"repetition_{repetition_number}")
        tensorboard_logger = TensorBoardLogger(outdir, name="tensorboard_logs", version=f"repetition_{repetition_number}")

    callbacks = [
        EarlyStopping(
            monitor=f"validation_{model.training_metric}_loss",
            mode="min",
            verbose=False,
            patience=patience,
        )
    ]
    if enable_checkpoints:
        callbacks.append(
            ModelCheckpoint(
                monitor=f"validation_{model.training_metric}_loss",
                dirpath=os.path.join(outdir, "checkpoints"),
                filename=f"repetition-{repetition_number}" + "-{epoch:02d}-{val_loss:.2f}",
                save_top_k=1,
                mode="min",
            )
        )

    trainer = pl.Trainer(
        max_epochs=model.num_epochs,
        accelerator=DEVICE,
        enable_progress_bar=False,
        enable_model_summary=verbose,
        logger=False if no_logs else [csv_logger, tensorboard_logger],
        log_every_n_steps=0 if no_logs else 1,
        enable_checkpointing=enable_checkpoints,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
    )

    t1_start = perf_counter()
    trainer.fit(
        model,
        datamodule,
    )
    t1_stop = perf_counter()
    if verbose:
        logger.info("Elapsed time during training: " + str(datetime.timedelta(seconds=t1_stop - t1_start)))
    validation_results = trainer.validate(model, datamodule, verbose=False)
    test_results = trainer.test(model, datamodule, verbose=False)
    if verbose:
        validation_results_df = pd.DataFrame.from_records(validation_results, index=("value",))
        logger.info("Displaying validation results for repetition %d:\n%s", repetition_number, validation_results_df.transpose().to_string())
        test_results_df = pd.DataFrame.from_records(test_results, index=("value",))
        logger.info("Displaying validation results for repetition %d:\n%s", repetition_number, test_results_df.transpose().to_string())
    return test_results, validation_results


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
    """Outer training loop to perform repeats.

    Args:
        number_repeats (int): Number of repetitions.
        number_features (int): Number of features in the input layer.
        target_scaler (sklearn scaler): Scaler used on target variables, used for reporting metrics in human-scale.
        number_epochs (int): Maximum allowed number of training epochs.
        hidden_size (int): Number of neurons in the hidden layers.
        learning_rate (float): Learning rate.
        fnn_layers (int): Number of layers in the FNN.
        problem_type (str): Problem type, i.e. regression, multiclass, multilabel, or binary.
        num_classes (int, optional): Number of classes for multiclass classification. Defaults to None.
        output_directory (str): Output directory for log files.
        datamodule (pl.DataModule): Basic data module.
        patience (int): Maximum number of epochs to wait before stopping training early.
        hopt (bool, optional): Set to true when running hyperparameter optimization to turn off logs, logging, etc. Defaults to False.

    Returns:
        list[dict{metric: score}]: Output of lightning model.test and model.validate, one pair per repetition.
    """
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
            # ensure that the dataset is re-sampled with this new random seed
            # by removing the previously set indices
            datamodule.train_idxs = datamodule.test_idxs = datamodule.val_idxs = None
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
