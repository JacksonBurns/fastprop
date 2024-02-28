"""
fastprop - molecular property prediction with simple descriptors

This file contains the core data handling and model definition functionality
for the fastprop framework.
"""

import datetime
import os
import pickle
import warnings
from time import perf_counter
from types import SimpleNamespace
from typing import OrderedDict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from astartes import train_val_test_split
from astartes.molecules import train_val_test_split_molecules
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.stats import ttest_ind
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as l2_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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


def _mock_inverse_transform(x):
    return x


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


class fastprop(pl.LightningModule):
    """Core fastprop LightningModule

    Args:
        pl (pl.LightningModule): Parent Lightning Module class.
    """

    def __init__(
        self,
        num_epochs,
        input_size,
        hidden_size,
        readout_size,
        learning_rate,
        fnn_layers,
        problem_type,
        cleaned_data,
        targets,
        target_names,
        batch_size,
        random_seed,
        train_size,
        val_size,
        test_size,
        sampler,
        smiles,
        verbose=True,
    ):
        """Core fastprop model.

        Args:
            feature_scaler (sklearn scaler): Scaler used on feature variables, used for reporting metrics in human-scale.
            target_scaler (sklearn scaler): Scaler used on target variables, used for reporting metrics in human-scale.
            num_epochs (int): Maximum allowed number of training epochs.
            hidden_size (int): Number of neurons in the hidden layers.
            learning_rate (float): Learning rate.
            fnn_layers (int): Number of layers in the FNN.
            problem_type (str): Problem type, i.e. regression, multiclass, multilabel, or binary.
            verbose (bool, optional): Reduces some logging if true. Defaults to False.
            num_classes (int, optional): Number of classes for multiclass classification. Defaults to None.
            cleaned_data (numpy.ndarray or torch.Tensor): Descriptors, already subjected to preprocessing (dropping operations)
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
        # used for data preparation and training
        self.random_seed = random_seed
        self.problem_type = problem_type
        self.num_classes = readout_size if self.problem_type == "multiclass" else None

        # used only for training
        self.training_metric, self.overfitting_metric = fastprop.get_metrics(problem_type)
        self.verbose = verbose
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
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

        # used only for data preparation
        self.data = cleaned_data
        self.targets = targets
        self.batch_size = batch_size
        self.sampler = sampler
        self.smiles = smiles
        self.train_size, self.val_size, self.test_size = train_size, val_size, test_size
        self.train_idxs, self.val_idxs, self.test_idxs = None, None, None
        # either using folds or random sampling, both of these will be incremented by the training loop
        self.fold_number = 0
        self.target_names = target_names

        # add derived attributes to the hparams attribute manually so that they are saved with checkpoints
        self.hparams["num_classes"] = self.num_classes
        self.save_hyperparameters(ignore=("cleaned_data", "targets", "smiles"))

    def _split(self):
        logger.info(f"Sampling dataset with {self.sampler} sampler.")
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
                self.data,
                # flatten 1D targets
                self.targets.flatten() if self.targets.shape[1] == 1 else self.targets,
                **split_kwargs,
            )
        else:
            (
                *_,
                self.train_idxs,
                self.val_idxs,
                self.test_idxs,
            ) = train_val_test_split_molecules(self.smiles, **split_kwargs)

    def setup(self, stage=None):
        if stage == "fit":
            self._split()
            logger.info("Imputing and rescaling input features.")
            # it is possible that the randomly chosen training set can have all entries missing a descriptor
            # even if other members of the dataset are valued! They will be zeroed out
            self.mean_imputer = SimpleImputer(missing_values=np.nan, strategy="mean", keep_empty_features=True)
            self.mean_imputer.fit(self.data[self.train_idxs, :])
            self.data = self.mean_imputer.transform(self.data)

            # scale each column to unit variance
            self.feature_scaler = StandardScaler()
            self.feature_scaler.fit(self.data[self.train_idxs, :])
            self.data = self.feature_scaler.transform(self.data)

            # mock the scaler object for classification tasks
            self.target_scaler = SimpleNamespace(n_features_in_=self.targets.shape[1], inverse_transform=_mock_inverse_transform)
            if self.problem_type == "regression":
                logger.info("Rescaling targets.")
                self.target_scaler = StandardScaler()
                self.target_scaler.fit(self.targets[self.train_idxs, :])
                self.targets = self.target_scaler.transform(self.targets)
            elif self.problem_type == "multiclass":
                logger.info("One-hot encoding target values.")
                self.target_scaler = OneHotEncoder(sparse_output=False)
                self.targets = self.target_scaler.fit_transform(self.targets)

            # finally, cast everything as needed
            self.data = torch.tensor(self.data, dtype=torch.float32)
            self.targets = torch.tensor(self.targets, dtype=torch.float32)

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
        X = self.mean_imputer.transform(X)
        X = self.feature_scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
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
        n_tasks = len(self.target_names)
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
            per_task_mape = np.abs(mape(rescaled_truth, rescaled_pred, multioutput="raw_values"))
            if n_tasks == 1:
                self.log(f"{name}_mape", np.mean(per_task_mape))
            else:
                self.log(f"{name}_mean_mape", np.mean(per_task_mape))
                for target, value in zip(self.target_names, per_task_mape):
                    self.log(f"{name}_mape_output_{target}", value)
            # same, but weighted
            # sklearn asks for an array of weights, but it actually just passes through to np.average which
            # accepts weights of the same shape as the inputs
            per_task_wmape = np.abs(mape(rescaled_truth, rescaled_pred, multioutput="raw_values", sample_weight=rescaled_truth))
            if n_tasks == 1:
                self.log(f"{name}_wmape", np.mean(per_task_wmape))
            else:
                self.log(f"{name}_mean_wmape", np.mean(per_task_wmape))
                for target, value in zip(self.target_names, per_task_wmape):
                    self.log(f"{name}_wmape_output_{target}", value)

            # mean absolute error
            all_loss = torch.nn.functional.l1_loss(torch.tensor(rescaled_pred), torch.tensor(rescaled_truth), reduction="none")
            per_task_loss = all_loss.numpy().mean(axis=0)
            if n_tasks == 1:
                self.log(f"{name}_l1", np.mean(per_task_loss))
                self.log(f"{name}_mdae", np.median(all_loss))
            else:
                self.log(f"{name}_l1_avg", np.mean(per_task_loss))
                for target, value in zip(self.target_names, per_task_loss):
                    self.log(f"{name}_l1_output_{target}", value)

            # rmse
            per_task_loss = l2_error(rescaled_truth, rescaled_pred, multioutput="raw_values", squared=False)
            if n_tasks == 1:
                self.log(f"{name}_rmse", np.mean(per_task_loss))
            else:
                self.log(f"{name}_rmse_avg", np.mean(per_task_loss))
                for target, value in zip(self.target_names, per_task_loss):
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
        if self.num_epochs > NUM_VALIDATION_CHECKS and (self.trainer.current_epoch + 1) % (self.num_epochs // NUM_VALIDATION_CHECKS) == 0:
            if self.verbose:
                logger.info(f"Epoch [{self.trainer.current_epoch + 1}/{self.num_epochs}]")


def train_and_test(
    outdir,
    lightning_module,
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
            repetition_number = len(os.listdir(os.path.join(outdir, "tensorboard_logs"))) + 1
        except FileNotFoundError:
            repetition_number = 1
        tensorboard_logger = TensorBoardLogger(outdir, name="tensorboard_logs", version=f"repetition_{repetition_number}", default_hp_metric=False)

    callbacks = [
        EarlyStopping(
            monitor=f"validation_{lightning_module.training_metric}_loss",
            mode="min",
            verbose=False,
            patience=patience,
        )
    ]
    if enable_checkpoints:
        callbacks.append(
            ModelCheckpoint(
                monitor=f"validation_{lightning_module.training_metric}_loss",
                dirpath=os.path.join(outdir, "checkpoints"),
                filename=f"repetition-{repetition_number}" + "-{epoch:02d}-{val_loss:.2f}",
                save_top_k=1,
                mode="min",
            )
        )

    trainer = pl.Trainer(
        max_epochs=lightning_module.num_epochs,
        accelerator=DEVICE,
        enable_progress_bar=False,
        enable_model_summary=verbose,
        logger=False if no_logs else tensorboard_logger,
        log_every_n_steps=0 if no_logs else 1,
        enable_checkpointing=enable_checkpoints,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        deterministic=True,
    )

    t1_start = perf_counter()
    trainer.fit(lightning_module)
    t1_stop = perf_counter()
    if verbose:
        logger.info("Elapsed time during training: " + str(datetime.timedelta(seconds=t1_stop - t1_start)))
    validation_results = trainer.validate(lightning_module, verbose=False)
    test_results = trainer.test(lightning_module, verbose=False)
    if verbose:
        validation_results_df = pd.DataFrame.from_records(validation_results, index=("value",))
        logger.info("Displaying validation results for repetition %d:\n%s", repetition_number, validation_results_df.transpose().to_string())
        test_results_df = pd.DataFrame.from_records(test_results, index=("value",))
        logger.info("Displaying validation results for repetition %d:\n%s", repetition_number, test_results_df.transpose().to_string())
    return test_results, validation_results


def _training_loop(
    number_repeats,
    number_epochs,
    input_size,
    hidden_size,
    readout_size,
    learning_rate,
    fnn_layers,
    output_directory,
    patience,
    problem_type,
    train_size,
    val_size,
    test_size,
    sampler,
    smiles,
    cleaned_data,
    targets,
    target_names,
    batch_size,
    random_seed,
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
    if not hopt:
        logger.info(f"Run 'tensorboard --logdir {os.path.join(output_directory, 'tensorboard_logs')}' to track training progress.")
    # hopt disables some printing from fastprop, as well as the training loop, disables logging, and disables writing checkpoints
    all_test_results, all_validation_results = [], []
    for i in range(number_repeats):
        # reinitialize model
        lightning_module = fastprop(
            number_epochs,
            input_size,
            hidden_size,
            readout_size,
            learning_rate,
            fnn_layers,
            problem_type,
            cleaned_data,
            targets,
            target_names,
            batch_size,
            random_seed,
            train_size,
            val_size,
            test_size,
            sampler,
            smiles,
            verbose=not hopt,
        )
        logger.info(f"Training model {i+1} of {number_repeats}")
        test_results, validation_results = train_and_test(
            output_directory,
            lightning_module,
            patience=patience,
            verbose=not hopt,
            no_logs=hopt,
            enable_checkpoints=not hopt,
        )
        all_test_results.append(test_results[0])
        all_validation_results.append(validation_results[0])
        # save the scalers for later use
        if not hopt:
            with open(os.path.join(output_directory, "checkpoints", f"repetition_{i+1}_scalers.yml"), "w") as file:
                file.write(
                    yaml.dump(
                        dict(
                            target_scaler=pickle.dumps(lightning_module.target_scaler),
                            mean_imputer=pickle.dumps(lightning_module.mean_imputer),
                            feature_scaler=pickle.dumps(lightning_module.feature_scaler),
                        ),
                        sort_keys=False,
                    )
                )

        # resample for repeat trials
        if i + 1 < number_repeats:
            random_seed += 1
            # ensure that the model is re-initialized on the next iteration
            del lightning_module

    validation_results_df = pd.DataFrame.from_records(all_validation_results)
    logger.info("Displaying validation results:\n%s", validation_results_df.describe().transpose().to_string())
    test_results_df = pd.DataFrame.from_records(all_test_results)
    logger.info("Displaying testing results:\n%s", test_results_df.describe().transpose().to_string())
    if number_repeats > 1:
        # regression results are reported per-task, so we might need to average a differently-named
        # column
        n_tasks = len(lightning_module.target_names)
        addendum = "_avg" if (problem_type == "regression" and n_tasks > 1) else ""
        ttest_result = ttest_ind(
            test_results_df[f"test_{lightning_module.overfitting_metric}" + addendum].to_numpy(),
            validation_results_df[f"validation_{lightning_module.overfitting_metric}" + addendum].to_numpy(),
        )
        if (p := ttest_result.pvalue) < 0.05:
            logger.warn(
                "Detected possible over/underfitting! 2-sided T-test between validation and testing"
                f" {lightning_module.overfitting_metric} yielded {p=:.3f}<0.05. Consider changing patience."
            )
        else:
            logger.info(f"2-sided T-test between validation and testing {lightning_module.overfitting_metric} yielded p value of {p=:.3f}>0.05.")
    else:
        logger.info("fastprop is unable to generate statistics to check for overfitting, consider increasing 'num_repeats' to at least 2.")
    return test_results_df, validation_results_df
