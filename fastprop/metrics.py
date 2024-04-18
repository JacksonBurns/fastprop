"""Wrap a number of loss and score functions so that they can all be called
using the same arguments.
"""

import torch
from torchmetrics.functional.classification import (
    auroc,
    accuracy,
    f1_score,
    average_precision,
)
from torchmetrics.functional.regression import r2_score as tm_r2_score
from sklearn.metrics import mean_absolute_percentage_error as mape


def r2_score(truth: torch.Tensor, prediction: torch.Tensor, multitask: bool = False):
    return tm_r2_score(prediction, truth, multioutput="raw_values" if multitask else "uniform_average")


def mean_absolute_percentage_error_score(truth: torch.Tensor, prediction: torch.Tensor, multitask: bool = False):
    return mape(truth.numpy(force=True), prediction.numpy(force=True), multioutput="raw_values" if multitask else "uniform_average")


def weighted_mean_absolute_percentage_error_score(truth: torch.Tensor, prediction: torch.Tensor, multitask: bool = False):
    return mape(
        truth.numpy(force=True),
        prediction.numpy(force=True),
        multioutput="raw_values" if multitask else "uniform_average",
        sample_weight=truth.numpy(force=True),
    )


def mean_absolute_error_score(truth: torch.Tensor, prediction: torch.Tensor, multitask: bool = False):
    return torch.nn.functional.l1_loss(prediction, truth, reduction="none" if multitask else "mean")


def mean_squared_error_loss(truth: torch.Tensor, prediction: torch.Tensor, multitask: bool = False):
    return torch.nn.functional.mse_loss(prediction, truth, reduction="none" if multitask else "mean")


def root_mean_squared_error_loss(truth: torch.Tensor, prediction: torch.Tensor, multitask: bool = False):
    return torch.sqrt(mean_squared_error_loss(truth, prediction, multitask))


def binary_accuracy_score(truth: torch.Tensor, prediction: torch.Tensor, multitask: bool = False):
    return accuracy(prediction, truth, task="binary")


def binary_f1_score(truth: torch.Tensor, prediction: torch.Tensor, multitask: bool = False):
    return f1_score(prediction, truth, task="binary")


def binary_auroc(truth: torch.Tensor, prediction: torch.Tensor, multitask: bool = False):
    return auroc(prediction, truth.int(), task="binary")


def binary_average_precision(truth: torch.Tensor, prediction: torch.Tensor, multitask: bool = False):
    return average_precision(prediction, truth.int(), task="binary")


def multilabel_auroc(truth: torch.Tensor, prediction: torch.Tensor, num_labels: int):
    return auroc(prediction, truth.int(), task="multilabel", num_labels=num_labels)


def multiclass_auroc(truth: torch.Tensor, prediction: torch.Tensor, num_classes: int):
    return auroc(prediction, torch.argmax(truth, dim=1), task="multiclass", num_classes=num_classes)


SCORE_LOOKUP = {
    "regression": (
        r2_score,
        mean_absolute_percentage_error_score,
        weighted_mean_absolute_percentage_error_score,
        mean_absolute_error_score,
        root_mean_squared_error_loss,
    ),
    "binary": (
        binary_accuracy_score,
        binary_f1_score,
        binary_auroc,
        binary_average_precision,
    ),
    "multiclass": multiclass_auroc,
    "multilabel": multilabel_auroc,
}
