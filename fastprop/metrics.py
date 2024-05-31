"""
This file wraps a number of loss functions and score functions so that
they can be called with a unified interface after being packed into a
dictionary.

Names of each function indicate its contents, and all of them call out
to external libraries with thorough documentation.
"""

import torch
from sklearn.metrics import mean_absolute_percentage_error as mape
from torchmetrics.functional.classification import (
    accuracy,
    auroc,
    average_precision,
    f1_score,
)
from torchmetrics.functional.regression import r2_score as tm_r2_score


def r2_score(truth: torch.Tensor, prediction: torch.Tensor, ignored: None, multitask: bool = False):
    return tm_r2_score(prediction, truth, multioutput="raw_values" if multitask else "uniform_average")


def mean_absolute_percentage_error_score(truth: torch.Tensor, prediction: torch.Tensor, ignored: None, multitask: bool = False):
    return mape(truth.numpy(force=True), prediction.numpy(force=True), multioutput="raw_values" if multitask else "uniform_average")


def weighted_mean_absolute_percentage_error_score(truth: torch.Tensor, prediction: torch.Tensor, ignored: None, multitask: bool = False):
    return mape(
        truth.numpy(force=True),
        prediction.numpy(force=True),
        multioutput="raw_values" if multitask else "uniform_average",
        sample_weight=truth.numpy(force=True),
    )


def mean_absolute_error_score(truth: torch.Tensor, prediction: torch.Tensor, ignored: None, multitask: bool = False):
    res = torch.nn.functional.l1_loss(prediction, truth, reduction="none" if multitask else "mean")
    if multitask:
        res = res.mean(dim=0)
    return res


def mean_squared_error_loss(truth: torch.Tensor, prediction: torch.Tensor, ignored: None, multitask: bool = False):
    res = torch.nn.functional.mse_loss(prediction, truth, reduction="none" if multitask else "mean")
    if multitask:
        res = res.mean(dim=0)
    return res


def root_mean_squared_error_loss(truth: torch.Tensor, prediction: torch.Tensor, ignored: None, multitask: bool = False):
    return torch.sqrt(mean_squared_error_loss(truth, prediction, ignored, multitask))


def binary_accuracy_score(truth: torch.Tensor, prediction: torch.Tensor, ignored: None, multitask: bool = False):
    return accuracy(prediction, truth, task="binary")


def binary_f1_score(truth: torch.Tensor, prediction: torch.Tensor, ignored: None, multitask: bool = False):
    return f1_score(prediction, truth, task="binary")


def binary_auroc(truth: torch.Tensor, prediction: torch.Tensor, ignored: None, multitask: bool = False):
    return auroc(prediction, truth.int(), task="binary")


def binary_average_precision(truth: torch.Tensor, prediction: torch.Tensor, ignored: None, multitask: bool = False):
    return average_precision(prediction, truth.int(), task="binary")


def multilabel_auroc(truth: torch.Tensor, prediction: torch.Tensor, num_labels: int):
    return auroc(prediction, truth.int(), task="multilabel", num_labels=num_labels)


def multilabel_average_precision(truth: torch.Tensor, prediction: torch.Tensor, num_labels: int):
    return average_precision(prediction, truth.int(), task="multilabel", num_labels=num_labels, average="micro")


def multilabel_f1_score(truth: torch.Tensor, prediction: torch.Tensor, num_labels: int):
    return f1_score(prediction, truth.int(), task="multilabel", num_labels=num_labels, average="micro")


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
    "multiclass": (multiclass_auroc,),
    "multilabel": (
        multilabel_auroc,
        multilabel_average_precision,
        multilabel_f1_score,
    ),
}
