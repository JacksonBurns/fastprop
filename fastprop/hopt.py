import optuna

# TODO: use this in the future:
# from optuna.integration import PyTorchLightningPruningCallback
import torch
import os

from fastprop.fastprop_core import train_and_test, _get_descs, ArbitraryDataModule, fastprop
from fastprop.preprocessing import preprocess
from fastprop.defaults import _LOGGING_ARGS

import pandas as pd

from fastprop.utils import load_from_csv

import time

import logging
import contextlib

logging.basicConfig(**_LOGGING_ARGS)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def hopt_fastprop(
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
    n_trials=100,
):
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

    # driver code of optimization
    pruner = optuna.pruners.MedianPruner()

    storage = "sqlite:///fastprop_hopt_studies.db"
    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        study_name=f"fastprop_hopt_{int(time.time())}",
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(
            trial,
            datamodule,
            number_epochs,
            learning_rate,
            X.shape[1],
            target_scaler,
            number_repeats,
            output_directory,
            random_seed,
        ),
        n_trials=n_trials,
        timeout=None,
        n_jobs=-1,
        show_progress_bar=True,
    )

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def objective(
    trial: optuna.trial.Trial,
    datamodule,
    number_epochs,
    learning_rate,
    number_features,
    target_scaler,
    number_repeats,
    output_directory,
    random_seed,
) -> float:
    with contextlib.redirect_stdout(None):
        # We optimize the number of layers, hidden units in each layer and dropouts.
        fnn_layers = trial.suggest_int("n_layers", 1, 5)
        hidden_size = trial.suggest_int("hidden_size", 128, 2048)

        model = fastprop(number_features, target_scaler, number_epochs, hidden_size, learning_rate, fnn_layers)

        all_results = []
        for _ in range(number_repeats):
            results = train_and_test(output_directory, number_epochs, datamodule, model)
            all_results.append(results[0])
            random_seed += 1

        results_df = pd.DataFrame.from_records(all_results)
        if target_scaler.n_features_in_ == 1:
            return results_df.describe().at["mean", "unitful_test_l1"]
        else:
            return results_df.describe().at["mean", "unitful_test_l1_avg"]


if __name__ == "__main__":
    hopt_fastprop(
        output_directory="boiling_hopt",
        input_file="examples/alkane_boiling.csv",
        smiles_column="py2opsin_smiles",
        target_columns=["boiling_point"],
        descriptors="all",
        number_epochs=100,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        n_trials=48,
    )
