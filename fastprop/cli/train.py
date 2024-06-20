import datetime
import logging
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import psutil
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from scipy.stats import ttest_ind

from fastprop.data import (
    clean_dataset,
    fastpropDataLoader,
    fastpropDataset,
    split,
    standard_scale,
)
from fastprop.defaults import DESCRIPTOR_SET_LOOKUP, _init_loggers, init_logger
from fastprop.descriptors import get_descriptors
from fastprop.io import load_saved_descriptors, read_input_csv
from fastprop.model import fastprop, train_and_test


tune, OptunaSearch = None, None
try:
    import ray
    from ray import tune
    from ray.train.torch import enable_reproducibility
    from ray.tune.search.optuna import OptunaSearch
except ImportError as ie:
    hopt_error = ie

logger = init_logger(__name__)


NUM_HOPT_TRIALS = 16


@rank_zero_only
def _get_out_subdir_name(output_directory: str):
    return os.path.join(output_directory, f"fastprop_{int(datetime.datetime.utcnow().timestamp())}")


def train_fastprop(
    output_directory: str,
    input_file: str,
    smiles_column: str,
    target_columns: List[str],
    descriptor_set: str,
    enable_cache: bool,
    precomputed: str,
    fnn_layers: int,
    hidden_size: int,
    clamp_input: bool,
    learning_rate: float,
    batch_size: int,
    number_epochs: int,
    number_repeats: int,
    problem_type: str,
    train_size: float,
    val_size: float,
    test_size: float,
    sampler: str,
    random_seed: int,
    patience: int,
    hopt: bool = False,
):
    if hopt and (tune is None or OptunaSearch is None):
        raise RuntimeError(
            "Unable to import hyperparameter optimization dependencies, please install fastprop[hopt].\nOriginal error: " + str(hopt_error)
        )
    # setup logging and output directories
    output_subdirectory = _get_out_subdir_name(output_directory)
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(output_subdirectory, exist_ok=True)
    os.makedirs(os.path.join(output_subdirectory, "checkpoints"), exist_ok=True)
    _init_loggers(output_subdirectory)
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
    seed_everything(random_seed)

    # load the data
    targets, smiles = read_input_csv(input_file, smiles_column, target_columns)
    if precomputed:
        descriptors = load_saved_descriptors(precomputed)
    else:
        cache_file = os.path.join(
            output_directory, "cached_" + Path(input_file).stem + "_" + descriptor_set + "_" + str(int(os.stat(input_file).st_ctime)) + ".csv"
        )
        if os.path.exists(cache_file) and enable_cache:
            logger.info(f"Found cached descriptor data at {cache_file}, loading instead of recalculating.")
            descriptors = load_saved_descriptors(cache_file)
        else:
            targets, rdkit_mols, smiles = clean_dataset(targets, smiles)
            descriptors = get_descriptors(enable_cache and cache_file, DESCRIPTOR_SET_LOOKUP[descriptor_set], rdkit_mols)
            descriptors = descriptors.to_numpy(dtype=float)

    descriptors = torch.tensor(descriptors, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)

    if problem_type == "multiclass" and not (targets.size(dim=1) > 1):
        logger.info("One-hot encoding targets.")
        targets = torch.nn.functional.one_hot(targets.long()).squeeze()

    input_size = descriptors.shape[1]
    readout_size = targets.shape[1]
    n_tasks = 1 if problem_type in {"binary", "multiclass"} else readout_size

    logger.info(f"Run 'tensorboard --logdir {os.path.join(output_subdirectory, 'tensorboard_logs')}' to track training progress.")
    if not hopt:
        return _replicates(
            number_repeats,
            smiles,
            train_size,
            val_size,
            test_size,
            descriptors,
            targets,
            problem_type,
            random_seed,
            sampler,
            input_size,
            hidden_size,
            readout_size,
            n_tasks,
            learning_rate,
            fnn_layers,
            batch_size,
            clamp_input,
            number_epochs,
            patience,
            target_columns,
            output_subdirectory,
        )
    else:
        # driver code of optimization
        search_space = {
            "hidden_size": tune.choice(range(100, 3001, 100)),
            "fnn_layers": tune.choice(range(1, 5, 1)),
        }
        algo = OptunaSearch()
        smiles_ref = ray.put(smiles)
        targets_ref = ray.put(targets)
        descriptors_ref = ray.put(descriptors)
        metric = fastprop.get_metric(problem_type)
        tuner = tune.Tuner(
            tune.with_resources(
                lambda trial: _hopt_objective(
                    trial,
                    smiles_ref,
                    targets_ref,
                    descriptors_ref,
                    number_repeats,
                    train_size,
                    val_size,
                    test_size,
                    problem_type,
                    random_seed,
                    sampler,
                    input_size,
                    hidden_size,
                    readout_size,
                    n_tasks,
                    learning_rate,
                    fnn_layers,
                    batch_size,
                    clamp_input,
                    number_epochs,
                    patience,
                    target_columns,
                    output_subdirectory,
                ),
                resources={"gpu": 1, "cpu": psutil.cpu_count()},
            ),
            tune_config=tune.TuneConfig(
                metric=metric,
                mode="min",
                search_alg=algo,
                max_concurrent_trials=1,
                num_samples=NUM_HOPT_TRIALS,
            ),
            param_space=search_space,
        )
        results = tuner.fit()
        best = results.get_best_result().config
        logger.info(f"Best hyperparameters identified: {', '.join([key + ': ' + str(val) for key, val in best.items()])}")
        return best


def _replicates(
    number_repeats,
    smiles,
    train_size,
    val_size,
    test_size,
    descriptors,
    targets,
    problem_type,
    random_seed,
    sampler,
    input_size,
    hidden_size,
    readout_size,
    n_tasks,
    learning_rate,
    fnn_layers,
    batch_size,
    clamp_input,
    number_epochs,
    patience,
    target_columns,
    output_subdirectory,
):
    all_test_results, all_validation_results = [], []
    for replicate_number in range(number_repeats):
        logger.info(f"Training model {replicate_number+1} of {number_repeats} ({random_seed=})")

        descriptors_copy = descriptors.detach().clone()
        targets_copy = targets.detach().clone()
        # prepare the dataloaders
        train_indexes, val_indexes, test_indexes = split(smiles, random_seed, train_size, val_size, test_size, sampler)
        descriptors_copy[train_indexes], feature_means, feature_vars = standard_scale(descriptors_copy[train_indexes])
        descriptors_copy[val_indexes] = standard_scale(descriptors_copy[val_indexes], feature_means, feature_vars)
        descriptors_copy[test_indexes] = standard_scale(descriptors_copy[test_indexes], feature_means, feature_vars)

        if problem_type == "regression":
            targets_copy[train_indexes], target_means, target_vars = standard_scale(targets_copy[train_indexes, :])
            targets_copy[val_indexes] = standard_scale(targets_copy[val_indexes, :], target_means, target_vars)
            targets_copy[test_indexes] = standard_scale(targets_copy[test_indexes, :], target_means, target_vars)
        else:
            target_means = None
            target_vars = None

        train_dataloader = fastpropDataLoader(
            fastpropDataset(descriptors_copy[train_indexes], targets_copy[train_indexes]),
            shuffle=True,
            batch_size=batch_size,
        )
        val_dataloader = fastpropDataLoader(
            fastpropDataset(descriptors_copy[val_indexes], targets_copy[val_indexes]),
            batch_size=batch_size,
        )
        test_dataloader = fastpropDataLoader(
            fastpropDataset(descriptors_copy[test_indexes], targets_copy[test_indexes]),
            batch_size=batch_size,
        )

        # initialize the model and train/test
        model = fastprop(
            input_size,
            hidden_size,
            readout_size,
            n_tasks,
            learning_rate,
            fnn_layers,
            clamp_input,
            problem_type,
            target_columns,
            feature_means,
            feature_vars,
            target_means,
            target_vars,
        )
        logger.info("Model architecture:\n{%s}", str(model))
        test_results, validation_results = train_and_test(
            output_subdirectory,
            model,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            number_epochs,
            patience,
        )
        all_test_results.append(test_results[0])
        all_validation_results.append(validation_results[0])

        random_seed += 1
        # ensure that the model is re-instantiated
        del model

    validation_results_df = pd.DataFrame.from_records(all_validation_results)
    logger.info("Displaying validation results:\n%s", validation_results_df.describe().transpose().to_string())
    test_results_df = pd.DataFrame.from_records(all_test_results)
    logger.info("Displaying testing results:\n%s", test_results_df.describe().transpose().to_string())

    if number_repeats > 1:
        metric = fastprop.get_metric(problem_type)
        ttest_result = ttest_ind(
            test_results_df[f"test_{metric}_scaled_loss"].to_numpy(),
            validation_results_df[f"validation_{metric}_scaled_loss"].to_numpy(),
        )
        if (p := ttest_result.pvalue) < 0.05:
            logger.warn(
                "Detected possible over/underfitting! 2-sided T-test between validation and testing"
                f" {metric} yielded {p=:.3f}<0.05. Consider changing patience."
            )
        else:
            logger.info(f"2-sided T-test between validation and testing {metric} yielded p value of {p=:.3f}>0.05.")
    else:
        logger.info("fastprop is unable to generate statistics to check for overfitting, consider increasing 'num_repeats' to at least 2.")
    return validation_results_df, test_results_df


def _hopt_objective(
    trial,
    smiles_ref,
    targets_ref,
    descriptors_ref,
    number_repeats,
    train_size,
    val_size,
    test_size,
    problem_type,
    random_seed,
    sampler,
    input_size,
    hidden_size,
    readout_size,
    n_tasks,
    learning_rate,
    fnn_layers,
    batch_size,
    clamp_input,
    number_epochs,
    patience,
    target_columns,
    output_subdirectory,
) -> Dict[str, float]:
    descriptors = ray.get(descriptors_ref)
    targets = ray.get(targets_ref)
    smiles = ray.get(smiles_ref)
    enable_reproducibility(random_seed)
    validation_results_df, test_results_df = _replicates(
        number_repeats,
        smiles,
        train_size,
        val_size,
        test_size,
        descriptors,
        targets,
        problem_type,
        random_seed,
        sampler,
        input_size,
        trial["hidden_size"],
        readout_size,
        n_tasks,
        learning_rate,
        trial["fnn_layers"],
        batch_size,
        clamp_input,
        number_epochs,
        patience,
        target_columns,
        output_subdirectory,
    )
    metric = fastprop.get_metric(problem_type)
    return {metric: validation_results_df.describe().at["mean", f"validation_{metric}_scaled_loss"]}
