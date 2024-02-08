"""
hopt.py

This file implements parallel hyperparameter optimization for fastprop using hyperopt.
"""

import logging
import os

import ray
import torch
import yaml

from fastprop.defaults import init_logger
from fastprop.fastprop_core import ArbitraryDataModule, _training_loop, fastprop
from fastprop.preprocessing import preprocess
from fastprop.utils import _get_descs, load_from_csv

tune, OptunaSearch = None, None
try:
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch
except ImportError as ie:
    hopt_error = ie

logger = init_logger(__name__)


CONFIG_FNAME = ".fastpropconfig"
_fpath = os.path.join(os.path.dirname(__file__), CONFIG_FNAME)
MODELS_PER_GPU, NUM_HOPT_TRIALS = 4, 64
if os.path.exists(_fpath):
    with open(_fpath) as file:
        cfg = yaml.safe_load(file)
        MODELS_PER_GPU = cfg.get("models_per_gpu", MODELS_PER_GPU)
        NUM_HOPT_TRIALS = cfg.get("num_hopt_trials", NUM_HOPT_TRIALS)


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
    patience=5,
    n_trials=NUM_HOPT_TRIALS,
    n_parallel=MODELS_PER_GPU,
):
    if tune is None or OptunaSearch is None:
        raise RuntimeError(
            "Unable to import hyperparameter optimization dependencies, please install fastprop[hopt]. Original error: " + str(hopt_error)
        )
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    torch.manual_seed(random_seed)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    targets, mols, smiles = load_from_csv(input_file, smiles_column, target_columns)
    descs = _get_descs(precomputed, input_file, output_directory, descriptors, enable_cache, mols)

    logger.info("Preprocessing data...")
    X, y, target_scaler = preprocess(descs, targets, rescaling, zero_variance_drop, colinear_drop, problem_type=problem_type)
    target_scaler.feature_names_in_ = target_columns
    num_classes = y.shape[1] if problem_type == "multiclass" else None
    logger.info("...done.")
    number_features = X.shape[1]

    datamodule = ArbitraryDataModule(X, y, batch_size, random_seed, train_size, val_size, test_size, sampler, smiles=smiles)
    return _hopt_loop(
        datamodule,
        problem_type,
        number_epochs,
        learning_rate,
        number_features,
        target_scaler,
        number_repeats,
        output_directory,
        random_seed,
        patience,
        num_classes,
        n_parallel,
        n_trials,
    )


def _hopt_loop(
    datamodule,
    problem_type,
    number_epochs,
    learning_rate,
    number_features,
    target_scaler,
    number_repeats,
    output_directory,
    random_seed,
    patience,
    num_classes,
    n_parallel,
    n_trials,
):
    datamodule_ref = ray.put(datamodule)

    _, metric = fastprop.get_metrics(problem_type)

    # driver code of optimization
    search_space = {
        "hidden_size": tune.choice(range(100, 3001, 100)),
        "fnn_layers": tune.choice(range(1, 6, 1)),
    }
    algo = OptunaSearch()
    tuner = tune.Tuner(
        tune.with_resources(
            lambda trial: objective(
                trial,
                datamodule_ref,
                number_epochs,
                learning_rate,
                number_features,
                target_scaler,
                number_repeats,
                output_directory,
                patience,
                problem_type,
                num_classes,
            ),
            # run n_parallel models at the same time (leave 20% for system)
            # don't specify cpus, and just let pl figure it out
            resources={"gpu": (1 - 0.20) / n_parallel},
        ),
        tune_config=tune.TuneConfig(
            metric=metric,
            mode="min" if problem_type == "regression" else "max",
            search_alg=algo,
            max_concurrent_trials=n_parallel,
            num_samples=n_trials,
        ),
        param_space=search_space,
    )

    results = tuner.fit()
    best = results.get_best_result().config
    logger.info(f"Best hyperparameters identified: {', '.join([key + ': ' + str(val) for key, val in best.items()])}")
    return best


def objective(
    trial,
    datamodule_ref,
    number_epochs,
    learning_rate,
    number_features,
    target_scaler,
    number_repeats,
    output_directory,
    patience,
    problem_type,
    num_classes,
) -> float:
    datamodule = ray.get(datamodule_ref)
    results_df, _ = _training_loop(
        number_repeats,
        number_features,
        target_scaler,
        number_epochs,
        trial["hidden_size"],
        learning_rate,
        trial["fnn_layers"],
        output_directory,
        datamodule,
        patience,
        problem_type,
        num_classes,
        hopt=True,
    )
    _, metric = fastprop.get_metrics(problem_type)
    if target_scaler.n_features_in_ == 1 or problem_type != "regression":
        return {metric: results_df.describe().at["mean", f"test_{metric}"]}
    else:
        return {metric: results_df.describe().at["mean", f"test_{metric}_avg"]}
