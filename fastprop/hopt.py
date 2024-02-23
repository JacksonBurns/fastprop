"""
hopt.py

Implements parallel hyperparameter optimization for fastprop using ray[tune] and Optuna.
"""

import logging
import os

import numpy as np
import yaml

from fastprop.defaults import init_logger
from fastprop.fastprop_core import _training_loop, fastprop
from fastprop.preprocessing import preprocess
from fastprop.utils import _get_descs, load_from_csv

tune, OptunaSearch = None, None
try:
    import ray
    from ray import tune
    from ray.train.torch import enable_reproducibility
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
            "Unable to import hyperparameter optimization dependencies, please install fastprop[hopt].\nOriginal error: " + str(hopt_error)
        )
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    targets, mols, smiles = load_from_csv(input_file, smiles_column, target_columns)
    descs = _get_descs(precomputed, input_file, output_directory, descriptors, enable_cache, mols, as_df=True)

    logger.info("Preprocessing features")
    X = preprocess(descs, zero_variance_drop, colinear_drop).to_numpy()

    input_size = X.shape[1]
    readout_size = targets.shape[1] if problem_type != "multiclass" else (np.max(targets[:, 1]) + 1)

    X_ref = ray.put(X)
    targets_ref = ray.put(targets)
    smiles_ref = ray.put(smiles)

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
                X_ref,
                targets_ref,
                smiles_ref,
                input_size,
                readout_size,
                number_epochs,
                learning_rate,
                number_repeats,
                patience,
                problem_type,
                train_size,
                val_size,
                test_size,
                sampler,
                target_columns,
                batch_size,
                random_seed,
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
    X_ref,
    targets_ref,
    smiles_ref,
    input_size,
    readout_size,
    number_epochs,
    learning_rate,
    number_repeats,
    patience,
    problem_type,
    train_size,
    val_size,
    test_size,
    sampler,
    target_columns,
    batch_size,
    random_seed,
) -> float:
    X = ray.get(X_ref)
    targets = ray.get(targets_ref)
    smiles = ray.get(smiles_ref)
    enable_reproducibility(random_seed)
    results_df, _ = _training_loop(
        number_repeats,
        number_epochs,
        input_size,
        trial["hidden_size"],
        readout_size,
        learning_rate,
        trial["fnn_layers"],
        None,  # output_directory ignored during hopt
        patience,
        problem_type,
        train_size,
        val_size,
        test_size,
        sampler,
        smiles,
        X,
        targets,
        target_columns,
        batch_size,
        random_seed,
        hopt=True,
    )
    _, metric = fastprop.get_metrics(problem_type)
    if readout_size == 1 or problem_type != "regression":
        return {metric: results_df.describe().at["mean", f"test_{metric}"]}
    else:
        return {metric: results_df.describe().at["mean", f"test_{metric}_avg"]}
