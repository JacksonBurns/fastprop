"""
hopt.py

This file implements parallel hyperparameter optimization for fastprop using hyperopt.
"""
import logging
import os

import pandas as pd
import ray
import torch
import yaml

from fastprop.defaults import init_logger
from fastprop.fastprop_core import (
    ArbitraryDataModule,
    _get_descs,
    fastprop,
    train_and_test,
)
from fastprop.preprocessing import preprocess
from fastprop.utils import load_from_csv

tune, OptunaSearch = None, None
try:
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch
except ImportError as ie:
    raise RuntimeError("Unable to import hyperparameter optimization dependencies, please install fastprop[hopt]. Original error: " + str(ie))

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
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    torch.manual_seed(random_seed)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    targets, mols = load_from_csv(input_file, smiles_column, target_columns)
    descs = _get_descs(precomputed, input_file, output_directory, descriptors, enable_cache, mols)

    logger.info("Preprocessing data...")
    X, y, target_scaler = preprocess(descs, targets, rescaling, zero_variance_drop, colinear_drop)
    target_scaler.feature_names_in_ = target_columns
    logger.info("...done.")
    number_features = X.shape[1]

    datamodule = ArbitraryDataModule(X, y, batch_size, random_seed, train_size, val_size, test_size, sampler)
    datamodule_ref = ray.put(datamodule)

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
                random_seed,
                patience,
            ),
            # run n_parallel models at the same time (leave 20% for system)
            # don't specify cpus, and just let pl figure it out
            resources={"gpu": (1 - 0.20) / n_parallel},
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            search_alg=algo,
            max_concurrent_trials=n_parallel,
            num_samples=n_trials,
        ),
        param_space=search_space,
    )

    results = tuner.fit()
    print(results.get_best_result().config)


def objective(
    trial,
    datamodule_ref,
    number_epochs,
    learning_rate,
    number_features,
    target_scaler,
    number_repeats,
    output_directory,
    random_seed,
    patience,
) -> float:
    datamodule = ray.get(datamodule_ref)
    all_results = []
    for i in range(number_repeats):
        model = fastprop(number_features, target_scaler, number_epochs, trial["hidden_size"], learning_rate, trial["fnn_layers"], shh=True)
        results, _ = train_and_test(
            output_directory,
            number_epochs,
            datamodule,
            model,
            verbose=False,
            no_logs=True,
            enable_checkpoints=False,
            patience=patience,
        )
        all_results.append(results[0])
        random_seed += 1
        if i + 1 < number_repeats:
            random_seed += 1
            datamodule.random_seed = random_seed
            datamodule.setup()
    results_df = pd.DataFrame.from_records(all_results)
    if target_scaler.n_features_in_ == 1:
        return {"loss": results_df.describe().at["mean", "unitful_test_l1"]}
    else:
        return {"loss": results_df.describe().at["mean", "unitful_test_l1_avg"]}
