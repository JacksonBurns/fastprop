import datetime
import logging
import os
import pickle

import torch
import yaml

from fastprop import ArbitraryDataModule, fastprop_core
from fastprop.defaults import _init_loggers, init_logger
from fastprop.preprocessing import preprocess

# choose the descriptor set absed on the args
from fastprop.utils import _get_descs, linear_baseline, load_from_csv

logger = init_logger(__name__)


def train_fastprop(
    output_directory,
    input_file,
    smiles_column,
    target_columns,
    descriptors="optimized",
    enable_cache=True,
    precomputed=None,
    rescaling=True,
    zero_variance_drop=False,
    colinear_drop=False,
    fnn_layers=2,
    hidden_size=1800,
    learning_rate=0.0001,
    batch_size=2048,
    number_epochs=100,
    number_repeats=1,
    problem_type="regression",
    checkpoint=None,
    train_size=0.8,
    val_size=0.1,
    test_size=0.1,
    sampler="random",
    random_seed=0,
    patience=5,
):
    """Driver function for automatic training.

    See the fastprop documentation or CLI --help for details on each argument.
    """
    if checkpoint is not None:
        raise RuntimeError("TODO: Restarting from checkpoint not currently supported. Exiting.")
    # make output directories
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    output_subdirectory = os.path.join(output_directory, f"fastprop_{int(datetime.datetime.utcnow().timestamp())}")
    os.mkdir(output_subdirectory)
    os.mkdir(os.path.join(output_subdirectory, "checkpoints"))
    _init_loggers(output_subdirectory)
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
    torch.manual_seed(random_seed)
    targets, mols, smiles = load_from_csv(input_file, smiles_column, target_columns)
    descs = _get_descs(precomputed, input_file, output_directory, descriptors, enable_cache, mols, as_df=True)

    logger.info("Preprocessing data...")
    X, y, target_scaler, feature_scalers = preprocess(
        descs,
        targets,
        rescaling,
        zero_variance_drop,
        colinear_drop,
        problem_type=problem_type,
        return_X_scalers=True,
    )
    target_scaler.feature_names_in_ = target_columns
    num_classes = y.shape[1] if problem_type == "multiclass" else None
    logger.info("...done.")

    linear_baseline(problem_type, random_seed, number_repeats, sampler, train_size, val_size, test_size, X.to_numpy(), y, smiles, target_scaler)

    datamodule = ArbitraryDataModule(X.to_numpy(), y, batch_size, random_seed, train_size, val_size, test_size, sampler, smiles=smiles)
    number_features = X.shape[1]

    # write information needed for feature importance, prediction, etc. into the checkpoints directory for later use
    with open(os.path.join(output_subdirectory, "checkpoints", "fastprop_config.yml"), "w") as file:
        file.write(
            yaml.dump(
                dict(
                    rescaling=rescaling,
                    zero_variance_drop=zero_variance_drop,
                    colinear_drop=colinear_drop,
                    problem_type=problem_type,
                    number_features=number_features,
                    hidden_size=hidden_size,
                    fnn_layers=fnn_layers,
                    smiles=smiles_column,
                    targets=target_columns,
                    descriptors=X.columns.to_list(),
                    target_scaler=pickle.dumps(target_scaler),
                    feature_scalers=[pickle.dumps(i) for i in feature_scalers],
                ),
                sort_keys=False,
            )
        )

    return fastprop_core._training_loop(
        number_repeats,
        number_features,
        target_scaler,
        number_epochs,
        hidden_size,
        learning_rate,
        fnn_layers,
        output_subdirectory,
        datamodule,
        patience,
        problem_type,
        num_classes,
    )
