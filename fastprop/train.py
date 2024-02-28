import datetime
import logging
import os

import numpy as np
import yaml
from lightning.pytorch import seed_everything

from fastprop import fastprop_core
from fastprop.defaults import _init_loggers, init_logger
from fastprop.preprocessing import preprocess

# choose the descriptor set absed on the args
from fastprop.utils import _get_descs, load_from_csv

logger = init_logger(__name__)


def train_fastprop(
    output_directory,
    input_file,
    smiles_column,
    target_columns,
    descriptors="optimized",
    enable_cache=True,
    precomputed=None,
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
    seed_everything(random_seed)
    targets, mols, smiles = load_from_csv(input_file, smiles_column, target_columns)
    descs = _get_descs(precomputed, input_file, output_directory, descriptors, enable_cache, mols, as_df=True)

    logger.info("Preprocessing features")
    X = preprocess(descs, zero_variance_drop, colinear_drop)

    # write information needed for feature importance, prediction, etc. into the checkpoints directory for later use
    with open(os.path.join(output_subdirectory, "checkpoints", "fastprop_config.yml"), "w") as file:
        file.write(
            yaml.dump(
                dict(smiles=smiles_column, targets=target_columns, descriptors=X.columns.to_list()),
                sort_keys=False,
            )
        )

    input_size = X.shape[1]
    readout_size = targets.shape[1] if problem_type != "multiclass" else int(np.max(targets) + 1)

    return fastprop_core._training_loop(
        number_repeats,
        number_epochs,
        input_size,
        hidden_size,
        readout_size,
        learning_rate,
        fnn_layers,
        output_subdirectory,
        patience,
        problem_type,
        train_size,
        val_size,
        test_size,
        sampler,
        smiles,
        X.to_numpy(),
        targets,
        target_columns,
        batch_size,
        random_seed,
        hopt=False,
    )
