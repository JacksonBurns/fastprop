import os
import pickle

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem

from fastprop.defaults import init_logger
from fastprop.utils import calculate_mordred_desciptors
from fastprop.utils.select_descriptors import mordred_descriptors_from_strings

from .fastprop_core import fastprop

logger = init_logger(__name__)


def predict_fastprop(checkpoints_dir, smiles, input_file, output=None):
    """Prediction CLI.

    Loads a model and runs inference on the input.

    Args:
        checkpoints_dir (str): 'checkpoints' directory from a previous fastprop train.
        smiles (str or list[str]): SMILES strings for prediction.
        input_file (str): Input file containing only SMILES strings for prediction.
        output (str or None): Either save to a file or just print result.
    """
    if input_file:
        raise NotImplementedError("TODO")
    if type(smiles) is str:
        smiles = [smiles]
    checkpoint_dir_contents = os.listdir(checkpoints_dir)
    predict_args = None
    try:
        with open(os.path.join(checkpoints_dir, "predict_config.yml")) as file:
            predict_args = yaml.safe_load(file)
            print()
    except FileNotFoundError:
        logger.error("checkpoints directory is missing 'preprocess_config.yml'. Re-execute training.")

    descs = calculate_mordred_desciptors(
        mordred_descriptors_from_strings(predict_args["descriptors"]),
        [Chem.MolFromSmiles(i) for i in smiles],
        n_procs=0,  # ignored for "strategy='low-memory'"
        strategy="low-memory",
    )
    descs = pd.DataFrame(data=descs, columns=predict_args["descriptors"])

    for pickled_scaler in predict_args["feature_scalers"]:
        scaler = pickle.loads(pickled_scaler)
        descs = scaler.transform(descs)

    X = torch.tensor(descs.to_numpy(), dtype=torch.float32)
    all_models = []
    for checkpoint in checkpoint_dir_contents:
        if not checkpoint.endswith(".ckpt"):
            continue
        model = fastprop.load_from_checkpoint(
            os.path.join(checkpoints_dir, checkpoint),
            number_features=predict_args["number_features"],
            hidden_size=predict_args["hidden_size"],
            target_scaler=pickle.loads(predict_args["target_scaler"]),
            fnn_layers=predict_args["fnn_layers"],
            problem_type=predict_args["problem_type"],
            num_epochs=None,
            learning_rate=None,
        )
        model.eval()
        all_models.append(model)

    # axis: contents
    # 0: smiles
    # 1: predictions
    # 2: per-model
    all_predictions = np.stack([model.predict_step(X.to(model.device)) for model in all_models], axis=2)
    perf = np.mean(all_predictions, axis=2)
    err = np.std(all_predictions, axis=2)
    # interleave the columns of these arrays, thanks stackoverflow.com/a/75519265
    res = np.empty((len(perf), perf.shape[1] * 2), dtype=perf.dtype)
    res[:, 0::2] = perf
    res[:, 1::2] = err
    column_names = []
    for target in predict_args["targets"]:
        column_names.extend([target, target + "_stdev"])
    out = pd.DataFrame(res, columns=column_names, index=smiles)
    if output is None:
        print("\n", out)
    else:
        out.to_csv(output)
