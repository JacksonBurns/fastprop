import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from torch.utils.data import TensorDataset

from fastprop.data import clean_dataset, fastpropDataLoader
from fastprop.defaults import DESCRIPTOR_SET_LOOKUP, init_logger
from fastprop.descriptors import get_descriptors
from fastprop.io import load_saved_descriptors
from fastprop.model import fastprop

logger = init_logger(__name__)


def predict_fastprop(
    checkpoints_dir: str,
    smiles_strings: List[str],
    descriptor_set: str,
    smiles_file: Optional[str] = None,
    precomputed_descriptors: Optional[np.ndarray] = None,
    output: Optional[str] = None,
):
    if smiles_file is not None:
        if smiles_strings:
            raise RuntimeError("Specify either smiles_strings or smiles_file, not both.")
        smiles_strings = [s.strip() for s in open(smiles_file, "r").readlines()]

    # load the models
    if precomputed_descriptors is None:
        _, rdkit_mols, _ = clean_dataset(np.zeros((1, len(smiles_strings))), np.array(smiles_strings))
        descs = get_descriptors(cache_filepath=False, descriptors=DESCRIPTOR_SET_LOOKUP[descriptor_set], rdkit_mols=rdkit_mols)
        descs = descs.to_numpy(dtype=float)
    else:
        descs = load_saved_descriptors(precomputed_descriptors)

    all_models = []
    for checkpoint in os.listdir(checkpoints_dir):
        model = fastprop.load_from_checkpoint(os.path.join(checkpoints_dir, checkpoint))
        all_models.append(model)

    descs = torch.tensor(descs, dtype=torch.float32)
    predict_dataloader = fastpropDataLoader(TensorDataset(descs))
    # run inference
    # axis: contents
    # 0: smiles
    # 1: predictions
    # 2: per-model
    trainer = Trainer(logger=False)
    all_predictions = np.stack([torch.vstack(trainer.predict(model, predict_dataloader)).numpy(force=True) for model in all_models], axis=2)
    perf = np.mean(all_predictions, axis=2)
    err = np.std(all_predictions, axis=2)
    # interleave the columns of these arrays, thanks stackoverflow.com/a/75519265
    res = np.empty((len(perf), perf.shape[1] * 2), dtype=perf.dtype)
    res[:, 0::2] = perf
    res[:, 1::2] = err
    column_names = []
    for target in [f"task_{i}" for i in range(all_predictions.shape[1])]:
        column_names.extend([target, target + "_stdev"])
    out = pd.DataFrame(res, columns=column_names, index=smiles_strings)
    if output is None:
        print("\n", out)
    else:
        out.to_csv(output)
