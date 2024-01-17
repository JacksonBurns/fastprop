import pandas as pd
from rdkit import Chem

import numpy as np

from fastprop.defaults import init_logger

logger = init_logger(__name__)


def load_from_csv(fpath, smiles_column, target_columns):
    # load a file of smiles and targets, returning rdkit mols and targets
    src = pd.read_csv(fpath)
    targets = src[target_columns].to_numpy()
    # keep everything 2D
    if len(target_columns) == 1:
        targets = targets.reshape(-1, 1)
    smiles: np.ndarray = src[smiles_column].to_numpy()
    logger.info("Generating RDKit molecules from SMILES.")
    rdkit_mols = np.array(list(Chem.MolFromSmiles(i) for i in smiles))
    error_mols_idxs = np.where(rdkit_mols == None)[0]  # noqa: E711
    for idx in error_mols_idxs:
        logger.warn(
            f"Unable to create RDKit molecule from SMILES string {smiles[idx]} (index {idx})."
            " Both the molecule and the target will be removed from the data."
        )
    if len(error_mols_idxs) > 0:
        # specify axis=0 to avoid changing dimensions (flattening)
        targets = np.delete(targets, error_mols_idxs, axis=0)
        rdkit_mols = np.delete(rdkit_mols, error_mols_idxs, axis=0)
        smiles = np.delete(smiles, error_mols_idxs)
        logger.warn(f"Removed {len(error_mols_idxs)} entries from the dataset ({100*len(error_mols_idxs)/len(rdkit_mols):.2f}% of the data).")
    return targets, rdkit_mols, smiles


def load_saved_desc(fpath):
    d = pd.read_csv(fpath, low_memory=False)
    d = d.apply(pd.to_numeric, errors="coerce")
    descs = d[d.columns[1:]].to_numpy(dtype=float)
    return descs
