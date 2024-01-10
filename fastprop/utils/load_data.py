import pandas as pd
from rdkit import Chem


def load_from_csv(fpath, smiles_column, target_columns):
    # load a file of smiles and targets, returning rdkit mols and targets
    src = pd.read_csv(fpath)
    targets = src[target_columns].to_numpy()
    if len(target_columns) == 1:
        targets = targets.reshape(-1, 1)
    smiles = src[smiles_column].to_numpy()
    rdkit_mols = list(Chem.MolFromSmiles(i) for i in smiles)
    return targets, rdkit_mols


def load_from_morded_csv(fpath):
    # read the csv file from mordred
    src = pd.read_csv(fpath)
    descs = src[src.columns[1:]].to_numpy(dtype=float)
    return descs


def load_cached_descs(fpath):
    # load descriptors cached by fastprop
    d = pd.read_csv(fpath, low_memory=False)
    d = d.apply(pd.to_numeric, errors="coerce")
    descs = d[d.columns[1:]].to_numpy(dtype=float)
    return descs
