import pandas as pd

from rdkit import Chem


def load_from_csv(fpath, smiles_column, target_column):
    src = pd.read_csv(fpath)
    targets = src[target_column].to_numpy()
    smiles = src[smiles_column].to_numpy()
    rdkit_mols = list(Chem.MolFromSmiles(i) for i in smiles)
    return targets, rdkit_mols
