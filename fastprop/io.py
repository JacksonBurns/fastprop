import numpy as np
import pandas as pd

from fastprop.defaults import init_logger

logger = init_logger(__name__)


def read_input_csv(filepath: str, smiles_column: str, target_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    src = pd.read_csv(filepath)
    try:
        targets = pd.concat([src.pop(x) for x in target_columns], axis=1).to_numpy()
        smiles = src.pop(smiles_column).to_numpy()
    except KeyError as ke:
        raise RuntimeError(f"Unable to find column(s) inside file {filepath}. Check spelling and file contents.") from ke

    # keep everything 2D
    if len(target_columns) == 1:
        targets = targets.reshape(-1, 1)

    return targets, smiles


def load_saved_descriptors(fpath):
    # loads descriptors previously saved by fastprop, forces any non-numeric values (missing, strings, etc) to be nan.
    d = pd.read_csv(fpath, low_memory=False)
    d = d.apply(pd.to_numeric, errors="coerce")
    descs = d[d.columns[1:]].to_numpy(dtype=float)
    return descs
