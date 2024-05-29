from typing import List, Tuple

import numpy as np
import pandas as pd
import polars as pl

from fastprop.defaults import init_logger

logger = init_logger(__name__)


def read_input_csv(filepath: str, smiles_column: str, target_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Read a CSV file provided by the user.

    Args:
        filepath (str): Filepath.
        smiles_column (str): Column holding SMILES strings.
        target_columns (list[str]): Column(s) with target values.

    Raises:
        RuntimeError: Missing columns name(s).

    Returns:
        tuple[np.ndarray, np.ndarray]: Targets and SMILES strings.
    """
    src = pl.read_csv(filepath).to_pandas()
    try:
        targets = pd.concat([src.pop(x) for x in target_columns], axis=1).to_numpy()
        smiles = src.pop(smiles_column).to_numpy()
    except KeyError as ke:
        raise RuntimeError(f"Unable to find column(s) inside file {filepath}. Check spelling and file contents.") from ke

    # keep everything 2D
    if len(target_columns) == 1:
        targets = targets.reshape(-1, 1)

    return targets, smiles


def load_saved_descriptors(fpath: str) -> np.ndarray:
    """Loads cached descriptors as calculated previously, forcing missing to nan.

    Args:
        fpath (str): Filepath to CSV.

    Returns:
        np.ndarray: Loaded descriptors.
    """
    d = pl.read_csv(fpath, ignore_errors=True).to_pandas()
    d = d.apply(pd.to_numeric, errors="coerce")
    descs = d[d.columns[1:]].to_numpy(dtype=float)
    return descs
