import datetime
from importlib.metadata import version
from time import perf_counter
from typing import List, Union

import numpy as np
import pandas as pd
import psutil
from mordred import Calculator, Descriptor, descriptors, get_descriptors_in_module
from packaging.version import Version
from rdkit.Chem import rdchem

from fastprop.defaults import init_logger

logger = init_logger(__name__)


def _descriptor_names_to_mordred_class(string_list: List[str] = [], include_3d: bool = False):
    """Wraps a weird quirk of the mordred calculator library - descriptors must be passed
    as their corresponding class, not their name string.

    Args:
        string_list (list[str], optional): Names of descriptors, leave empty to get all.
        include_3d (bool, optional): Set True to ignore descriptors needing 3D coordinates. Defaults to False.
    """
    out = []
    string_set = set(string_list)
    if (input_length := len(string_list)) != (set_length := len(string_set)):
        raise RuntimeError(f"Input contains {input_length - set_length} duplicate entires.")
    __version__ = Version(version("mordredcommunity"))
    for mdl in descriptors.all:
        for Desc in get_descriptors_in_module(mdl, submodule=False):
            for desc in Desc.preset(__version__):
                if not string_list or desc.__str__() in string_list:
                    if desc.require_3D and not include_3d:
                        continue
                    out.append(desc)
    return out


def _mols_to_desciptors(descriptors: List[Descriptor], rdkit_mols: List[rdchem.Mol]) -> np.ndarray:
    """Wraps the mordred descriptor calculator.

    Args:
        descriptors (Mordred descriptor instances): Descriptors to calculate.
        rdkit_mols (list[rdkit mols]): List of RDKit molecules.

    Returns:
        np.array: Calculated descriptors.
    """
    start = perf_counter()
    mordred_calc = Calculator(descriptors)
    logger.info("Calculating descriptors")
    mordred_descs = np.array(list(mordred_calc.map(rdkit_mols, nproc=psutil.cpu_count(logical=True), quiet=False)))
    logger.info(f"Descriptor calculation complete, elapsed time: {str(datetime.timedelta(seconds=perf_counter() - start))}")
    return mordred_descs


def get_descriptors(cache_filepath: Union[str, bool], descriptors: List[str], rdkit_mols: List[rdchem.Mol]) -> pd.DataFrame:
    """Calculates requested descriptors for the given molecules, optionally writing to a cache file.

    Args:
        cache_filepath (str | bool): Filepath for cache, False to not cache.
        descriptors (list[str]): Names of mordred-community descriptors to calculate.
        rdkit_mols (list[rdchem.Mol]): Molecules.

    Returns:
        pd.DataFrame: Calculated descriptors.
    """
    d2c = _descriptor_names_to_mordred_class(descriptors)
    out = pd.DataFrame(data=_mols_to_desciptors(d2c, rdkit_mols), columns=descriptors)
    out = out.apply(pd.to_numeric, errors="coerce")
    if cache_filepath:
        out.to_csv(cache_filepath)
        logger.info(f"Cached descriptors to {cache_filepath}")
    return out
