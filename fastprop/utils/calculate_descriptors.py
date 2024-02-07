import os
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import psutil
from mordred import Calculator

from fastprop.defaults import init_logger

from .descriptor_lists import descriptors_lookup
from .load_data import load_saved_desc
from .select_descriptors import mordred_descriptors_from_strings

logger = init_logger(__name__)


# mordred tried to avoid instantiating multiple Calculator classes, which makes
# the parallelism slower but more memory efficient. We do this manually:
def _f(in_tuple):
    mordred_calc = Calculator(in_tuple[2], ignore_3D=in_tuple[3])
    mordred_descs = np.array(list(mordred_calc.map(in_tuple[0], nproc=1, quiet=in_tuple[1])))
    return mordred_descs


def calculate_mordred_desciptors(descriptors, rdkit_mols, n_procs, strategy: Literal["fast", "low-memory"] = "fast", ignore_3d=True):
    """Wraps the mordred descriptor calculator.

    Args:
        descriptors (Mordred descriptor instances): Descriptors to calculate
        rdkit_mols (list[rdkit mols]): List of RDKit molecules.
        n_procs (int): Number of parallel processes.
        strategy (Literal["fast", "low-memory", optional): Parallelization strategy. Defaults to "fast".
        ignore_3d (bool, optional): Include 3D descriptors, if in given list. Defaults to True.

    Raises:
        RuntimeError: Invalid choice of parallel strategy.

    Returns:
        np.array: Calculated descriptors.
    """
    # descriptors should be a list of mordred descriptors classes
    if strategy not in {"fast", "low-memory"}:
        raise RuntimeError(f"Strategy {strategy} not supported, only 'fast' and 'low-memory'.")

    mordred_descs = None
    logger.info(f"Calculating descriptors using {strategy=}")
    if strategy == "fast":
        # higher level parallelism - uses more memory
        # TODO: subdivide batches further to avoid large communication bottleneck after all descriptors are calculated
        batches = np.array_split(rdkit_mols, n_procs)
        # let the root process show a progress bar, since array split will make
        # that one the largest
        # convert to starmap to avoid having to duplicate the list of descriptor classes - too much communication (?)
        to_procs = [(batch, bool(i), descriptors, ignore_3d) for i, batch in enumerate(batches)]
        with Pool(n_procs) as p:
            mordred_descs = np.vstack(p.map(_f, to_procs, 1))
    else:
        # mordred parallelism
        mordred_calc = Calculator(descriptors, ignore_3D=ignore_3d)
        mordred_descs = np.array(list(mordred_calc.map(rdkit_mols, nproc=psutil.cpu_count(logical=True), quiet=False)))
    return mordred_descs


def _get_descs(precomputed, input_file, output_directory, descriptors, enable_cache, mols, as_df=False):
    """Loads descriptors according to the user-specified configuration.

    This is a 'hidden' function since the caching logic is specific to fastprop.

    Args:
        precomputed (str): Use precomputed descriptors if str is.
        input_file (str): Filepath of input data.
        output_directory (str): Destination directory for caching.
        descriptors (list): fastprop set of descriptors to calculate.
        enable_cache (bool): Allow/disallow caching mechanism.
        mols (list): RDKit molecules.
    """
    descs = None
    if precomputed:
        del mols
        logger.info(f"Loading precomputed descriptors from {precomputed}.")
        descs = load_saved_desc(precomputed)
    else:
        in_name = Path(input_file).stem
        # cached descriptors, which contains (1) cached (2) source filename (3) types of descriptors (4) timestamp when file was last touched
        cache_file = os.path.join(output_directory, "cached_" + in_name + "_" + descriptors + "_" + str(int(os.stat(input_file).st_ctime)) + ".csv")

        if os.path.exists(cache_file) and enable_cache:
            logger.info(f"Found cached descriptor data at {cache_file}, loading instead of recalculating.")
            descs = load_saved_desc(cache_file)
        else:
            d2c = mordred_descriptors_from_strings(descriptors_lookup[descriptors])
            # use all the cpus available
            descs = calculate_mordred_desciptors(d2c, mols, psutil.cpu_count(logical=False), "fast")
            # cache these
            if enable_cache:
                d = pd.DataFrame(descs)
                d.to_csv(cache_file)
                logger.info(f"Cached descriptors to {cache_file}.")
    if as_df:
        return pd.DataFrame(data=descs, columns=descriptors_lookup[descriptors])
    return descs
