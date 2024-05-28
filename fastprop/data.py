from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
from astartes import train_val_test_split
from astartes.molecules import train_val_test_split_molecules
from rdkit import Chem
from torch.utils.data import DataLoader as TorchDataloader
from torch.utils.data import Dataset as TorchDataset

from fastprop.defaults import init_logger

logger = init_logger(__name__)


def split(
    smiles: List[str],
    random_seed: int = 42,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    sampler: Literal["random", "scaffold"] = "random",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a dataset into training, validation, and testing subsets and return the indices for each.

    Args:
        smiles (list[str]): SMILES strings to split.
        random_seed (int, optional): Seed for splitting. Defaults to 42.
        train_size (float, optional): Fraction of data for training. Defaults to 0.8.
        val_size (float, optional): Fraction of data for validation. Defaults to 0.1.
        test_size (float, optional): Fraction of data for testing. Defaults to 0.1.
        sampler (Literal["random", "scaffold"], optional): Type of sampler from astartes. Defaults to "random".

    Raises:
        TypeError: Unsupported sampler requested

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Indices for training, validation, and testing.
    """
    split_kwargs = dict(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        sampler=sampler,
        random_state=random_seed,
        return_indices=True,
    )
    if sampler == "random":
        (
            *_,
            train_idxs,
            val_idxs,
            test_idxs,
        ) = train_val_test_split(np.arange(len(smiles)), **split_kwargs)
    elif sampler == "scaffold":
        (
            *_,
            train_idxs,
            val_idxs,
            test_idxs,
        ) = train_val_test_split_molecules(smiles, **split_kwargs)
    else:
        raise TypeError(f"Unknown sampler {sampler=}.")
    return train_idxs, val_idxs, test_idxs


def standard_scale(data: torch.Tensor, means: Optional[torch.Tensor] = None, variances: Optional[torch.Tensor] = None):
    """Applies standard scaling, i.e. Z-Score normalization.

    Args:
        data (torch.Tensor): Data to be scaled.
        means (torch.Tensor, optional): Precomputed means. Defaults to None and are calculated from data.
        variances (torch.Tensor, optional): Precomputed means. Defaults to None and are calculated from data.

    Returns:
        torch.Tensor: scaled data and its means and variances if they were not provided in call.
    """
    return_stats = False
    if means is None:
        return_stats = True
        means = data.nanmean(dim=0)
    # treat missing features' means as zero
    torch.nan_to_num_(means)

    # set missing rows with their column's average value
    data = data.where(~data.isnan(), means)
    if variances is None:
        variances = data.var(dim=0)

    # center around zero and apply unit variance
    data = (data - means) / variances.sqrt()

    # the above will result in features with no variance going to nan,
    # -inf, or inf (if they were zero or slightly below/above it) so
    # we reset them to zero
    torch.nan_to_num_(data, posinf=0.0, neginf=0.0)
    if return_stats:
        return data, means, variances
    else:
        return data


def inverse_standard_scale(data: torch.Tensor, means: torch.Tensor, variances: torch.Tensor) -> torch.Tensor:
    """Undo standard scaling.

    Args:
        data (torch.Tensor): Input data.
        means (torch.Tensor): Precomputed means which were used for scaling.
        variances (torch.Tensor): Precomputed variances which were used for scaling.

    Returns:
        torch.Tensor: Unscaled data.
    """
    return data * variances.sqrt() + means


def clean_dataset(targets: np.ndarray, smiles: np.ndarray):
    """Removes targets with missing values and SMILES which cannot be converted to molecules.

    Args:
        targets (np.ndarray): Targets corresponding to mols.
        smiles (np.ndarray): SMILES corresponding to the targets.

    Returns:
        tuple[np.ndarray, np.ndarray]: Valid targets and RDKit Molecules.
    """
    rdkit_mols = np.array(list(Chem.MolFromSmiles(i) for i in smiles))
    starting_length = len(rdkit_mols)

    # remove dataset entries where the molecule could not be built
    error_mols_idxs = np.where(rdkit_mols == None)[0]  # noqa: E711
    for idx in error_mols_idxs:
        logger.warn(
            f"Unable to create RDKit molecule from SMILES string {smiles[idx]} (index {idx})."
            " Both the molecule and the target will be removed from the data."
        )
    if len(error_mols_idxs) > 0:
        # specify axis=0 to avoid changing dimensions (flattening)
        logger.warn(f"Removed {len(error_mols_idxs)} entries from the dataset ({100*len(error_mols_idxs)/starting_length:.2f}% of the data).")
        targets = np.delete(targets, error_mols_idxs, axis=0)
        rdkit_mols = np.delete(rdkit_mols, error_mols_idxs, axis=0)
        smiles = np.delete(smiles, error_mols_idxs)

    # also remove dataset entries where the target is missing
    # TODO: weight masking instead of removal?
    error_target_idxs = [idx for idx, arr in enumerate(targets) if np.any(np.isnan(arr))]
    for idx in error_target_idxs:
        logger.warn(
            f"Missing target value (target={targets[idx]}) for SMILES {smiles[idx]}. Both the molecule and the target will be removed from the data."
        )
    if len(error_target_idxs) > 0:
        # specify axis=0 to avoid changing dimensions (flattening)
        logger.warn(f"Removed {len(error_target_idxs)} entries from the dataset ({100*len(error_target_idxs)/starting_length:.2f}% of the data).")
        targets = np.delete(targets, error_target_idxs, axis=0)
        rdkit_mols = np.delete(rdkit_mols, error_target_idxs, axis=0)
        smiles = np.delete(smiles, error_target_idxs)
    return targets, rdkit_mols, smiles


# wrap the basic pytorch Dataset and Dataloader to set some arguments in a convenient way


class fastpropDataset(TorchDataset):
    def __init__(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
    ):
        self.features = features
        self.length = len(targets)
        self.targets = targets

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


class fastpropDataLoader(TorchDataloader):
    def __init__(
        self,
        dataset: fastpropDataset,
        batch_size: int = 128,
        num_workers: int = 1,
        persistent_workers: bool = True,
        shuffle=False,
        **torch_kwargs,
    ):
        super().__init__(
            dataset,
            batch_size,
            shuffle=shuffle,
            persistent_workers=persistent_workers,
            num_workers=num_workers,
            **torch_kwargs,
        )
