"""
quantumscents.py

Usage:
python quantumscents.py

Description:
This file shows how to use the internal fastprop classes and functions
to train your own fastprop model with a custom set of descriptors
and data splitting function.

It is commented throughout to explain further!
"""

import os
import pickle as pkl

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from fastprop.data import (
    clean_dataset,
    fastpropDataLoader,
    fastpropDataset,
    standard_scale,
)
from fastprop.descriptors import _descriptor_names_to_mordred_class, get_descriptors
from fastprop.io import load_saved_descriptors
from fastprop.model import fastprop, train_and_test

# the QuantumScents Zenodo includes a number of utilities for loading the dataset
# two of which are imported here.
from dataset.loader import CID_TO_SMILES, QS_DATA  # isort:skip

# this file is a list of list of list - the outer list is len=number of folds, then the sublist is len=3
# i.e. train/val/test and then then innermost lists contain the indexes for train/val/test
DATASPLIT = os.path.join(os.path.dirname(__file__), "models", "dataset-partitioning", "szymanski-splits", "szy-splits-3-folds.pkl")

# for comparison, can either generate 2d or 3d descriptors
DESCRIPTOR_SET = "2d"  # "3d"

# to save time between runs, we save the calculated descriptors
CACHED_DESCRIPTORS = "cached_" + DESCRIPTOR_SET + "_quantumscents_descriptors.csv"


RANDOM_SEED = 60221023

descriptors = None
if os.path.exists(CACHED_DESCRIPTORS):
    descriptors = pd.DataFrame(load_saved_descriptors(CACHED_DESCRIPTORS))
else:
    # QuantumScents provides a ground-state structure for each molecule in the dataset,
    # present in the Zenodo as an xyz file. We can calculate descriptors for these
    # structures using RDKit and mordred (both part of fastprop)
    #
    # Adapted from the QuantumScents model demo code:
    # https://github.com/JacksonBurns/chemprop/blob/c056dc7b5119bf0dbc365e6d40d9e5609d3b473f/chemprop/rdkit.py#L36
    def make_mol_from_xyz(filepath: str):
        raw_mol = Chem.rdmolfiles.MolFromXYZFile(filepath)
        mol = Chem.Mol(raw_mol)
        try:
            rdDetermineBonds.DetermineBonds(mol, charge=0)
        except ValueError as ve:
            print(f"Error inferring bonds in 3D for file {xyz_fpath}, original error: " + str(ve))
            print("Falling back to SMILES connectivity only.")
            mol = False
        return mol

    # load all of the xyz files/smiles as rdkit molecules
    all_mols = []
    for cid in QS_DATA.index:
        # 3d-based descriptors
        if DESCRIPTOR_SET == "3d":
            xyz_fpath = os.path.join("dataset", "xyzfiles", "mol" + cid + ".xyz")
            mol = make_mol_from_xyz(xyz_fpath)
            if mol:
                all_mols.append(mol)
                continue
        # 2d-based - use as a backup for 3d
        all_mols.append(Chem.MolFromSmiles(CID_TO_SMILES[cid]))

    # calculate the descriptors for each
    descriptors = get_descriptors(
        CACHED_DESCRIPTORS,
        _descriptor_names_to_mordred_class(include_3d=DESCRIPTOR_SET == "3d"),
        all_mols,
    )

    descriptors = pd.DataFrame(descriptors)
    descriptors.to_csv(CACHED_DESCRIPTORS)


target_names = QS_DATA.columns
targets = QS_DATA.to_numpy()
# clean the dataset, just in case
targets, rdkit_mols, smiles = clean_dataset(targets, np.array([CID_TO_SMILES[i] for i in QS_DATA.index]))

targets = torch.tensor(QS_DATA.to_numpy(), dtype=torch.float32)
descriptors_og = torch.tensor(descriptors.to_numpy(), dtype=torch.float32)

# iterate through the three folds
all_test_results, all_validation_results = [], []
for fold_number in range(3):
    with open(DATASPLIT, "rb") as file:
        train_indexes, val_indexes, test_indexes = pkl.load(file)[fold_number]
    descriptors = descriptors_og.detach().clone()
    descriptors[train_indexes], feature_means, feature_vars = standard_scale(descriptors[train_indexes])
    descriptors[val_indexes] = standard_scale(descriptors[val_indexes], feature_means, feature_vars)
    descriptors[test_indexes] = standard_scale(descriptors[test_indexes], feature_means, feature_vars)

    # initialize dataloaders and model, then train
    train_dataloader = fastpropDataLoader(fastpropDataset(descriptors[train_indexes], targets[train_indexes]), shuffle=True)
    val_dataloader = fastpropDataLoader(fastpropDataset(descriptors[val_indexes], targets[val_indexes]))
    test_dataloader = fastpropDataLoader(fastpropDataset(descriptors[test_indexes], targets[test_indexes]))
    model = fastprop(
        input_size=descriptors.shape[1],
        hidden_size=3000,
        readout_size=len(target_names),
        num_tasks=len(target_names),
        learning_rate=0.0001,
        fnn_layers=3,
        problem_type="multilabel",
        feature_means=feature_means,
        feature_vars=feature_vars,
    )
    test_results, validation_results = train_and_test("output_" + DESCRIPTOR_SET, model, train_dataloader, val_dataloader, test_dataloader)
    all_test_results.append(test_results[0])
    all_validation_results.append(validation_results[0])

validation_results_df = pd.DataFrame.from_records(all_validation_results)
print(f"Displaying validation results:\n{validation_results_df.describe().transpose().to_string()}")
test_results_df = pd.DataFrame.from_records(all_test_results)
print(f"Displaying testing results:\n {test_results_df.describe().transpose().to_string()}")
