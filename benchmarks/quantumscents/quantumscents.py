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

import pandas as pd
import psutil

# the QuantumScents Zenodo includes a number of utilities for loading the dataset
# two of which are imported here.
from dataset.loader import CID_TO_SMILES, QS_DATA
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from fastprop.fastprop_core import fastprop, train_and_test
from fastprop.preprocessing import preprocess
from fastprop.utils import (
    calculate_mordred_desciptors,
    load_saved_desc,
    mordred_descriptors_from_strings,
)

# this file is a list of list of list - the outer list is len=number of folds, then the sublist is len=3
# i.e. train/val/test and then then innermost lists contain the indexes for train/val/test
DATASPLIT = os.path.join(os.path.dirname(__file__), "models", "dataset-partitioning", "szymanski-splits", "szy-splits-3-folds.pkl")

# for comparison, can either generate 2d or 3d descriptors
DESCRIPTOR_SET = "3d"  # "2d"

# to save time between runs, we save the calculated descriptors
CACHED_DESCRIPTORS = "cached_" + DESCRIPTOR_SET + "_quantumscents_descriptors.csv"


RANDOM_SEED = 60221023

descs = None
if os.path.exists(CACHED_DESCRIPTORS):
    descs = pd.DataFrame(load_saved_desc(CACHED_DESCRIPTORS))
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
    descs = calculate_mordred_desciptors(
        mordred_descriptors_from_strings([], include_3d=DESCRIPTOR_SET == "3d"),
        all_mols,
        psutil.cpu_count(logical=True),
        ignore_3d=DESCRIPTOR_SET == "2d",
    )

    descs = pd.DataFrame(descs)
    descs.to_csv(CACHED_DESCRIPTORS)

# the fastprop preprocess function applies some basic rescaling and inference
X = preprocess(descs).to_numpy()
# count how many features remain
number_features = X.shape[1]
# make sure to set the names of the input features!
target_names = QS_DATA.columns
targets = QS_DATA.to_numpy()


# we override the _split method in the fastprop class and replace it with one
# which loads our custom splits
class quantumscents_fastprop(fastprop):
    def __init__(self, fold_number, **kwargs):
        self.fold_number = fold_number
        super().__init__(**kwargs)

    def _split(self):
        with open(DATASPLIT, "rb") as file:
            self.train_idxs, self.val_idxs, self.test_idxs = pkl.load(file)[self.fold_number]


all_valid_results = []
all_test_results = []
for fold_number in range(3):
    lightning_module = quantumscents_fastprop(
        fold_number=fold_number,
        num_epochs=300,
        input_size=number_features,
        hidden_size=3000,
        readout_size=113,
        learning_rate=0.0001,
        fnn_layers=3,
        problem_type="multilabel",
        cleaned_data=X,
        targets=targets,
        target_names=target_names,
        batch_size=4096,
        random_seed=RANDOM_SEED,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        sampler="",
        smiles=None,
        verbose=True,
    )
    test_results, valid_results = train_and_test("quantumscents_" + DESCRIPTOR_SET, lightning_module, patience=30)
    all_valid_results.append(valid_results[0])
    all_test_results.append(test_results[0])
validation_results_df = pd.DataFrame.from_records(all_valid_results)
print("Displaying validation results:\n", validation_results_df.describe().transpose().to_string())
test_results_df = pd.DataFrame.from_records(all_test_results)
print("Displaying testing results:\n", test_results_df.describe().transpose().to_string())
