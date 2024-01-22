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

import psutil
from fastprop.utils import mordred_descriptors_from_strings, calculate_mordred_desciptors, load_saved_desc
from fastprop.fastprop_core import _training_loop, ArbitraryDataModule
from fastprop.hopt import _hopt_loop
from fastprop.preprocessing import preprocess
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
import pandas as pd

# the QuantumScents Zenodo includes a number of utilities for loading the dataset
# two of which are imported here.
from dataset.loader import QS_DATA, CID_TO_SMILES

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
    descs = load_saved_desc(CACHED_DESCRIPTORS)
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

    d = pd.DataFrame(descs)
    d.to_csv(CACHED_DESCRIPTORS)

# the fastprop preprocess function applies some basic rescaling and inference
X, y, target_scaler = preprocess(
    descs,
    QS_DATA.to_numpy(),
    zero_variance_drop=False,
    problem_type="multilabel",
)
# count how many features remain (some had no values for the entire dataset)
number_features = X.shape[1]
# make sure to set the names of the input features!
target_scaler.feature_names_in_ = QS_DATA.columns


# because QuantumScents includes a file with the desired splits for the data, we will override
# the fastprop default DataModule class
class QuantumScentsDataModule(ArbitraryDataModule):
    # the setup method is called to split the data, so all we need to do is assign the indexes
    # to do so, we open the file which is described at the top of this file
    def setup(self, stage=None):
        with open(DATASPLIT, "rb") as file:
            self.train_idxs, self.val_idxs, self.test_idxs = pkl.load(file)[self.fold_number]


# initialize the module
qs_datamodule = QuantumScentsDataModule(X, y, 4096, RANDOM_SEED, None, None, None, None, None)

# call the _hopt_loop and _training_loop function, which will automatically increment fold_number
# for each of number_repeats - in this case we have 3 folds from QuantumScents, so
# we set number_repeats to 3
general_args = dict(
    number_repeats=3,
    number_features=number_features,
    target_scaler=target_scaler,
    number_epochs=300,
    learning_rate=0.0001,
    output_directory="quantumscents_" + DESCRIPTOR_SET,
    datamodule=qs_datamodule,
    patience=10,
    problem_type="multilabel",
    num_classes=113,
)

# best_result = _hopt_loop(
#     **general_args,
#     random_seed=RANDOM_SEED,
#     n_parallel=4,
#     n_trials=128,
# )

# previous 2d run
# best_result = {"hidden_size": 3000, "fnn_layers": 3}
# previous 3d run
best_result = {"hidden_size": 2600, "fnn_layers": 3}

_training_loop(
    # use the results from hyperparameter optimization
    fnn_layers=best_result["fnn_layers"],
    hidden_size=best_result["hidden_size"],
    # same as above
    **general_args,
)
