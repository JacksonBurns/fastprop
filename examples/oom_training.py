"""
oom_training.py

This script demonstrates how to run fastprop on a dataset which is too large to fit in memory
(GPU or main memory) and instead must be dynamically loaded during training, i.e. "Out-of-memory"
(OOM) training.

The dataset contains the free energy of solvation for all combinations of ~340 solvents and
~340k solutes for a total of 101MM samples. Both the solute and the solvent have 1600 input features,
so the total size of this dataset is far too large to fit in memory.

At a high level, we do the following:
 1. Calculate the features for all of the unique molecules in this dataset and save them to a file
    which can be passed around via the disk to the other training processes.
 2. Splits the data according to some chosen approach and rescales the input features.
 3. Initialize a group of processes, each of which has access to all the data and loads only a part of
    it in small batches at a time, and then runs the training.

Additional notes:
 - This training was run on a local machine with 8 GPUs to speed up training. The conda environment
   was set up with:
    - pip install -e ../ torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
      in an environment with Python 3.10
   and then tun under GNU screen with: screen -L -Logfile tranining.log -S train

"""

import csv
import os
import pickle as pkl
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from astartes import train_test_split, train_val_test_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from rdkit import Chem
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.dataloader import DataLoader as TorchDataloader

from fastprop.fastprop_core import fastprop
from fastprop.utils import (
    ALL_2D,
    calculate_mordred_desciptors,
    mordred_descriptors_from_strings,
)

# global variables:
# - pickled dataframe containing the smiles of the solute, solvent, and their corresponding Gsolv
PAIR_DATA_FILE = "hwhp_gsolv.pkl"
# - file to write the molecular features to
PROPERTY_LOOKUP_FILE = "hwhp_property_lookup.csv"
# - random seed for splitting the dataset
RANDOM_SEED = 42
# - split type for performing training
SPLIT = "all_extrapolation"  # "random_interpolation", "solvent_extrapolation"

# re-calculate the features if they have not been calculated before
if not os.path.exists(PROPERTY_LOOKUP_FILE):
    with open("hwhp_gsolv.pkl", "rb") as file:
        gsolv_df = pkl.load(file)
    unique_smiles = np.hstack((pd.unique(gsolv_df["solvent_smiles"]), pd.unique(gsolv_df["solute_smiles"])))
    descs = calculate_mordred_desciptors(
        mordred_descriptors_from_strings(ALL_2D),
        list(Chem.MolFromSmiles(i) for i in unique_smiles),
        n_procs=-1,
        strategy="low-memory",
    )

    # used to convert the descriptor to a plaintext format for writing to a file
    def safe_cast(i):
        try:
            return format(i, ".6e")
        except:
            return ""

    with open(PROPERTY_LOOKUP_FILE, "w", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(["smiles"] + list(ALL_2D))
        for smiles, values in zip(unique_smiles, descs):
            writer.writerow([smiles] + [safe_cast(i) for i in values])


# define a class which gradually loads the input features as needed
class OOMDataset(TorchDataset):
    def __init__(self, idxs: list[int], transformed_features: pd.DataFrame):
        # read all the solute/solvent combinations
        with open(PAIR_DATA_FILE, "rb") as file:
            self.all_pairs: pd.DataFrame = pkl.load(file)
        # keep only those that are part of this dataset (train, val, or test)
        self.all_pairs = self.all_pairs.iloc[idxs]
        # load the features for all of the molecules
        self.descriptor_lookup_df: pd.DataFrame = transformed_features
        # save the length for later
        self.len: int = len(self.all_pairs)

    def __getitem__(self, index):
        # pull out the solvent and solute at the given index
        solute, solvent = self.all_pairs.iloc[index][["solute_smiles", "solvent_smiles"]]
        solute_features = torch.tensor(self.descriptor_lookup_df.loc[solute].to_numpy(), dtype=torch.float32)
        solvent_features = torch.tensor(self.descriptor_lookup_df.loc[solvent].to_numpy(), dtype=torch.float32)
        gsolv = torch.tensor(self.all_pairs.iloc[index]["Gsolv (kcal/mol)"], dtype=torch.float32).unsqueeze(dim=0)
        # concatenate their representations, look up the target value as well
        return torch.cat((solute_features, solvent_features), dim=0), gsolv

    def __len__(self):
        return self.len


# split the data according to the requested approach
with open(PAIR_DATA_FILE, "rb") as file:
    all_pairs: pd.DataFrame = pkl.load(file)
    DATASET_SIZE = len(all_pairs)
if SPLIT == "solvent_extrapolation":  # model will see all solutes and only some solvents during training
    solvents_train, solvents_val, solvents_test = train_val_test_split(pd.unique(all_pairs["solvent_smiles"]), random_state=RANDOM_SEED)
    INDEXES_TRAIN = all_pairs.index[all_pairs["solvent_smiles"].isin(solvents_train)].tolist()
    INDEXES_VAL = all_pairs.index[all_pairs["solvent_smiles"].isin(solvents_val)].tolist()
    INDEXES_TEST = all_pairs.index[all_pairs["solvent_smiles"].isin(solvents_test)].tolist()
elif SPLIT == "random_interpolation":  # model will see all solvents and solutes during training
    INDEXES_TRAIN, INDEXES_VAL, INDEXES_TEST = train_val_test_split(np.array(list(range(DATASET_SIZE))), random_state=RANDOM_SEED)
elif SPLIT == "all_extrapolation":  # model will see only some solutes and some solvents during training
    solvents_train, _ = train_test_split(
        pd.unique(all_pairs["solvent_smiles"]),
        train_size=0.90,
        test_size=0.1,
    )
    solutes_train, _ = train_test_split(
        pd.unique(all_pairs["solute_smiles"]),
        train_size=0.90,
        test_size=0.1,
    )
    INDEXES_TRAIN = all_pairs.index[all_pairs["solvent_smiles"].isin(solvents_train) & all_pairs["solute_smiles"].isin(solutes_train)].tolist()
    unclaimed_indexes = np.array(list(set(all_pairs.index.tolist()).difference(INDEXES_TRAIN)))
    _indexes_val, _indexes_test = train_test_split(unclaimed_indexes, train_size=0.5, test_size=0.5)
    INDEXES_VAL, INDEXES_TEST = _indexes_val.tolist(), _indexes_test.tolist()
    del solutes_train, solvents_train, unclaimed_indexes, _indexes_test, _indexes_val
else:
    print(f"what is {SPLIT=}?")
    exit(1)

print("Imputing and Rescaling input features...")
_descriptor_lookup_df = pd.read_csv(PROPERTY_LOOKUP_FILE, index_col="smiles")


def _impute_and_scale(descriptor_lookup_df, s):
    # s is either solvent or solute column name in the pairs file
    mean_imputer = SimpleImputer(missing_values=np.nan, strategy="mean", keep_empty_features=True).set_output(transform="pandas")
    # subset of s smiles strings for fitting the rescalers, i.e. only the solutes
    # or solvents that the model will see during training
    train_smiles = pd.unique(all_pairs.iloc[INDEXES_TRAIN][s])
    # all the smiles strings in s (for indexing the overall dataframe of descriptors)
    # i.e. all solutes or all solvents
    all_smiles = pd.unique(all_pairs[s])
    # fit the mean imputer based on the entire dataset
    mean_imputer.fit(descriptor_lookup_df.loc[train_smiles])
    descs = mean_imputer.transform(descriptor_lookup_df.loc[all_smiles])
    # fit the feature scaler on the data which was not imputed
    feature_scaler = StandardScaler().set_output(transform="pandas")
    feature_scaler.fit(descriptor_lookup_df.loc[train_smiles])
    descs = feature_scaler.transform(descs)
    # save the trained scalers for later inference
    if not os.path.exists(f"fixedinputs_oom_{s}_scalers.yml"):
        with open(f"fixedinputs_oom_{s}_scalers.yml", "w") as file:
            file.write(
                yaml.dump(
                    dict(
                        mean_imputer=pkl.dumps(mean_imputer),
                        feature_scaler=pkl.dumps(feature_scaler),
                    ),
                    sort_keys=False,
                )
            )
    return descs.fillna(0)  # nan (missing features for whole training dataset) -> zero


_solvent_descs = _impute_and_scale(_descriptor_lookup_df, "solvent_smiles")
_solute_descs = _impute_and_scale(_descriptor_lookup_df, "solute_smiles")
INPUT_FEATURES = pd.concat((_solvent_descs, _solute_descs))
del _descriptor_lookup_df, _solute_descs, _solvent_descs


# re-define the fastprop class to use our new dataloader and pre-calculated splits
class OOMfastprop(fastprop):
    def __init__(
        self,
        num_epochs,
        input_size,
        hidden_size,
        readout_size,
        learning_rate,
        fnn_layers,
        problem_type,
        target_names,
        batch_size,
        random_seed,
    ):
        super().__init__(
            num_epochs,
            input_size,
            hidden_size,
            readout_size,
            learning_rate,
            fnn_layers,
            problem_type,
            cleaned_data=None,
            targets=None,
            target_names=target_names,
            batch_size=batch_size,
            random_seed=random_seed,
            train_size=None,
            val_size=None,
            test_size=None,
            sampler=None,
            smiles=None,
            verbose=True,
        )
        # mock the target scaler used for reporting some human-readable metrics
        self.target_scaler = SimpleNamespace(n_features_in_=1, inverse_transform=lambda i: np.array(i))

    def setup(self, stage=None): ...  # skip feature scaling and dataset splitting

    def _init_dataloader(self, shuffle, idxs):
        return TorchDataloader(
            OOMDataset(idxs, INPUT_FEATURES),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=1,
            persistent_workers=True,
        )

    def train_dataloader(self):
        return self._init_dataloader(True, INDEXES_TRAIN)

    def val_dataloader(self):
        return self._init_dataloader(False, INDEXES_VAL)

    def test_dataloader(self):
        return self._init_dataloader(False, INDEXES_TEST)

    def on_train_epoch_end(self):
        print(f"Epoch [{self.trainer.current_epoch + 1}/{self.num_epochs}]")

    # override the basic optimizer to add regularization to help with generalization to new molecules
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.001)
        return {"optimizer": optimizer}


def train():
    EPOCHS = 50
    lightning_module = OOMfastprop(
        num_epochs=EPOCHS,
        input_size=1613 * 2,
        hidden_size=3200,
        readout_size=1,
        learning_rate=0.0001,
        fnn_layers=3,
        problem_type="regression",
        target_names=["gsolv"],
        batch_size=1024,
        random_seed=RANDOM_SEED,
    )
    tensorboard_logger = TensorBoardLogger(os.getcwd(), name="l2_prod_" + SPLIT, default_hp_metric=False)
    callbacks = [
        EarlyStopping(
            monitor=f"validation_{lightning_module.training_metric}_loss",
            mode="min",
            verbose=False,
            patience=EPOCHS // 5,
        ),
        ModelCheckpoint(
            monitor=f"validation_{lightning_module.training_metric}_loss",
            dirpath=os.path.join(os.getcwd(), "l2_prod_checkpoints"),
            filename=SPLIT + "-{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
            mode="min",
        ),
    ]
    warnings.filterwarnings(action="ignore", message=".*late the root mean squared error.*")
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="cuda",
        devices="auto",
        enable_progress_bar=False,
        enable_model_summary=True,
        logger=tensorboard_logger,
        log_every_n_steps=1_000,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        deterministic=True,
    )
    trainer.fit(lightning_module)
    validation_results = trainer.validate(lightning_module, verbose=False)
    test_results = trainer.test(lightning_module, verbose=False)
    validation_results_df = pd.DataFrame.from_records(validation_results, index=("value",))
    print("Displaying validation results:\n{:s}".format(validation_results_df.transpose().to_string()))
    test_results_df = pd.DataFrame.from_records(test_results, index=("value",))
    print("Displaying testing results:\n{:s}".format(test_results_df.transpose().to_string()))


if __name__ == "__main__":
    train()
