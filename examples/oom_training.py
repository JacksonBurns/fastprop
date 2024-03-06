# to work on mithrim, had to install fastprop with
# pip install -e ../ torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# in an environment with Python 3.10
import pickle as pkl
import pandas as pd
from types import SimpleNamespace
import os
import warnings

from astartes import train_val_test_split
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.dataloader import DataLoader as TorchDataloader
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl

from fastprop.fastprop_core import fastprop


PROPERTY_LOOKUP_FILE = "hwhp_property_lookup.csv"
PAIR_DATA_FILE = "hwhp_gsolv.pkl"

# train a random set of the solvents
with open(PAIR_DATA_FILE, "rb") as file:
    all_pairs: pd.DataFrame = pkl.load(file)
    solvents_train, solvents_val, solvents_test = train_val_test_split(pd.unique(all_pairs["solvent_smiles"]))

INDEXES_TRAIN = all_pairs.index[all_pairs["solvent_smiles"].isin(solvents_train)].tolist()
INDEXES_VAL = all_pairs.index[all_pairs["solvent_smiles"].isin(solvents_val)].tolist()
INDEXES_TEST = all_pairs.index[all_pairs["solvent_smiles"].isin(solvents_test)].tolist()


class OOMDataset(TorchDataset):
    def __init__(self, idxs: list[int]):
        # read all the solute/solvent combinations
        with open(PAIR_DATA_FILE, "rb") as file:
            self.all_pairs: pd.DataFrame = pkl.load(file)
        # keep only those that are part of this dataset (train, val, or test)
        self.all_pairs = self.all_pairs.iloc[idxs]
        # doing the type casts here uses too much memory!
        # convert the target into the datatype for torch
        # print("Converting target values to torch...")
        # self.all_pairs["Gsolv (kcal/mol)"] = self.all_pairs["Gsolv (kcal/mol)"].apply(
        #     lambda i: torch.as_tensor(i, dtype=torch.float32).unsqueeze(dim=0)
        # )
        # load the features for all of the molecules
        self.descriptor_lookup_df: pd.DataFrame = pd.read_csv(PROPERTY_LOOKUP_FILE, index_col="smiles")
        # find which molecules are actually in this split (not all that are in the property lookup file)
        # include_smiles = set(np.hstack((pd.unique(self.all_pairs["solvent_smiles"]), pd.unique(self.all_pairs["solute_smiles"]))))
        # map the molecules included in this dataset to their properties, in the appropriate data type
        # print("Converting features to torch...")
        # THIS EATS MEMORY - need to delete the dataframe as you go or find a way to do this that frees the
        # previous memory at the same time as this is allocated
        # self.smiles_to_features = {
        #     smiles: torch.tensor(row.to_numpy(), dtype=torch.float32) for smiles, row in descriptor_lookup_df.iterrows() # if smiles in include_smiles
        # }
        # # explicitly remove some unwanted objects
        # del descriptor_lookup_df  #, include_smiles
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
            OOMDataset(idxs),
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

    def validation_step(self, batch, batch_idx):
        # skip human loss for this huge, slow dataset
        loss = self._machine_loss(batch, return_all=False)
        self.log(f"validation_{self.training_metric}_loss", loss, sync_dist=True)
        return loss


EPOCHS = 50
lightning_module = OOMfastprop(
    num_epochs=EPOCHS,
    input_size=63 * 2,
    hidden_size=200,
    readout_size=1,
    learning_rate=0.0001,
    fnn_layers=2,
    problem_type="regression",
    target_names=["gsolv"],
    batch_size=1024,
    random_seed=42,
)
try:
    repetition_number = len(os.listdir(os.path.join(os.getcwd(), "tensorboard_logs"))) + 1
except FileNotFoundError:
    repetition_number = 1
tensorboard_logger = TensorBoardLogger(os.getcwd(), name="tensorboard_logs", version=f"repetition_{repetition_number}", default_hp_metric=False)

callbacks = [
    EarlyStopping(
        monitor=f"validation_{lightning_module.training_metric}_loss",
        mode="min",
        verbose=False,
        patience=EPOCHS // 5,
    )
]
callbacks.append(
    ModelCheckpoint(
        monitor=f"validation_{lightning_module.training_metric}_loss",
        dirpath=os.path.join(os.getcwd(), "checkpoints"),
        filename=f"repetition-{repetition_number}" + "-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
)

warnings.filterwarnings(action="ignore", message=".*late the root mean squared error.*")
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    accelerator="cuda",
    devices="auto",
    enable_progress_bar=True,
    enable_model_summary=True,
    logger=tensorboard_logger,
    log_every_n_steps=1,
    enable_checkpointing=True,
    check_val_every_n_epoch=1,
    callbacks=callbacks,
    deterministic=True,
)

trainer.fit(lightning_module)
validation_results = trainer.validate(lightning_module, verbose=False)
test_results = trainer.test(lightning_module, verbose=False)
validation_results_df = pd.DataFrame.from_records(validation_results, index=("value",))
print("Displaying validation results for repetition {:d}:\n{:s}".format(repetition_number, validation_results_df.transpose().to_string()))
test_results_df = pd.DataFrame.from_records(test_results, index=("value",))
print("Displaying validation results for repetition {:d}:\n{:s}".format(repetition_number, test_results_df.transpose().to_string()))
