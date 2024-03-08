import pickle as pkl
import csv

import pandas as pd
import numpy as np

from fastprop.utils import calculate_mordred_desciptors, mordred_descriptors_from_strings

shap_important = (
    "ATS0Z",
    "ATS5m",
    "ATS2i",
    "ATS8m",
    "ATS2v",
    "ATS3v",
    "ATS7v",
    "ATSC2m",
    "ATS6v",
    "ATS0m",
    "ATSC0v",
    "ATS3i",
    "ATS6i",
    "ATSC7v",
    "ATSC6m",
    "ATS8v",
    "ATS0i",
    "ATS6are",
    "ATS1m",
    "ATSC0m",
    "ATSC3v",
    "ATS5i",
    "ATS4v",
    "ATS7m",
    "ATS3m",
    "ATS2Z",
    "ATS7i",
    "ATSC3m",
    "ATS0v",
    "ATS8pe",
    "ATS2m",
    "ATS3pe",
    "ATS8i",
    "ATS5v",
    "ATS1v",
    "ATSC2v",
    "ATS1i",
    "ATS5se",
    "ATS4i",
    "ATS4se",
    "ATS4m",
    "ATSC5v",
)

from rdkit import Chem


with open("hwhp_gsolv.pkl", "rb") as file:
    gsolv_df = pkl.load(file)
    # for downsampling
    # gsolv_df = gsolv_df.sample(n=1_000, random_state=2)
    # gsolv_df.reset_index(drop=True, inplace=True)

unique_smiles = np.hstack((pd.unique(gsolv_df["solvent_smiles"]), pd.unique(gsolv_df["solute_smiles"])))

# calculate the molecular descriptors
descs = calculate_mordred_desciptors(
    mordred_descriptors_from_strings(shap_important),
    list(Chem.MolFromSmiles(i) for i in unique_smiles),
    n_procs=-1,  # ignored anyway
    strategy="low-memory",
)

with open("hwhp_property_lookup.csv", "w", newline="") as file:
    # with open("hwhp_property_lookup_downsample.csv", "w", newline="") as file:  # for downsampling
    writer = csv.writer(file, delimiter=",")
    writer.writerow(["smiles"] + list(shap_important))
    for smiles, values in zip(unique_smiles, descs):
        writer.writerow([smiles] + [format(i, ".6f") if isinstance(i, float) else "0" for i in values])
