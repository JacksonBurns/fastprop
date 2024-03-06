import pickle as pkl
import csv

import pandas as pd
import numpy as np

from fastprop.utils import calculate_mordred_desciptors, mordred_descriptors_from_strings, ALL_2D

from rdkit import Chem


with open("hwhp_gsolv.pkl", "rb") as file:
    gsolv_df = pkl.load(file)
    # for downsampling
    # gsolv_df = gsolv_df.sample(n=1_000, random_state=2)
    # gsolv_df.reset_index(drop=True, inplace=True)

unique_smiles = np.hstack((pd.unique(gsolv_df["solvent_smiles"]), pd.unique(gsolv_df["solute_smiles"])))

# calculate the molecular descriptors
descs = calculate_mordred_desciptors(
    mordred_descriptors_from_strings(ALL_2D),
    list(Chem.MolFromSmiles(i) for i in unique_smiles),
    n_procs=-1,  # ignored anyway
    strategy="low-memory",
)

with open("hwhp_property_lookup_downsample.csv", "w", newline="") as file:
    # with open("hwhp_property_lookup_downsample.csv", "w", newline="") as file:  # for downsampling
    writer = csv.writer(file, delimiter=",")
    writer.writerow(["smiles"] + list(ALL_2D))
    for smiles, values in zip(unique_smiles, descs):
        writer.writerow([smiles] + [format(i, ".6f") if isinstance(i, float) else "0" for i in values])
