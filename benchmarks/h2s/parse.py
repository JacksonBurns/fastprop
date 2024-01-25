import pandas as pd
import numpy as np

from fastprop.utils import calculate_mordred_desciptors, ALL_2D, mordred_descriptors_from_strings

from rdkit import Chem
from py2opsin import py2opsin

import itertools

study_data = pd.read_excel("1-s2.0-S0167732217350985-mmc1.xlsx")

# pull out the names of the molecules
il_names = study_data["ILs (full name)"].to_numpy()
split_names = list(i.split() for i in il_names)
all_molecules = np.array(list(itertools.chain.from_iterable(split_names)))
unique_molecules = set(all_molecules)
# they used some non-standard non-iupac names, replace them
cleaned_unique_molecules = []
for name in unique_molecules:
    if name == "dimethylethanolammonium":
        name = "(2-hydroxyethyl)dimethylazanium"
    elif name == "methyldiethanolammonium":
        name = "(2-hydroxyethyl)(methyl)azanium"
    elif name == "tris(pentafluoroethyl)trifluorophosphate":
        # OPSIN just can't translate this one, so we use a placeholder
        # and replace it with the manually-found SMILES string later
        name = "methane"
    cleaned_unique_molecules.append(name)
unique_smiles: list = py2opsin(cleaned_unique_molecules)
unique_smiles[unique_smiles.index("C")] = "C(C(F)(F)[P-](C(C(F)(F)F)(F)F)(C(C(F)(F)F)(F)F)(F)(F)F)(F)(F)F"

# calculate the molecular descriptors
descs = calculate_mordred_desciptors(
    mordred_descriptors_from_strings(ALL_2D),
    list(Chem.MolFromSmiles(i) for i in unique_smiles),
    n_procs=1,
    strategy="low-memory",
)

# map name -> molecular descriptors
name_to_value = {name: values for name, values in zip(unique_molecules, descs)}
name_to_smiles = {name: smiles for name, smiles in zip(unique_molecules, unique_smiles)}

# assemble everything into one dataframe and then export that to the variosu inputs we need
input_data = study_data[["T/K", "P/bar", "Solubility (Exp.)"]].copy()
input_data.rename(columns={"T/K": "temp", "P/bar": "pressure", "Solubility (Exp.)": "solubility"}, inplace=True)
input_data["cation_name"] = [i[0] for i in split_names]
input_data["cation_smiles"] = [name_to_smiles[i[0]] for i in split_names]
input_data["anion_name"] = [i[1] for i in split_names]
input_data["anion_smiles"] = [name_to_smiles[i[1]] for i in split_names]
cation_desc_columns = ["cation_desc_" + str(i) for i in range(len(descs[0]))]
anion_desc_columns = ["anion_desc_" + str(i) for i in range(len(descs[0]))]
input_data = input_data.reindex(columns=input_data.columns.tolist() + cation_desc_columns)
input_data = input_data.reindex(columns=input_data.columns.tolist() + anion_desc_columns)
input_data[cation_desc_columns] = [name_to_value[name[0]] for name in split_names]
input_data[anion_desc_columns] = [name_to_value[name[1]] for name in split_names]

# for fastprop
input_data[["temp", "pressure"] + cation_desc_columns + anion_desc_columns].to_csv("precomputed.csv")
input_data[["cation_name", "anion_name", "cation_smiles", "anion_smiles", "solubility"]].to_csv("benchmark_data.csv")

# for chemprop
input_data[["temp", "pressure"]].to_csv("chemprop_features.csv")
