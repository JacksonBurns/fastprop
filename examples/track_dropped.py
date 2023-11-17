import os
import warnings
from time import perf_counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from astartes import train_val_test_split
from mordred import Calculator, descriptors
from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from rdkit import Chem
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset as TorchDataset
from torchmetrics.functional import mean_absolute_percentage_error as mape

data = pd.read_csv("qm8.csv")

targets = data["E1-PBE0"].to_numpy()
target_scaler = MinMaxScaler()
y = target_scaler.fit_transform(targets.reshape(-1, 1))

mordred_calc = Calculator(descriptors, ignore_3D=True)
rdkit_mols = tuple(Chem.MolFromSmiles(i) for i in data["smiles"][0:1001])
mordred_descs = mordred_calc.pandas(rdkit_mols, 8)

print(mordred_descs.shape)

imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean").set_output(transform="pandas")
with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore", message="Skipping features without any observed values.*")
    cleaned_mordred_descs = imp_mean.fit_transform(mordred_descs, targets)

print(cleaned_mordred_descs.shape)
# scale each column 0-1
# feature_scaler = MinMaxScaler()
# cleaned_normalized_mordred_descs = feature_scaler.fit_transform(cleaned_mordred_descs, targets)
# print("size after clean (drop empty, impute missing, scale 0-1):", cleaned_normalized_mordred_descs.shape)

# drop low variance features
X = VarianceThreshold(threshold=0).set_output(transform="pandas").fit_transform(cleaned_mordred_descs, y)
print("size after invariant feature removal:", X.shape)

threshold=0.95
df_corr = X.corr().abs()
upper = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
df = X.drop(to_drop, axis=1)
print(df.shape)
# print(df.columns)

# print(X)
with open("out_less.txt", "w") as file:
    for desc in df.columns:
        file.write(desc + "\n")
print(df.columns)
