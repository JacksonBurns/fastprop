import os

import numpy as np
import torch
from tqdm import tqdm
import pickle as pkl
import pandas as pd

from fastprop.utils import ALL_2D

import oom_training

import shap

checkpoints_dir = "checkpoints"
importance_threshold = 0.05

starting_features = ALL_2D + ALL_2D
checkpoint_dir_contents = os.listdir(checkpoints_dir)

with open("hwhp_gsolv.pkl", "rb") as file:
    gsolv_df = pkl.load(file)
    gsolv_df = gsolv_df.sample(n=1_000, random_state=2)
    gsolv_df.reset_index(drop=True, inplace=True)

descriptor_lookup_df: pd.DataFrame = pd.read_csv("hwhp_property_lookup_downsample.csv", index_col="smiles")

data = []
for i, row in gsolv_df.iterrows():
    solute = row["solute_smiles"]
    solvent = row["solvent_smiles"]
    data.append(np.hstack((descriptor_lookup_df.loc[solute].to_numpy(), descriptor_lookup_df.loc[solvent].to_numpy())))
descs = np.array(data)

# load each of the models in the group
all_models = []
for checkpoint in checkpoint_dir_contents:
    if not checkpoint.endswith(".ckpt"):
        continue
    model = oom_training.OOMfastprop.load_from_checkpoint(
        os.path.join(checkpoints_dir, checkpoint),
    )
    all_models.append(model)

# we will use half of the data for 'integrating', and the other half for getting shap values
halfway_idx = descs.shape[0] // 2

# shap terminology explanation:
# background: 100 to 1000 random samples from the training data
# X: 100+ other samples
# run shap on each of these models, then average the results
per_model_shap = []
for model in tqdm(all_models, desc="Calculating SHAP values for each model"):
    # now scale and send the data to whatever device lightning put the model on
    X = torch.tensor(descs, dtype=torch.float32).to(model.device)
    e = shap.DeepExplainer(model, X[:halfway_idx])
    model_shap_values = e.shap_values(X[halfway_idx:])
    # returns a list for multi-target problems, cast for uniformity
    if not isinstance(model_shap_values, list):
        model_shap_values = [model_shap_values]
    per_model_shap.append(model_shap_values)
all_shap_values = np.mean(np.array(per_model_shap), axis=0)

# for each target, create a plot of the most important features
for i, target_name in enumerate(["gsolv"]):
    shap_values = all_shap_values[i]
    # include features until the shap value is half the highest, aka half as important
    avg_shaps = np.mean(np.abs(shap_values), axis=0)
    avg_shaps, names, keep_idxs = zip(*sorted(zip(avg_shaps, starting_features, list(range(len(avg_shaps)))), reverse=True))
    include_idx = 0
    for val in avg_shaps[1:]:
        if val > avg_shaps[0] * importance_threshold:
            include_idx += 1
        else:
            break
    with open("IMPORTANT_FEATURES.txt", "w") as file:
        file.writelines("\n".join(set(names[0:include_idx])))
