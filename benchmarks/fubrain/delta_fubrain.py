"""
delta_fubrain.py

Usage:
python delta_fubrain.py

Description:
Demonstrates fitting fastprop under the delta-training approach on the Fubrain dataset.

The reference paper does 5x10 CV, where 10-fold CV is performed with 5 random repeats.
For each repeat, the predictions of each model on its fold are saved and then combined
with the other folds. Summary statistics are calculated on these, and then averages across
the 5 repeats are calculated. See:
https://github.com/RekerLab/DeepDelta/blob/main/Code/cross_validations.py
for their implementation, which is drawn from for this code.
"""

import glob
from itertools import product

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import KFold
from pytorch_lightning import Trainer

from fastprop.data import (
    clean_dataset,
    fastpropDataLoader,
    fastpropDataset,
    standard_scale,
)
from fastprop.metrics import SCORE_LOOKUP
from fastprop.model import fastprop

# load the descriptors from a previous execution of `fastprop train fubrain.yml`
descriptors = pd.read_csv(glob.glob("./cached_benchmark_data_all_*.csv")[-1], index_col=0)
# load the targets from the fubrain data directly, and then take the log10 of them
df = pd.read_csv("benchmark_data.csv")
targets = df["fraction"].to_numpy()
smiles = df["SMILES"].to_numpy()

# clean the dataset, just in case
targets, rdkit_mols, smiles = clean_dataset(targets, smiles)

# take the base10 log of the target values
targets = np.log10(targets)

targets_og = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
descriptors_og = torch.tensor(descriptors.to_numpy(dtype=np.float32), dtype=torch.float32)

all_delta_results = []
for fold_seed in range(5):
    kf = KFold(n_splits=10, shuffle=True, random_state=fold_seed)

    fold_predictions, fold_truth = [], []
    for train_indexes, test_indexes in kf.split(np.arange(len(smiles))):
        # copy and re-scale the descriptors and targets
        descriptors = descriptors_og.detach().clone()
        descriptors[train_indexes], feature_means, feature_vars = standard_scale(descriptors[train_indexes])
        descriptors[test_indexes] = standard_scale(descriptors[test_indexes], feature_means, feature_vars)

        targets = targets_og.detach().clone()
        targets[train_indexes], targets_means, targets_vars = standard_scale(targets[train_indexes])
        targets[test_indexes] = standard_scale(targets[test_indexes], targets_means, targets_vars)

        # initialize dataloaders and model, then train
        train_dataloader = fastpropDataLoader(fastpropDataset(descriptors[train_indexes], targets[train_indexes]), shuffle=True, batch_size=16)
        test_dataloader = fastpropDataLoader(fastpropDataset(descriptors[test_indexes], targets[test_indexes]), batch_size=512)
        model = fastprop(
            clamp_input=True,
            target_names=["f_unbound"],
            feature_means=feature_means,
            feature_vars=feature_vars,
            target_means=targets_means,
            target_vars=targets_vars,
        )

        tensorboard_logger = TensorBoardLogger(
            "delta_output",
            name="tensorboard_logs",
            default_hp_metric=False,
        )
        trainer = Trainer(
            max_epochs=30,
            enable_progress_bar=True,
            enable_model_summary=True,
            logger=tensorboard_logger,
            log_every_n_steps=1,
            enable_checkpointing=False,
            check_val_every_n_epoch=1,
        )
        trainer.fit(model, train_dataloader)
        test_results = trainer.test(model, test_dataloader, verbose=False)

        # evaluate the performance as if it were delta learning instead
        test_predictions: np.ndarray = trainer.predict(model, test_dataloader)[0].numpy().ravel()  # predict auto-magically undoes the scaling
        test_truth: np.ndarray = targets_og[test_indexes].detach().clone().numpy().ravel()
        fold_predictions.extend(i - j for (i, j) in product(test_predictions, test_predictions))
        fold_truth.extend(i - j for (i, j) in product(test_truth, test_truth))

    predicted_differences = torch.tensor(fold_truth)
    true_differences = torch.tensor(fold_predictions)
    all_delta_results.append(
        {
            f"delta_{metric.__name__}": metric(true_differences, predicted_differences, 1).item()
            for metric in SCORE_LOOKUP["regression"]
            if "percentage_error" not in metric.__name__  # delta metrics are centered by definition, percentage errors undefined
        }
    )

all_delta_results = pd.DataFrame.from_records(all_delta_results)
print(f"Displaying delta-based testing results:\n {all_delta_results.describe().transpose().to_string()}")
