import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from fastprop.defaults import init_logger
from fastprop.fastprop_core import fastprop
from fastprop.utils import calculate_mordred_desciptors, load_from_csv
from fastprop.utils.select_descriptors import mordred_descriptors_from_strings

shap = None
try:
    import shap
except ImportError as ie:
    shape_error = ie


logger = init_logger(__name__)


def shap_fastprop(checkpoints_dir, input_file, importance_threshold=0.75):
    if shap is None:
        raise RuntimeError("Unable to import shap dependencies, please install fastprop[shap]. Original error: " + str(shape_error))
    # get the configuration file with some run metadata
    checkpoint_dir_contents = os.listdir(checkpoints_dir)
    with open(os.path.join(checkpoints_dir, "fastprop_config.yml")) as file:
        config_dict = yaml.safe_load(file)

    # calculate the features in the same way the model did when training,
    # downsampling to the shap suggested size where required
    # (also see https://github.com/shap/shap/issues/955 for an explanation of
    # Why the thresholds are the way they are)
    targets, mols, smiles = load_from_csv(input_file, config_dict["smiles"], config_dict["targets"])
    num_mols = len(mols)
    threshold_1 = 200
    threshold_2 = 2000
    if num_mols > threshold_1:
        rng = np.random.default_rng(42)
        target = threshold_1 if num_mols < threshold_2 else threshold_2
        logger.info(f"Randomly downsampling to {target} molecules.")
        mols = rng.choice(mols, size=target, replace=False)
    descs = calculate_mordred_desciptors(
        mordred_descriptors_from_strings(config_dict["descriptors"]),
        mols,
        n_procs=0,  # ignored for "strategy='low-memory'"
        strategy="low-memory",
    )
    descs = pd.DataFrame(data=descs, columns=config_dict["descriptors"])
    for pickled_scaler in config_dict["feature_scalers"]:
        scaler = pickle.loads(pickled_scaler)
        descs = scaler.transform(descs)

    # load each of the models in the group
    all_models = []
    for checkpoint in checkpoint_dir_contents:
        if not checkpoint.endswith(".ckpt"):
            continue
        model = fastprop.load_from_checkpoint(
            os.path.join(checkpoints_dir, checkpoint),
            number_features=config_dict["number_features"],
            hidden_size=config_dict["hidden_size"],
            target_scaler=pickle.loads(config_dict["target_scaler"]),
            fnn_layers=config_dict["fnn_layers"],
            problem_type=config_dict["problem_type"],
            num_epochs=None,
            learning_rate=None,
        )
        model.eval()
        all_models.append(model)

    # now send the data to whatever device lightning put the model on
    X = torch.tensor(descs.to_numpy(), dtype=torch.float32).to(all_models[0].device)

    # we will use half of the data for 'integrating', and the other half for getting shap values
    halfway_idx = len(X) // 2

    # shap terminology explanation:
    # background: 100 to 1000 random samples from the training data
    # X: 100+ other samples
    # run shap on each of these models, then average the results
    per_model_shap = []
    for model in tqdm(all_models, desc="Calculating SHAP values for each model"):
        e = shap.DeepExplainer(model, X[:halfway_idx])
        model_shap_values = e.shap_values(X[halfway_idx:])
        # returns a list for multi-target problems, cast for uniformity
        if not isinstance(model_shap_values, list):
            model_shap_values = [model_shap_values]
        per_model_shap.append(model_shap_values)
    all_shap_values = np.mean(np.array(per_model_shap), axis=0)

    # for each target, create a plot of the most important features
    for i, target_name in enumerate(config_dict["targets"]):
        shap_values = all_shap_values[i]
        # include features until the shap value is half the highest, aka half as important
        avg_shaps = np.mean(np.abs(shap_values), axis=0)
        avg_shaps, names, keep_idxs = zip(*sorted(zip(avg_shaps, config_dict["descriptors"], list(range(len(avg_shaps)))), reverse=True))
        include_idx = 0
        for val in avg_shaps[1:]:
            if val > avg_shaps[0] * importance_threshold:
                include_idx += 1
            else:
                break
        explanation = shap.Explanation(values=shap_values[:, keep_idxs[:include_idx]], feature_names=names[0:include_idx])
        plt.cla()
        axes = shap.plots.beeswarm(explanation, max_display=include_idx + 1, color_bar=False, color="shap_red", show=False)
        out_fname = target_name + "_feature_importance_beeswarm.png"
        if os.path.exists(out_fname):
            logger.warning(f"Output file exists! {__name__} will overwrite '{out_fname}'.")
        plt.savefig(out_fname, pad_inches=0.5, bbox_inches="tight")
    logger.info("Visit the mordred-community docs to lookup descriptors: https://jacksonburns.github.io/mordred-community/descriptors.html")
