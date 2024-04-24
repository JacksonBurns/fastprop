import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from fastprop.data import standard_scale
from fastprop.defaults import DESCRIPTOR_SET_LOOKUP, init_logger
from fastprop.io import load_saved_descriptors
from fastprop.model import fastprop

shap = None
try:
    import shap
except ImportError as ie:
    shape_error = ie


logger = init_logger(__name__)


def shap_fastprop(
    checkpoints_dir: str,
    cached_descriptors: str,
    descriptor_set: str,
    importance_threshold: float = 0.75,
):
    if shap is None:
        raise RuntimeError("Unable to import shap dependencies, please install fastprop[shap]. Original error: " + str(shape_error))
    # load the models
    all_models = []
    for checkpoint in os.listdir(checkpoints_dir):
        model = fastprop.load_from_checkpoint(os.path.join(checkpoints_dir, checkpoint))
        all_models.append(model)

    # downsampling to the shap suggested size where required
    # (also see https://github.com/shap/shap/issues/955 for an explanation of
    # Why the thresholds are the way they are)
    descs = load_saved_descriptors(cached_descriptors)
    num_mols = descs.shape[0]
    threshold_1 = 200
    threshold_2 = 2000
    if num_mols > threshold_1:
        rng = np.random.default_rng(42)
        target = threshold_1 if num_mols < threshold_2 else threshold_2
        logger.info(f"Randomly downsampling to {target} molecules.")
        idxs = rng.choice(num_mols, size=target, replace=False)
        descs = descs[idxs]

    descs = torch.tensor(descs, dtype=torch.float32, device=all_models[0].device)
    X = standard_scale(descs, all_models[0].feature_means, all_models[0].feature_vars)

    # we will use half of the data for 'integrating', and the other half for getting shap values
    halfway_idx = descs.shape[0] // 2

    # shap terminology explanation:
    # background: 100 to 1000 random samples from the training data
    # X: 100+ other samples
    # run shap on each of these models, then average the results
    per_model_shap = []
    for model in tqdm(all_models, desc="Calculating SHAP values for each model"):
        # now scale and send the data to whatever device lightning put the model on
        e = shap.DeepExplainer(model, X[:halfway_idx])
        model_shap_values = e.shap_values(X[halfway_idx:])
        # returns a list for multi-target problems, cast for uniformity
        if not isinstance(model_shap_values, list):
            model_shap_values = [model_shap_values]
        per_model_shap.append(model_shap_values)
    all_shap_values = np.mean(np.array(per_model_shap), axis=0)

    # for each target, create a plot of the most important features
    for i, target_name in enumerate(f"task_{i}" for i in range(all_models[0].readout.out_features)):
        shap_values = all_shap_values[i]
        # include features until the shap value is half the highest, aka half as important
        avg_shaps = np.mean(np.abs(shap_values), axis=0)
        avg_shaps, names, keep_idxs = zip(*sorted(zip(avg_shaps, DESCRIPTOR_SET_LOOKUP[descriptor_set], list(range(len(avg_shaps)))), reverse=True))
        include_idx = 0
        for val in avg_shaps[1:]:
            if val > avg_shaps[0] * importance_threshold:
                include_idx += 1
            else:
                break
        explanation = shap.Explanation(values=shap_values[:, keep_idxs[:include_idx]], feature_names=names[0:include_idx])
        plt.cla()
        _ = shap.plots.beeswarm(explanation, max_display=include_idx + 1, color_bar=False, color="shap_red", show=False)
        out_fname = target_name + "_feature_importance_beeswarm.png"
        if os.path.exists(out_fname):
            logger.warning(f"Output file exists! {__name__} will overwrite '{out_fname}'.")
        plt.savefig(out_fname, pad_inches=0.5, bbox_inches="tight")
    logger.info("Visit the mordred-community docs to lookup descriptors: https://jacksonburns.github.io/mordred-community/descriptors.html")
