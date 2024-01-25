# given a filepath and some target columns,
# retrieve the data as numpy arrays

import warnings
from types import SimpleNamespace

import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from .defaults import init_logger

logger = init_logger(__name__)


def preprocess(descriptors, targets, rescaling=True, zero_variance_drop=False, colinear_drop=False, problem_type="regression"):
    # mock the scaler object for classification tasks
    target_scaler = SimpleNamespace(feature_names_in_=None, n_features_in_=targets.shape[1])
    y = targets
    if problem_type == "regression":
        target_scaler = StandardScaler()
        y = target_scaler.fit_transform(targets)
    elif problem_type == "multiclass":
        logger.info("One-hot encoding target values.")
        target_scaler = OneHotEncoder(sparse_output=False)
        y = target_scaler.fit_transform(targets)

    # make it optional to either drop columns with any missing or do this
    imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message="Skipping features without any observed values.*")
        descriptors = imp_mean.fit_transform(descriptors, targets)

    if rescaling:
        # scale each column 0-1
        feature_scaler = StandardScaler()
        descriptors = feature_scaler.fit_transform(descriptors, targets)
        logger.info(f"size after clean (drop empty, impute missing, scale 0-1): {descriptors.shape}")

    if zero_variance_drop:
        # drop low variance features
        descriptors = VarianceThreshold(threshold=0).fit_transform(descriptors, y)
        logger.info(f"size after invariant feature removal: {descriptors.shape}")

    if colinear_drop:
        raise NotImplementedError("TODO")

    X = descriptors

    if not np.isfinite(X).all():
        raise RuntimeError("Postprocessing failed finite check, please file a bug report.")

    return X, y, target_scaler
