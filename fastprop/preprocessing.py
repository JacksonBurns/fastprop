# given a filepath and some target columns,
# retrieve the data as numpy arrays

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fastprop.defaults import init_logger

logger = init_logger(__name__)


def _mock_inverse_transform(x):
    return x


def preprocess(
    descriptors,
    targets,
    rescaling=True,
    zero_variance_drop=False,
    colinear_drop=False,
    problem_type="regression",
    return_X_scalers=False,
):
    # mock the scaler object for classification tasks
    target_scaler = SimpleNamespace(feature_names_in_=None, n_features_in_=targets.shape[1], inverse_transform=_mock_inverse_transform)
    y = targets
    if problem_type == "regression":
        target_scaler = StandardScaler()
        y = target_scaler.fit_transform(targets)
    elif problem_type == "multiclass":
        logger.info("One-hot encoding target values.")
        target_scaler = OneHotEncoder(sparse_output=False)
        y = target_scaler.fit_transform(targets)

    # drop missing features - if this is direct output from mordred, missing values are
    # strings describing why they are missing. If this is output from fastprop.utils.load_daved_desc
    # the missing descriptors are nan. To deal with the former case, force all str->nan
    descriptors: pd.DataFrame
    descriptors = descriptors.apply(pd.to_numeric, errors="coerce")
    descriptors = descriptors.dropna(axis=1, how="all")

    if zero_variance_drop:
        # drop invariant features
        var_scaler = VarianceThreshold(threshold=0).set_output(transform="pandas")
        # exclude from the returned feature scalers - the columns which should be dropped are done
        # automatically by not calculating them in the first place during prediction.
        descriptors = var_scaler.fit_transform(descriptors)
        logger.info(f"size after invariant feature removal: {descriptors.shape}")

    if colinear_drop:
        raise NotImplementedError("TODO")

    imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean").set_output(transform="pandas")
    scalers = [imp_mean]
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message="Skipping features without any observed values.*")
        descriptors = imp_mean.fit_transform(descriptors)

    if rescaling:
        # scale each column to unit variance
        feature_scaler = StandardScaler().set_output(transform="pandas")
        scalers.append(feature_scaler)
        descriptors = feature_scaler.fit_transform(descriptors)
        logger.info(f"size after clean (impute missing, scale to unit variance): {descriptors.shape}")

    X: pd.DataFrame = descriptors

    assert np.isfinite(X.to_numpy()).all(), "Postprocessing failed finite check, please file a bug report."

    if return_X_scalers:
        return (
            X,
            y,
            target_scaler,
            scalers,
        )
    else:
        return X, y, target_scaler
