# given a filepath and some target columns,
# retrieve the data as numpy arrays

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer

import numpy as np

import warnings


def preprocess(descriptors, targets, rescaling=True, zero_variance_drop=True, colinear_drop=False):
    target_scaler = StandardScaler()
    y = target_scaler.fit_transform(targets)

    imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message="Skipping features without any observed values.*")
        descriptors = imp_mean.fit_transform(descriptors, targets)

    if rescaling:
        # scale each column 0-1
        feature_scaler = StandardScaler()
        descriptors = feature_scaler.fit_transform(descriptors, targets)
        print("size after clean (drop empty, impute missing, scale 0-1):", descriptors.shape)

    if zero_variance_drop:
        # drop low variance features
        descriptors = VarianceThreshold(threshold=0).fit_transform(descriptors, y)
        print("size after invariant feature removal:", descriptors.shape)
    
    if colinear_drop:
        raise NotImplementedError("TODO")

    X = descriptors

    if not np.isfinite(X).all():
        raise RuntimeError("Postprocessing failed finite check, please file a bug report.")

    return X, y, target_scaler
