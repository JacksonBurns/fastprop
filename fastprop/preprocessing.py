# given a filepath and some target columns,
# retrieve the data as numpy arrays


import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from fastprop.defaults import init_logger

logger = init_logger(__name__)


def preprocess(descriptors, zero_variance_drop=False, colinear_drop=False):
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

    logger.info(f"Size after preprocessing: {descriptors.shape}")

    return descriptors
