import copy

import pandas as pd
from astartes import train_val_test_split
from astartes.molecules import train_val_test_split_molecules
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as l2_error

from fastprop.defaults import init_logger

logger = init_logger(__name__)


def linear_baseline(problem_type, random_seed, number_repeats, sampler, train_size, val_size, test_size, X, y, smiles, target_scaler):
    """Train a basic linear model on the input data as a baseline.

    Args:
        problem_type (str): Problem type (regression, binary, multilabel, multiclass)
        random_seed (int): Random seed.
        number_repeats (int): Number of repetitions.
        sampler (str): Sampler, see astartes documentation for all possible samplers.
        train_size (float): Fraction of data for training.
        val_size (flaot): Fraction of data for validation.
        test_size (float): Fraction of data for testing.
        X (np.ndarray): Input features after preprocessing.
        y (np.ndarray): Output targets after proprocessing.
        smiles (list[str]): SMILES strings corresponding to the molecules.
        target_scaler (sklearn.scaler): Scaler used on the targets, used to report human-readable metrics.
    """
    if problem_type != "regression":
        logger.warning("TODO: implement baseline model for classification")
    else:
        baseline_seed = copy.deepcopy(random_seed)  # just to be safe
        baseline_performance = []
        for repetition in range(number_repeats):
            # split the data in the same way fastprop would
            X_train = X_val = X_test = y_train = y_val = y_test = None
            if sampler != "scaffold":
                X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
                    X,
                    y,
                    train_size=train_size,
                    val_size=val_size,
                    test_size=test_size,
                    sampler=sampler,
                    random_state=baseline_seed,
                )
            else:
                *_, idxs_train, idxs_val, idxs_test = train_val_test_split_molecules(
                    smiles,
                    train_size=train_size,
                    val_size=val_size,
                    test_size=test_size,
                    sampler=sampler,
                    return_indices=True,
                    random_state=baseline_seed,
                )
                X_train, X_val, X_test = X[idxs_train], X[idxs_val], X[idxs_test]
                y_train, y_val, y_test = y[idxs_train], y[idxs_val], y[idxs_test]
            # train some basic models
            model = None
            model = ElasticNet(random_state=baseline_seed)
            logger.info(f"Training {model.__repr__()} model repetition {repetition} as baseline...")
            model.fit(X_train, y_train)
            logger.info("...done.")
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
            for pred_arr, truth_arr, name in zip((val_pred, test_pred), (y_val, y_test), ("validation", "test")):
                if len(pred_arr.shape) == 1 or pred_arr.shape[1] == 1:  # capture 2d and 1d
                    rescaled_pred = target_scaler.inverse_transform(pred_arr.reshape(-1, 1))
                    rescaled_truth = target_scaler.inverse_transform(truth_arr.reshape(-1, 1))
                else:
                    rescaled_pred = target_scaler.inverse_transform(pred_arr)
                    rescaled_truth = target_scaler.inverse_transform(truth_arr)
                logger.info(f"Baseline linear model statistics for {name}:")
                l2 = l2_error(rescaled_truth, rescaled_pred, squared=False)
                logger.info(f"RMSE: {l2:.4f}")
                l1 = mae(rescaled_truth, rescaled_pred)
                logger.info(f"MAE: {l1:.4f}")
                wm = mape(rescaled_truth, rescaled_pred, sample_weight=rescaled_truth)
                logger.info(f"wMAPE: {wm:.4f}")
                m = mape(rescaled_truth, rescaled_pred)
                logger.info(f"MAPE: {m:.4f}")
                baseline_performance.append({name + "-l2": l2, name + "-l1": l1, name + "-wmape": wm, name + "-mape": m})
            baseline_seed += 1
        perf_df = pd.DataFrame.from_records(baseline_performance)
        logger.info("Displaying summary baseline results:\n%s", perf_df.describe().transpose().to_string())
