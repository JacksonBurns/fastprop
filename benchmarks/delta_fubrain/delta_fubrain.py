from itertools import product
from statistics import mean, stdev

import pandas as pd


maes = []
rmses = []
for i in range(1, 11):
    dframe = pd.read_csv(f"r{i}.csv")
    pairs = list(product(dframe.index, repeat=2))
    err = []
    for pair in pairs:
        actual_delta = dframe.iloc[pair[0]]["real"] - dframe.iloc[pair[1]]["real"]
        pred_delta = dframe.iloc[pair[0]]["pred"] - dframe.iloc[pair[1]]["pred"]
        err.append(abs(actual_delta - pred_delta))
    print(f"Repetition {i} results:")
    mae = mean(err)
    print(f"MAE = {mae:.4f}")
    maes.append(mae)
    rmse = mean(i**2 for i in err) ** (1 / 2)
    print(f"RMSE = {rmse:.4f}")
    rmses.append(rmse)
print("Overall performance:")
print(f"Average MAE +/- stdev = {mean(maes):.4f}+/-{stdev(maes):.4f}")
print(f"Average RMSE +/- stdev = {mean(rmses):.4f}+/-{stdev(rmses):.4f}")
