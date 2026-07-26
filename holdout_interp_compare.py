"""
Held-out train/test comparison of the ANN against the non-ML baseline interpolation
methods (linear, nearest-neighbor, RBF) in baseline_interp.py.

Unlike compare_interp_methods.py (which scores against a synthetic ionosphere's known
ground truth via amisrsynthdata, and needs a matching config yaml), this script only needs
a single AMISR-format radar file (real or synthetic). It splits that file's own sample
points into a train/test set (default 90/10, random - see the manuscript's Section 3.2
reviewer comment about how that split is performed), fits every method on the train split
only, predicts at the test split's exact coordinates, and compares against their measured
values. This directly answers reviewer comment 2.2 ("No baseline comparisons").

Authors: Nicolas Gachancipa, Leslie Lamarche

Stanford Research Institute (SRI)
Embry-Riddle Aeronautical University
"""

# Imports.
import datetime as dt
import numpy as np
import pandas as pd
from support_functions_mod import read_datafile, fit_volumetric_models, ensemble_predict

# Inputs: same convention as model_mod.py.
data_file = ["risrn_synthetic_imaging_chapman.h5"]
start = dt.datetime.strptime("2016-09-13T00:00:01", '%Y-%m-%dT%H:%M:%S')
end = dt.datetime.strptime("2016-09-13T00:10:00", '%Y-%m-%dT%H:%M:%S')

# Methods to compare. 'ann' is the existing neural network; the rest are the non-ML
# baselines requested by the reviewer.
methods = ['ann', 'linear', 'nearest', 'rbf']

# Train/test split settings.
test_fraction = 0.10
random_state = 0

# Same density-range filter volumetric_nn applies before fitting.
density_range = (1e10, 1e12)


def split_data(df, test_fraction=0.10, random_state=0):
    """Random train/test split of the radar sample points (see Section 3.2 comment)."""
    shuffled = df.sample(frac=1.0, random_state=random_state)
    n_test = int(len(shuffled) * test_fraction)
    return shuffled.iloc[n_test:].reset_index(drop=True), shuffled.iloc[:n_test].reset_index(drop=True)


def evaluate_method(train_df, test_df, method):
    """
    Fit `method` on `train_df` (via fit_volumetric_models, the same fitting code
    volumetric_nn uses) and score it against the held-out `test_df`.

    Returns:
        mae, rmse: [float] mean/root-mean-squared absolute error, in real density units.
    """
    models, _, _, _, x_lim, y_lim = fit_volumetric_models(train_df.copy(), density_range=density_range, model=method)

    # Normalize the test coordinates using the TRAINING data's own min/max (x_lim) - the
    # same limits the training coordinates were normalized with. Using the test set's own
    # min/max instead would leak information about the held-out points into evaluation.
    coord_cols = ['Latitude', 'Longitude', 'Altitude']
    x_test = np.array([(test_df[c].values - lo) / (hi - lo) for c, (lo, hi) in zip(coord_cols, x_lim)]).T

    pred_log_norm = ensemble_predict(models, x_test)
    predicted = 10 ** (pred_log_norm.flatten() * (y_lim[1] - y_lim[0]) + y_lim[0])
    actual = test_df['Value'].values

    mae = np.mean(np.abs(predicted - actual))
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    return mae, rmse


def main():
    data = read_datafile(data_file, start, end)

    # Apply the same density-range filter to the whole dataset before splitting, so both
    # the train and held-out test points are valid (physically plausible) measurements.
    data = data.dropna()
    data = data[(data['Value'] >= density_range[0]) & (data['Value'] <= density_range[1])]

    train_df, test_df = split_data(data, test_fraction, random_state)
    print(f"Train points: {len(train_df)}  Test points: {len(test_df)}")

    print(f"\n{'Method':<10}{'MAE':>15}{'RMSE':>15}")
    results = {}
    for method in methods:
        mae, rmse = evaluate_method(train_df, test_df, method)
        results[method] = (mae, rmse)
        print(f"{method:<10}{mae:>15.4e}{rmse:>15.4e}")
    return results


if __name__ == '__main__':
    main()
