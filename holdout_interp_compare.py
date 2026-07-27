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

# Methods to compare: 'ann' (standard MLP), 'fourier' (Fourier Feature ANN), and non-ML baselines ('linear', 'nearest', 'rbf').
methods = ['ann', 'fourier', 'linear', 'nearest', 'rbf']

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
        metrics: [dict] containing normalized MAE, log10 MAE, MAPE (%), and physical MAE/RMSE.
    """
    models, _, _, _, x_lim, y_lim = fit_volumetric_models(train_df.copy(), density_range=density_range, model=method)

    # Normalize the test coordinates using the TRAINING data's own min/max (x_lim).
    coord_cols = ['Latitude', 'Longitude', 'Altitude']
    x_test = np.array([(test_df[c].values - lo) / (hi - lo) for c, (lo, hi) in zip(coord_cols, x_lim)]).T

    # Normalized log-density predictions (range 0 to 1) and actuals
    pred_log_norm = ensemble_predict(models, x_test).flatten()
    actual_log = np.log10(test_df['Value'].values)
    actual_log_norm = (actual_log - y_lim[0]) / (y_lim[1] - y_lim[0])

    # Convert back to physical density scale (e-/m^3)
    predicted_log = pred_log_norm * (y_lim[1] - y_lim[0]) + y_lim[0]
    predicted_phys = 10 ** predicted_log
    actual_phys = test_df['Value'].values

    # Compute metrics across normalized, log, percentage, and physical scales
    norm_mae = np.mean(np.abs(pred_log_norm - actual_log_norm))
    log_mae = np.mean(np.abs(predicted_log - actual_log))
    mape_percent = np.mean(np.abs(predicted_phys - actual_phys) / actual_phys) * 100.0
    phys_mae = np.mean(np.abs(predicted_phys - actual_phys))
    phys_rmse = np.sqrt(np.mean((predicted_phys - actual_phys) ** 2))

    return {
        'norm_mae': norm_mae,
        'log_mae': log_mae,
        'mape_percent': mape_percent,
        'phys_mae': phys_mae,
        'phys_rmse': phys_rmse
    }


def main():
    data = read_datafile(data_file, start, end)

    # Apply the same density-range filter to the whole dataset before splitting.
    data = data.dropna()
    data = data[(data['Value'] >= density_range[0]) & (data['Value'] <= density_range[1])]

    train_df, test_df = split_data(data, test_fraction, random_state)
    print(f"Dataset: {data_file[0]}")
    print(f"Total points: {len(data)} | Train points (90%): {len(train_df)} | Held-out test points (10%): {len(test_df)}\n")

    header = f"{'Method':<10}{'Norm MAE (0-1)':>16}{'Log10 MAE (dex)':>18}{'MAPE Error (%)':>16}{'Phys MAE (e-/m3)':>20}"
    print(header)
    print("-" * len(header))

    results = {}
    for method in methods:
        res = evaluate_method(train_df, test_df, method)
        results[method] = res
        print(f"{method:<10}{res['norm_mae']:>16.4f}{res['log_mae']:>18.4f}{res['mape_percent']:>15.2f}%{res['phys_mae']:>20.4e}")

    return results


if __name__ == '__main__':
    main()
