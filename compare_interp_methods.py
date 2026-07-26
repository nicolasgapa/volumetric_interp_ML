"""
Compare the ANN's volumetric interpolation against non-ML baseline interpolation methods
(linear, nearest-neighbor, and RBF), evaluated exactly the way the ANN is evaluated in
graph_diff.py: fit each method on the same sparse, radar-like samples of a synthetic
ionosphere, predict over a dense grid, and compare against the synthetic model's known
ground-truth densities.

Addresses reviewer comment 2.2 ("No baseline comparisons"): "Implement at least two
non-ML interpolation methods on the same synthetic and real data, and compare their
errors (MAE) with your ANN."

Authors: Nicolas Gachancipa, Leslie Lamarche

Stanford Research Institute (SRI)
Embry-Riddle Aeronautical University
"""

# Imports.
import datetime as dt
import numpy as np
from graph_support_functions import create_all_data

# Lists of potential data to test (same convention as graph_diff.py's get_data()).
datas = [["chapman_data_025.h5", "chapman_data_ac.h5"], ["circular_data_025.h5", "circular_data_ac.h5"],
         ["gradient_data_025.h5", "gradient_data_ac.h5"], ["tubular_data_025.h5", "tubular_data_ac.h5"],
         ["gradient_circular_data_ac.h5", "gradient_circular_data_025.h5"],
         ["chapman_circular_data_ac.h5", "chapman_circular_data_025.h5"],
         ["chapman_gradient_data_ac.h5", "chapman_gradient_data_025.h5"],
         ["wave2_data_ac.h5", "wave2_data_025.h5"], ["two_circles_data_ac.h5", "two_circles_data_025.h5"]]
config_files = ['chapman_config.yaml', 'circular_config.yaml', 'gradient_config.yaml', 'tubular_config.yaml',
                "gradient_circular_config.yaml", "chapman_circular_config.yaml", "chapman_gradient_config.yaml",
                "wave2_config.yaml", "two_circles_config.yaml"]

# n: Determines which datafile is tested.
n = 1
data_file = datas[n]
config_file = config_files[n]
start = dt.datetime.strptime("2016-09-13T00:00:01", '%Y-%m-%dT%H:%M:%S')
end = dt.datetime.strptime("2016-09-13T00:00:10", '%Y-%m-%dT%H:%M:%S')

# Methods to compare. 'ann' is the existing neural network; the rest are the non-ML
# baselines requested by the reviewer.
methods = ['ann', 'linear', 'nearest', 'rbf']

# Number of repeated trials per method, to report a mean +/- standard deviation MAE (same
# approach graph_diff.py uses for the ANN). The baselines are deterministic, so their
# std is expected to be ~0; they're still run n_trials times to keep the comparison code
# identical across methods.
n_trials = 3


def compare():
    """
    Fit and evaluate every method in `methods` on the same synthetic dataset, using
    `create_all_data` (the same function/metric already used to evaluate the ANN in
    graph_diff.py).

    Returns:
        results: [dict]
            method -> (mean MAE, std MAE) across n_trials.
    """
    results = {}
    for method in methods:
        mae_list = []
        for _ in range(n_trials):
            df, _ = create_all_data(start, end, data_file, 0, config_file, method=method)
            mae_list.append(np.mean(abs(np.array(df['Value']))))
        results[method] = (np.mean(mae_list), np.std(mae_list))
    return results


def main():
    results = compare()
    print(f"{'Method':<10}{'MAE':>15}{'STD':>15}")
    for method, (mae, std) in results.items():
        print(f"{method:<10}{mae:>15.4e}{std:>15.4e}")
    return results


if __name__ == '__main__':
    main()
