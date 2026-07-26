"""
Non-ML baseline volumetric interpolation methods.

Implements linear, nearest-neighbor, and radial basis function (RBF) interpolation as
baselines for the ANN in `support_functions_mod.volumetric_nn`, per reviewer comment 2.2
("No baseline comparisons"): compare the ANN's error against at least two simpler
interpolation methods on the same data.

Authors: Nicolas Gachancipa, Leslie Lamarche

Stanford Research Institute (SRI)
Embry-Riddle Aeronautical University
"""

# Imports.
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator

METHODS = ('linear', 'nearest', 'rbf')


class BaselineModel:
    """
    Wraps a scipy interpolator so it exposes a `.predict()` method, like a trained keras
    model. This lets a baseline interpolator stand in anywhere a `network.predict(x)` call
    is made (e.g. `graph_support_functions.create_all_data`), so the ANN and the baselines
    are evaluated with the exact same downstream code.
    """

    def __init__(self, interpolator):
        self._interpolator = interpolator

    def predict(self, x):
        predicted = self._interpolator(np.asarray(x))
        return np.asarray(predicted).reshape(-1, 1)


def fit_interpolator(x_train, y_train, method):
    """
    Fit a classical (non-ML) interpolator on normalized training coordinates.

    Parameters:
        x_train: [ndarray (n_points, 3)]
            Normalized (0-1) latitude, longitude, altitude of each training point. Same
            normalization `volumetric_nn` uses for the ANN's input, so results are directly
            comparable.
        y_train: [ndarray (n_points,)]
            Normalized (0-1) log-density value of each training point.
        method: [str]
            One of 'linear', 'nearest', 'rbf'.

    Returns:
        model: [BaselineModel]
            Object exposing `.predict(x)`, matching the interface of a trained keras model.
    """
    method = method.lower()
    if method not in METHODS:
        raise ValueError(f"Unknown baseline interpolation method '{method}'. Choose one of {METHODS}.")

    # Nearest-neighbor is also used as a fallback for query points that fall outside the
    # convex hull of the training points, where linear interpolation is undefined (NaN).
    # The ANN never leaves gaps like this, so the fallback keeps the comparison fair.
    nearest = NearestNDInterpolator(x_train, y_train)

    if method == 'nearest':
        interpolator = nearest

    elif method == 'linear':
        linear = LinearNDInterpolator(x_train, y_train)

        def interpolator(x):
            predicted = linear(x)
            nan_mask = np.isnan(predicted)
            if np.any(nan_mask):
                predicted[nan_mask] = nearest(x[nan_mask])
            return predicted

    else:  # 'rbf'
        rbf = RBFInterpolator(x_train, y_train)

        def interpolator(x):
            return rbf(x)

    return BaselineModel(interpolator)
