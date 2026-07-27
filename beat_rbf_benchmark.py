"""
Benchmark script demonstrating high-capacity neural architectures (RFF ResNet, SIREN)
competing directly against RBF interpolation on held-out test data.
"""

import datetime as dt
import numpy as np
import pandas as pd
import keras
from keras import ops
from support_functions_mod import read_datafile, fit_volumetric_models, ensemble_predict, EnsembleModel
from holdout_interp_compare import evaluate_method, split_data

# Data split setup
data_file = ["risrn_synthetic_imaging_chapman.h5"]
start = dt.datetime.strptime("2016-09-13T00:00:01", '%Y-%m-%dT%H:%M:%S')
end = dt.datetime.strptime("2016-09-13T00:10:00", '%Y-%m-%dT%H:%M:%S')
density_range = (1e10, 1e12)
test_fraction = 0.10
random_state = 0

# Load data (do not pre-add 'Log Value', let fit_volumetric_models handle it)
data = read_datafile(data_file, start, end).dropna()
data = data[(data['Value'] >= density_range[0]) & (data['Value'] <= density_range[1])].reset_index(drop=True)

train_df, test_df = split_data(data, test_fraction, random_state)

# Evaluate RBF baseline using exact evaluate_method function
rbf_res = evaluate_method(train_df.copy(), test_df.copy(), 'rbf')

header = f"{'Method':<32}{'Norm MAE (0-1)':>16}{'Log10 MAE (dex)':>18}{'MAPE Error (%)':>16}{'Phys MAE (e-/m3)':>20}"
print(header, flush=True)
print("-" * len(header), flush=True)
print(f"{'RBF (Target Baseline)':<32}{rbf_res['norm_mae']:>16.5f}{rbf_res['log_mae']:>18.5f}{rbf_res['mape_percent']:>15.3f}%{rbf_res['phys_mae']:>20.4e}", flush=True)

# Prepare training data inputs
_, _, _, _, x_lim, y_lim = fit_volumetric_models(train_df.copy(), density_range=density_range, model='rbf')
coord_cols = ['Latitude', 'Longitude', 'Altitude']

train_df_proc = train_df.copy().dropna()
train_df_proc = train_df_proc[(train_df_proc['Value'] >= density_range[0]) & (train_df_proc['Value'] <= density_range[1])].reset_index(drop=True)
train_df_proc['Log Value'] = np.log10(train_df_proc['Value'])

x_train_norm = np.array([(train_df_proc[c].values - lo) / (hi - lo) for c, (lo, hi) in zip(coord_cols, x_lim)]).T
y_train_norm = (train_df_proc['Log Value'].values - y_lim[0]) / (y_lim[1] - y_lim[0])

x_test_norm = np.array([(test_df[c].values - lo) / (hi - lo) for c, (lo, hi) in zip(coord_cols, x_lim)]).T
y_test_log = np.log10(test_df['Value'].values)
y_test_phys = test_df['Value'].values


def score_custom_model(model_obj, name):
    pred_log_norm = np.asarray(model_obj.predict(x_test_norm)).flatten()
    pred_log = pred_log_norm * (y_lim[1] - y_lim[0]) + y_lim[0]
    pred_phys = 10.0 ** pred_log

    norm_mae = np.mean(np.abs(pred_log_norm - (y_test_log - y_lim[0]) / (y_lim[1] - y_lim[0])))
    log_mae = np.mean(np.abs(pred_log - y_test_log))
    mape = np.mean(np.abs(pred_phys - y_test_phys) / y_test_phys) * 100.0
    phys_mae = np.mean(np.abs(pred_phys - y_test_phys))

    print(f"{name:<32}{norm_mae:>16.5f}{log_mae:>18.5f}{mape:>15.3f}%{phys_mae:>20.4e}", flush=True)
    return mape, phys_mae


# Custom RFF Layer
class RFFLayer(keras.layers.Layer):
    def __init__(self, num_rff=256, sigma=16.0, **kwargs):
        super().__init__(**kwargs)
        self.num_rff = num_rff
        self.sigma = sigma

    def build(self, input_shape):
        np.random.seed(42)
        init_B = np.random.normal(scale=self.sigma, size=(input_shape[-1], self.num_rff)).astype(np.float32)
        self.B = self.add_weight(shape=(input_shape[-1], self.num_rff), initializer=keras.initializers.Constant(init_B), trainable=False)

    def call(self, inputs):
        x_proj = ops.matmul(inputs * np.pi, self.B)
        return ops.concatenate([ops.sin(x_proj), ops.cos(x_proj)], axis=-1)


def create_deep_rff_resnet(input_dim=3, num_rff=256, sigma=16.0):
    inputs = keras.Input(shape=(input_dim,))
    rff_features = RFFLayer(num_rff=num_rff, sigma=sigma)(inputs)

    x = keras.layers.Dense(512, activation='swish')(rff_features)

    # 3 Fast ResNet Blocks with Skip Connections
    for _ in range(3):
        residual = x
        y = keras.layers.Dense(512, activation='swish')(x)
        y = keras.layers.Dense(512, activation='swish')(y)
        x = keras.layers.Add()([residual, y])

    x = keras.layers.Concatenate()([x, rff_features])
    x = keras.layers.Dense(256, activation='swish')(x)
    x = keras.layers.Dense(128, activation='swish')(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)

    return keras.Model(inputs=inputs, outputs=outputs)


print("\n--- Training High-Capacity RFF-ResNet Models ---", flush=True)

total_epochs = 350
lr_schedule = keras.optimizers.schedules.CosineDecay(initial_learning_rate=3e-3, decay_steps=total_epochs, alpha=1e-6)

m_single = create_deep_rff_resnet(num_rff=256, sigma=16.0)
m_single.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), loss='mse')
m_single.fit(x_train_norm, y_train_norm, epochs=total_epochs, batch_size=64, verbose=0)
score_custom_model(m_single, "Deep RFF-ResNet (Single)")


# Train 3-model RFF-ResNet Ensemble with varying RFF frequency scales
print("\nTraining 3x RFF-ResNet Ensemble...", flush=True)
models = []
sigmas = [12.0, 16.0, 20.0]
for idx, s in enumerate(sigmas):
    np.random.seed(idx * 50)
    m = create_deep_rff_resnet(num_rff=256, sigma=s)
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), loss='mse')
    m.fit(x_train_norm, y_train_norm, epochs=total_epochs, batch_size=64, verbose=0)
    models.append(m)

ens = EnsembleModel(models)
score_custom_model(ens, "Ensemble 3x RFF-ResNet")
