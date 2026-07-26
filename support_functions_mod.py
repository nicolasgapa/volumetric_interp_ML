"""
Volumetric interpolation using artificial neural networks.

Authors: Nicolas Gachancipa, Leslie Lamarche

Stanford Research Institute (SRI)
Embry-Riddle Aeronautical University
"""

# Imports.
try:
    import cartopy.crs as ccrs
except ImportError:
    ccrs = None
import datetime as dt
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymap3d as pm
from scipy.spatial import ConvexHull
try:
    import tables
except ImportError:
    tables = None
import h5py
try:
    import tensorflow as tf
except ImportError:
    tf = None
from sklearn.neural_network import MLPRegressor
import geopy.distance
from baseline_interp import fit_interpolator




def read_datafile(filenames, start_time, end_time, chi2lim=(0.1, 10)):
    """
    Read a processed AMISR hdf5 file and return the time, coordinates, values, and errors as arrays.

    Parameters:
        filename: [str]
            filename/path of processed AMISR hdf5 file
        start_time: [datetime]
            All the data before the start time is filtered out. Format: '%Y-%m-%dT%H:%M:%S'
        end_time: [datetime]
            All the data after the end time is filtered out. Format: '%Y-%m-%dT%H:%M:%S'
        chi2lim: [tuple]
            remove all points outside the given chi2 range (lower limit, upper limit)

    Returns:
        utime: [ndarray (nrecordsx2)]
            start and end time of each record (Unix Time)
        latitude: [ndarray (npoints)]
            geodetic latitude of each point
        longitude: [ndarray (npoints)]
            geodetic longitude of each point
        altitude: [ndarray (npoints)]
            geodetic altitude of each point
        value: [ndarray (nrecordsxnpoints)]
            parameter value of each data point
        error: [ndarray (nrecordsxnpoints)]
            error in parameter values
    """
    # Creating an empty dataframe to hold the data.
    df = pd.DataFrame()

    # Iterating through all of the input files.
    for filename in filenames:
        # Open the file, and extract the relevant data.
        data_curr = pd.DataFrame()
        if tables is not None:
            with tables.open_file(filename, 'r') as h5file:
                # Obtain time.
                utime = h5file.get_node('/Time/UnixTime')[:]

                # Obtain altitude, longitude, latitude, chi2, and fit code for the given times.
                altitude = h5file.get_node('/Geomag/Altitude')[:]
                latitude = h5file.get_node('/Geomag/Latitude')[:]
                longitude = h5file.get_node('/Geomag/Longitude')[:]

                # Filter time to only include times within the given start and end times.
                idx = np.argwhere((utime[:, 0] >= (start_time - dt.datetime.utcfromtimestamp(0)).total_seconds()) & (
                        utime[:, 1] <= (end_time - dt.datetime.utcfromtimestamp(0)).total_seconds())).flatten()
                if len(idx) == 0:
                    idx = [0]
                utime = utime[idx, :]
                utime = utime.mean(axis=1)

                # Obtain the density values, errors, chi2 values, and fitcode (only for the selected times).
                value = h5file.get_node('/FittedParams/Ne')[idx, :, :]
                error = h5file.get_node('/FittedParams/dNe')[idx, :, :]
                chi2 = h5file.get_node('/FittedParams/FitInfo/chi2')[idx, :, :]
                fc = h5file.get_node('/FittedParams/FitInfo/fitcode')[idx, :, :]
        else:
            with h5py.File(filename, 'r') as h5file:
                utime = h5file['/Time/UnixTime'][:]
                altitude = h5file['/Geomag/Altitude'][:]
                latitude = h5file['/Geomag/Latitude'][:]
                longitude = h5file['/Geomag/Longitude'][:]
                idx = np.argwhere((utime[:, 0] >= (start_time - dt.datetime.utcfromtimestamp(0)).total_seconds()) & (
                        utime[:, 1] <= (end_time - dt.datetime.utcfromtimestamp(0)).total_seconds())).flatten()
                if len(idx) == 0:
                    idx = [0]
                utime = utime[idx, :]
                utime = utime.mean(axis=1)
                value = h5file['/FittedParams/Ne'][idx, :, :]
                error = h5file['/FittedParams/dNe'][idx, :, :]
                chi2 = h5file['/FittedParams/FitInfo/chi2'][idx, :, :]
                fc = h5file['/FittedParams/FitInfo/fitcode'][idx, :, :]

        # Flatten the arrays.
        altitude = altitude.flatten()
        latitude = latitude.flatten()
        longitude = longitude.flatten()

        # Reshape arrays.
        value = value.reshape(value.shape[0], -1)
        error = error.reshape(error.shape[0], -1)
        chi2 = chi2.reshape(chi2.shape[0], -1)
        fit_code = fc.reshape(fc.shape[0], -1)

        # This accounts for an error in some of the hdf5 files where chi2 is overestimated by 369.
        if np.nanmedian(chi2) > 100.:
            chi2 = chi2 - 369.

        # data_check: 2D boolean array for removing "bad" data.
        # Each column corresponds to a different "check" condition.
        # TRUE for "GOOD" point; FALSE for "BAD" point.
        # A "good" record that shouldn't be removed should be TRUE for EVERY check condition.
        data_check = np.array([chi2 > chi2lim[0], chi2 < chi2lim[1], np.isin(fit_code, [1, 2, 3, 4])])

        # If ANY elements of data_check are FALSE, flag index as bad data. Remove the bad data from the array.
        bad_data = np.squeeze(np.any(data_check is False, axis=0, keepdims=True))
        value[bad_data] = np.nan
        error[bad_data] = np.nan

        # Remove the points where coordinate arrays are NaN.
        value = value[:, np.isfinite(altitude)]
        error = error[:, np.isfinite(altitude)]
        latitude = latitude[np.isfinite(altitude)]
        longitude = longitude[np.isfinite(altitude)]
        altitude = altitude[np.isfinite(altitude)]

        # Convert coordinates to hull_vert.
        x, y, z = pm.geodetic2ecef(latitude, longitude, altitude)
        r_cart = np.array([x, y, z]).T
        c_hull = ConvexHull(r_cart)
        hull_vert = r_cart[c_hull.vertices]

        # Run NN.
        max_n = 1512 # Maximum number of points wou want to use (useful for testing).
        data_curr = pd.DataFrame([latitude, longitude, altitude, value[0], error[0]]).T.iloc[:max_n, :]
        data_curr.columns = ['Latitude', 'Longitude', 'Altitude', 'Value', 'Error']  
        df = pd.concat([df, data_curr])
    # Return.
    return df


class StopAtLossValue(tf.keras.callbacks.Callback):
    """
    Callback to stop training a neural network when a certain loss is achieved.
    Change the set_point_loss value to define the desired loss value.
    """

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        set_point_loss = 1e-6
        if logs.get('loss') <= set_point_loss:
            self.model.stop_training = True


def fit_volumetric_models(df, real_dist=False, density_range=(1e10, 1e12), model='ann', n_trials=15):
    """
    Preprocess radar data the same way `volumetric_nn` does (drop nans, filter to
    density_range, log10-transform, 0-1 normalize coordinates), then fit either an ensemble
    of ANNs or a single non-ML baseline interpolator (see baseline_interp.py). Splitting this
    out of `volumetric_nn` lets any prediction point - not just the fixed 3D grid
    `volumetric_nn` plots - be scored against a fitted model with `ensemble_predict` (used
    for the held-out train/test comparison in holdout_interp_compare.py).

    Parameters:
        df: [dataframe] same as `volumetric_nn`'s `df` (columns 'Latitude', 'Longitude',
            'Altitude', 'Value', 'Error').
        real_dist, density_range: same meaning as in `volumetric_nn`.
        model: [str] 'ann' (default) fits an ensemble of n_trials ANNs. Any other value
            ('linear', 'nearest', 'rbf') fits the corresponding baseline_interp.py method
            once (these are deterministic).
        n_trials: [int] number of ANNs to train and ensemble-average (ignored for baselines).

    Returns:
        models: [list] one or more fitted models, each exposing `.predict(x)` on normalized
            (0-1) coordinates, returning normalized (0-1) log-density predictions.
        df: [dataframe] the processed dataframe (post dropna/filter/log-transform), with a
            'Log Value' column and without 'Error'.
        x: [dataframe] normalized (0-1) Latitude/Longitude/Altitude for every row of df.
        weights: [ndarray] per-point weight used for the weighted error metrics.
        x_lim: [list] [[min, max], ...] per coordinate column (Latitude, Longitude,
            Altitude), in real (unnormalized) units. Use these to normalize any new query
            point the same way the training coordinates were normalized.
        y_lim: [list] [min, max] of Log Value, in real (unnormalized) units. Use these to
            convert a normalized prediction back to a real electron density value:
            10 ** (prediction * (y_lim[1] - y_lim[0]) + y_lim[0]).
    """
    #################
    # DATA PROCESSING
    #################

    # Drop nans and sort dataframe by energy value.
    df = df.dropna()
    df = df.sort_values(by='Value')

    # Convert latitudes/longitudes to real distances in km (if real_dist == True).
    if real_dist:

        # Define the location of RISR.
        risr_coords = (74.72, -94.9)

        # Compute the real distances (in km) between every measurement's latitude and the radar's latitude.
        # Do the same for longitude.
        new_lats, new_lons = [], []
        for lat, lon in zip(df['Latitude'], df['Longitude']):
            lat_coord = (lat, risr_coords[1])
            lon_coord = (risr_coords[0], lon)
            lat_dist = geopy.distance.geodesic(risr_coords, lat_coord).km
            lon_dist = geopy.distance.geodesic(risr_coords, lon_coord).km
            if lat < risr_coords[0]:
                lat_dist *= -1
            if lon < risr_coords[1]:
                lon_dist *= -1
            new_lats.append(lat_dist)
            new_lons.append(lon_dist)

        # Replace in the dataframe. From now on, the 'Latitude' and 'Longitude' columns contain
        df['Latitude'] = new_lats
        df['Longitude'] = new_lons

    # Filter out electron densities outside the given range.
    df = df[df['Value'] >= density_range[0]]
    df = df[df['Value'] <= density_range[1]]

    # Find the log10 of the electron density values and errors. This conversion allows the neural network to be trained more
    # efficiently. Save the converted values to a new column.
    df['Log Value'] = np.log10(df['Value'])
    df['Log Error'] = np.log10(df['Error'])

    # Drop the data instances that have nan values in any column. Sometimes the log conversion leads to nan errors.
    df = df.dropna()

    # Defining the weight of each point.
    y_actual = list(df['Log Value'])
    y_error = list(df['Log Error'])

    weights = []
    for i in range(len(y_actual)):
        # Finding how each error relates to the actual value.
        weight = y_actual[i] / y_error[i]**2
        weights.append(weight)
    weights = np.array(weights)

    # Dropping the Error column.
    df = df.drop(['Error'], axis=1)

    # Normalize the data (in a new dataframe). Normalized data allows the neural network to be trained faster and more
    # effectively. Normalization is a rescaling of the data from the original range so that all values are within the
    # range of 0 and 1.
    # https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/
    df_train = df.copy()
    for column in df_train:
        df_train[column] = (df_train[column] - np.min(df_train[column])) / (np.max(df_train[column]) - np.min(df_train[column]))

    ##########
    # TRAINING
    ##########

    # Define neural network inputs (x and y).
    # x: Normalized logarithmic density values (response variable).
    # y: Normalized latitude, longitude, and altitude (predictor variables).
    y = df_train['Log Value']
    x = df_train.drop(['Value', 'Log Value', 'Log Error'], axis=1)

    # Get the minimum and maximum values of each column (lat, lon, alt, logdensities). Save the min/max values to
    # x_lim (lat, lot, alt) and y_lim (densities in log form).
    y_org = pd.DataFrame(df['Log Value'])
    x_org = df.drop(['Value', 'Log Value'], axis=1)
    x_lim = [[np.min(x_org[c]), np.max(x_org[c])] for c in x_org]
    y_lim = [np.min(y_org.iloc[:, 0]), np.max(y_org.iloc[:, 0])]

    models = []
    if model.lower() == 'ann':
        # Training the nn n_trials times (can be changed) and keeping every trained network,
        # so predictions elsewhere can be ensemble-averaged exactly like `volumetric_nn` does.
        for i in range(n_trials):
            ####################
            ## NEURAL NETWORK ##
            ####################
            if tf is not None:
                network = tf.keras.Sequential([tf.keras.layers.Dense(units=512, input_shape=[x.shape[1]], activation='tanh'),
                                           tf.keras.layers.Dense(units=256, activation='tanh'),
                                           tf.keras.layers.Dense(units=128, activation='tanh'),
                                           tf.keras.layers.Dense(units=64, activation='tanh'),
                                           tf.keras.layers.Dense(units=32, activation='tanh'),
                                           tf.keras.layers.Dense(units=16, activation='tanh'),
                                           tf.keras.layers.Dense(units=8, activation='tanh'),
                                           tf.keras.layers.Dense(units=4, activation='tanh'),
                                           tf.keras.layers.Dense(units=1, activation='sigmoid')])

                # Compile the network: Adam optimizer with learning rate 1e-3 and MeanSquaredError loss.
                network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError())

                # Define a checkpoint. This allows the network to save the "best weights" to a keras file.
                checkpoint = tf.keras.callbacks.ModelCheckpoint('weights.keras', verbose=1, monitor='loss', save_best_only=True, mode='auto')

                # Early stopping: patience increased to 50 epochs.
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='auto', verbose=0, patience=50)

                # Train the network for up to 600 epochs.
                network.fit(x, y, epochs=600, sample_weight=weights, callbacks=[StopAtLossValue(), early_stopping, checkpoint])

                # Load the best weights that have been saved in the h5 file.
                network.load_weights('weights.keras')
            else:
                mlp = MLPRegressor(hidden_layer_sizes=(512, 256, 128, 64, 32, 16, 8, 4), activation='tanh', max_iter=600, random_state=i)
                mlp.fit(x.values, y.values)
                network = mlp

            models.append(network)
    else:
        ##########################
        ## BASELINE INTERPOLATOR ##
        ##########################
        # Non-ML baseline ('linear', 'nearest', or 'rbf'). These are deterministic, so a single
        # fit replaces the n_trials-trial averaging used for the (stochastic) ANN above.
        models.append(fit_interpolator(x.values, y.values, model))

    return models, df, x, weights, x_lim, y_lim


def ensemble_predict(models, x):
    """
    Average `.predict(x)` across an ensemble of fitted models (see `fit_volumetric_models`).
    For a single-model ensemble (e.g. a baseline interpolator), this is just that model's
    prediction.
    """
    preds = [np.asarray(m.predict(x)) for m in models]
    return np.mean(preds, axis=0)


class EnsembleModel:
    """
    Wraps a list of fitted models (see `fit_volumetric_models`) so `.predict()` returns
    their ensemble-averaged prediction - the same quantity `volumetric_nn` plots and scores
    - rather than an arbitrary single member. Lets callers elsewhere (e.g.
    `graph_support_functions.create_all_data`) do `network.predict(x)` and get the ensemble
    average.
    """

    def __init__(self, models):
        self.models = models

    def predict(self, x):
        return ensemble_predict(self.models, x)


def volumetric_nn(df, start=None, end=None, resolution=(10, 10, 10), cbar_lim=None, real_dist=False,
                  density_range=(1e10, 1e12), fig3D=True, save_imgs=False, model='ann'):
    """
    Parameters:
        df: [dataframe]
            Pandas daraframe with 4 columns (not including the index column). Such columns must be: 'Latitude',
            'Longitude', 'Altitude' and 'Value'. The 'Value' column must contain the electron density measures.
        resolution: [tuple or list]
            Tuple or list containing three integers. The resolution is the size of the 3D grid (x, y, z), corresponding
            to (longitude, latitude, altitude). The higher the resolution, the longer it takes for the code to run.
        cbar_lim: [tuple]
            Elctron density color bar limits (e.g. (1e10, 1e11)).
        real_dist: [boolean]
            Set to True if you want the output to show the distances in kilometers from the radar (rather than in
            longitude and latitude degrees).
        density_range: [tuple]
            Tuple containing the lower and upper limits of allowed electron density values. Any radar measurement
            outside this range is removed before training the neural network. Default: (10e10, 10e12).
        fig3D [boolean]:
            True for 3D plot, False for 2D plot only.
        save_imgs:
            True if you want to save the images in a png format to the local directory.
        model: [str]
            Interpolation method used to fill the 3D grid. 'ann' (default) trains the neural network as before.
            Any other value ('linear', 'nearest', 'rbf') fits the corresponding non-ML baseline interpolator
            from baseline_interp.py instead, so its error can be compared against the ANN's using this same
            function (see graph_support_functions.create_all_data).
    Returns:
        2D or 3D plot.
    """

    # Preprocess the data and fit the model (an ensemble of ANNs, or a single non-ML
    # baseline interpolator - see fit_volumetric_models for details).
    models, df, x, weights, x_lim, y_lim = fit_volumetric_models(df, real_dist=real_dist,
                                                                  density_range=density_range, model=model)

    # Compute the lat/lon aspect ratio (useful for plotting).
    aspect_ratio = abs(x_lim[1][0] - x_lim[1][1]) / abs(x_lim[0][0] - x_lim[0][1])

    # Extract the resolution of the grid (x, y, z) = (lon, lat, alt). Given by the user.
    rows, columns, heights = resolution

    # Predict a value for each element in the 3D grid (every lon, lat, alt combination). Save the results to a variable
    # called y_preditct.
    predict = np.array([[r / (rows - 1), c / (columns - 1), h / (heights - 1)] for r in range(rows) for c in range(columns) for
                h in range(heights)])

    # y_predict is used for plotting. Contains the ensemble-averaged prediction for every grid point.
    y_predict = ensemble_predict(models, predict)

    # y_preds_avg is used for calculating performance metrics. Contains the ensemble-averaged
    # prediction for every point in the training dataframe.
    y_preds_avg = ensemble_predict(models, x.values)

    # Wrap the ensemble so callers elsewhere (e.g. graph_support_functions.create_all_data)
    # get this same averaged prediction via `network.predict(...)`, rather than an arbitrary
    # single ensemble member.
    network = EnsembleModel(models)

    # Calculating the performance metrics.
    df_score = df.copy()
    nn_predicted = []

    for i in range(len(x)):
        # Convert each predicted log density value to absolute scale.
        nn_predicted.append(10 ** (y_preds_avg[i] * (y_lim[1] - y_lim[0]) + y_lim[0]))
    
    df_score['Predicted'] = np.array(nn_predicted)
    df_score['Actual'] = df['Value']
    
    # Calculating performance metrics.
    weighted_mean_abs_error_sklearn = np.average(abs(df_score['Actual']-df_score['Predicted']), axis=0, weights=weights) 
    weighted_root_mean_sq_error_sklearn = (np.average(abs((df_score['Actual']-df_score['Predicted'])**2), axis=0, weights=weights))**0.5
    weighted_mean_sq_error_sklearn = np.average(abs((df_score['Actual']-df_score['Predicted'])**2), axis=0, weights=weights)

    # ##########
    # # PLOTTING
    # ##########

    # Create a 3D array with the results (ressembling a 3D desity map). Convert the results (which are in a log scale)
    # to real density values.
    # Add the original data points to their closest layer: Add a new column to the original dataframe that contains
    # the corresponding layer index of each data point.
    stations = list(np.linspace(x_lim[2][0], x_lim[2][1], heights))
    indices = []

    def find_closest_station(a, stations):
        closest_station = stations[0]
        smallest_diff = abs(stations[0] - a)
        for station in stations:
            diff = abs(station - a)
            if diff < smallest_diff:
                closest_station = station
                smallest_diff = diff
        return closest_station

    for a in list(df['Altitude']):
        closest_altitude = find_closest_station(a, stations)
        indices.append(stations.index(closest_altitude))
    
    df['Layer'] = indices


    # Create a copy of the original data frame to filter out "out-of-bounds" predictions.
    # Normalize the latitude and longitude data.
    # Dropping every column besides longitude, latitude, and layer. Then sorting the data frame
    # by layer.
    df_new = df.copy()
    for column in ['Latitude', 'Longitude']:
        df_new[column] = (df_new[column] - np.min(df_new[column])) / (np.max(df_new[column]) - np.min(df_new[column]))
    df_new = df_new.drop(['Altitude', 'Value', 'Log Value', 'Log Error'], axis=1)
    df_new = df_new.sort_values(by=['Layer'])

    # Finding the latitude and longitude boundaries for each layer.
    lat_min_max = []
    long_min_max = []

    # Getting the latitude limits for each layer.
    for i in range(heights):
        min = np.min(df_new['Latitude'][df_new['Layer'] == i])
        max = np.max(df_new['Latitude'][df_new['Layer'] == i])
        if i > 0:
            last_min = np.min(df_new['Latitude'][df_new['Layer'].isin(range(i))])
            last_max = np.max(df_new['Latitude'][df_new['Layer'].isin(range(i))])
            if last_min < min:
                min = last_min
            if last_max > max:
                max = last_max
        lat_min_max.append([min, max])

    # Getting the longitude limits for each layer.
    for i in range(heights): 
        min = np.min(df_new['Longitude'][df_new['Layer'] == i])
        max = np.max(df_new['Longitude'][df_new['Layer'] == i])
        if i > 0:
            last_min = np.min(df_new['Longitude'][df_new['Layer'].isin(range(i))])
            last_max = np.max(df_new['Longitude'][df_new['Layer'].isin(range(i))])
            if last_min < min:
                min = last_min
            if last_max > max:
                max = last_max
        long_min_max.append([min, max])

    # Create a 3D array with the results (ressembling a 3D desity map). 
    array = np.zeros((rows, columns, heights))
    for e, i in enumerate(predict):
            # Obtain the x, y, z (lon, lat, alt) location.
            row = int(round(i[0] * (rows - 1), 0)) #i[0] is lon
            column = int(round(i[1] * (columns - 1), 0)) #i[1] is lat
            level = int(round(i[2] * (heights - 1), 0)) # level is layer

            if (i[0] >= lat_min_max[level][0] and i[0] <= lat_min_max[level][1]) and (i[1] >= long_min_max[level][0] and i[1] <= long_min_max[level][1]):
                # Convert predicted log density value to absolute scale, and save the value to the 3D array.
                array[row, column, level] = (10 ** (y_predict[e][0] * (y_lim[1] - y_lim[0]) + y_lim[0]))
            else:
                array[row, column, level] = np.nan
    
        
    # Define the color bar limits (if not provided by the user).
    if cbar_lim is None:
        max_val = np.max(df['Value'])
        min_val = np.min(df['Value'])
        cbar_lim = [min_val, max_val]

    # Create a plot, and corresponding subplots.
    fig = plt.figure()
    if fig3D:
        ax = fig.add_subplot(121)
        ax_slider = plt.axes([0.15, 0.1, 0.25, 0.03])
    else:
        if real_dist:
            ax = fig.add_subplot()
        else:
            ax = fig.add_subplot(projection=ccrs.PlateCarree())
        ax_slider = plt.axes([0.3, 0.1, 0.42, 0.03])
    plt.subplots_adjust(bottom=0.25)

    # Extract the first layer (lowest altitude).
    layer = array[:, :, 0]

    # Create grid plot and slider. Plot the first layer (lower altitude).
    im = ax.imshow(layer, cmap='jet', origin='lower', extent=[x_lim[1][0], x_lim[1][1], x_lim[0][0], x_lim[0][1]],
                   vmin=cbar_lim[0], vmax=cbar_lim[1], aspect=aspect_ratio)
    slider = Slider(ax_slider, 'Altitude (m)', x_lim[2][0], x_lim[2][1], x_lim[2][0],
                    valstep=((x_lim[2][1] - x_lim[2][0]) / (heights - 1)))

    # Set x and y axis labels.
    if real_dist:
        ax.set_xlabel('Km from radar - East (+)')
        ax.set_ylabel('Km from radar - North (+)')
    else:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    

    # Plot colorbar, using the predefined color bar limits.
    fig.colorbar(im, ax=ax, location='right', orientation='vertical', )
    cmap = im.set_clim(cbar_lim[0], cbar_lim[1])

    # Plot original data points of the first layer (layer = 0).
    points = df[df['Layer'] == 0]
    ax.scatter(list(points['Longitude']), list(points['Latitude']), color='white', s=40)
    ax.scatter(list(points['Longitude']), list(points['Latitude']), cmap=cmap, s=20)

    # Create function to be called when the slider value (altitude) is changed.
    def update(alt_value, save=False):
        val = (alt_value - x_lim[2][0]) / (x_lim[2][1] - x_lim[2][0])
        ax.clear()
        pts = df[df['Layer'] == stations.index(alt_value)]
        ax.imshow(array[:, :, int(round(val * (heights - 1), 0))], cmap='jet', origin='lower',
                  extent=[x_lim[1][0], x_lim[1][1], x_lim[0][0], x_lim[0][1]], vmin=cbar_lim[0], vmax=cbar_lim[1],
                  aspect=aspect_ratio)
        if real_dist:
            ax.set_xlabel('Km from radar - East (+)')
            ax.set_ylabel('Km from radar - North (+)')
        else:
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        ax.scatter(list(pts['Longitude']), list(pts['Latitude']), color='white', s=40)
        ax.scatter(list(pts['Longitude']), list(pts['Latitude']), c=list(pts['Value']), cmap='jet', s=20,
                   vmin=cbar_lim[0], vmax=cbar_lim[1])

        # Save the layer to a png file.
        if save:
            ax.set_title('Altitude: {} km.'.format(round(alt_value / 1000, 1)))
            plt.savefig('fig_{}.jpg'.format(int(alt_value / 1000)))

    # Save the images in png files (if selected by the user).
    if save_imgs:
        for value in range(int(x_lim[2][0]), int(x_lim[2][1]), int((x_lim[2][1] - x_lim[2][0]) / (heights - 1))):
            update(value, save=True)

    # Call update function when slider value is changed.
    slider.on_changed(update)

    # Plot 3D figure.
    if fig3D:

        # Create new meshgrid. (xx = lon, yy = lat).
        xx, yy = np.meshgrid(np.linspace(x_lim[1][0], x_lim[1][1], rows),
                             np.linspace(x_lim[0][0], x_lim[0][1], columns))
        axis = fig.add_subplot(122, projection='3d')

        # Set z limits.
        axis.set_zlim(x_lim[2][0] / 1000, x_lim[2][1] / 1000)

        # 3D Plot.
        number_of_layers = heights
        if heights < number_of_layers:
            number_of_layers = heights
        for v in range(0, array.shape[-1], int(array.shape[-1] / number_of_layers)):
            print('Plotting 3D layers: {} out of {}.'.format(v + 1, number_of_layers))
            layer = array[:, :, v]
            axis.contourf(xx, yy, layer, 100, zdir='z',
                          offset=((v / number_of_layers) * (x_lim[2][1] - x_lim[2][0]) + x_lim[2][0]) / 1000,
                          vmin=cbar_lim[0], vmax=cbar_lim[1], cmap='jet')
        # Set labels.
        if real_dist:
            plt.xlabel('Km from radar - North (+)')
            plt.ylabel('Km from radar - East (+)')
        else:
            plt.xlabel('Latitude (°)')
            plt.ylabel('Longitude (°)')
        axis.set_zlabel('Altitude (km)')
        if start is not None and end is not None:
            plt.suptitle(f'Date: {start:%Y-%m-%d}  Timeframe (UTC): {start:%H:%M:%S} - {end:%H:%M:%S}')

        # Metrics.
        print("Weighted Mean Average Error (WMAE):", weighted_mean_abs_error_sklearn)
        print("Weighted Mean Squared Error (WMSE):", weighted_mean_sq_error_sklearn)
        print("Weighted Root Mean Squared Error (WRMSE):", weighted_root_mean_sq_error_sklearn)
        
        plt.show()
        
    # Return.
    # Used in the process to graph the difference in synthetic data and predicted data.
    return network, df, y_lim, y_predict, array, stations

