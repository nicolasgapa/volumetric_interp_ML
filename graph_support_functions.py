# Imports.
import numpy as np
import pandas as  pd
import datetime as dt
from support_functions_mod import read_datafile
from support_functions_mod import volumetric_nn
from amisrsynthdata.ionosphere import Ionosphere
import yaml


def create_all_data(start, end, data_file, n, config_file, method='ann'):
    # method: 'ann' (default) evaluates the neural network. Any other value ('linear',
    # 'nearest', 'rbf') evaluates the corresponding non-ML baseline interpolator instead,
    # using this exact same function, so the ANN and the baselines are compared with
    # identical evaluation code (see compare_interp_methods.py).

    # Open file and extract the following arrays: time, latitude, longitude, altitude, value, error.
    data = read_datafile(data_file, start, end)

    # Configuring the synthetic ionosphere file.
    with open(config_file, 'r') as cf:
        config = yaml.load(cf, Loader=yaml.FullLoader)
    iono = Ionosphere(config)

    # Importing the Network (or fitting the baseline interpolator, if method != 'ann').
    network, _, y_lim, _, _, stations = volumetric_nn(data, resolution=(100, 100, 30), cbar_lim=(1e10, 3e11), real_dist=False, fig3D=True, model=method)

    # Finding the min and max of the latitude, longitude, and altitude.
    min_lat = int(np.min(data['Latitude']))
    max_lat = int(np.max(data['Latitude']))

    min_long = int(np.min(data['Longitude']))
    max_long = int(np.max(data['Longitude']))

    min_alt = int(np.min(data['Altitude']))
    max_alt = int(np.max(data['Altitude']))

    # Setting Utime to a numpy array.
    utime = np.array((start-dt.datetime.utcfromtimestamp(0)).total_seconds())
   
    # Create an empty dataframe to store the latitude, longitude, and altitude combinations.
    df = pd.DataFrame()
    lats = []
    longs = []
    alts = []

    # List to store the actual densities.
    actual_densities = []

    # Finding the actual density at every combination of the latitudes, longitudes, and altitudes.
    for lat in range(min_lat*4, max_lat*4, 1):
        for long in range(min_long*8, max_long*8, 1):
            for alt in range(min_alt, max_alt, 10_000):
                actual_densities.append(iono.density(utime, lat/4, long/8, alt))
                lats.append(lat/4)
                longs.append(long/8)
                alts.append(alt)

    # Adding the columns to the empty dataframe.
    df['Latitude'] = lats
    df['Longitude'] = longs
    df['Altitude'] = alts
    df['actual_densities'] = actual_densities

    # Filtering out negative densities.
    df = df[df['actual_densities'] >= 0]

    # Normalizing the longitude, latitude, and altitude.
    # Finding the predicted electron density from the network.
    # Convert each predicted log density value to absolute scale.
    df_x = df[['Latitude', 'Longitude', 'Altitude']]
    x = df_x.copy()
    for column in x:
        x[column] = (df_x[column] - np.min(df_x[column])) / (np.max(df_x[column]) - np.min(df_x[column]))
    network_predicted_densities = network.predict(x)
    network_predicted = 10 ** (network_predicted_densities * (y_lim[1] - y_lim[0]) + y_lim[0])


    # Adding predictions to the DataFrame
    df['network_predicted'] = network_predicted

    # Converting None values to NaN.
    df['network_predicted'] = df['network_predicted'].apply(lambda x: np.nan if x is None else x).astype(float)
    df['actual_densities'] = df['actual_densities'].apply(lambda x: np.nan if x is None else x).astype(float)

    # Compute the difference
    df['Value'] = df['network_predicted'] - df['actual_densities']

    # Return.
    return df, stations

def index_data(datafiles, stations):
    # Add the original data points to their closest layer: Add a new column to the original dataframe
    # that contains the corresponding layer index of each data point.
    for data_frame in datafiles:
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

        for a in list(data_frame['Altitude']):
            closest_altitude = find_closest_station(a, stations)
            indices.append(stations.index(closest_altitude))
        data_frame['Layer'] = indices
    return datafiles

def filter_data(df, ref_data, height):
    # Finding the maximum and minimum values for latitude and longitude in each layer.
    lat_min_max = []
    long_min_max = []

    # Latitude.
    for i in range(height): 
        min = np.min(ref_data['Latitude'][ref_data['Layer'] == i])
        max = np.max(ref_data['Latitude'][ref_data['Layer'] == i])
        if i > 0:
            last_min = np.min(ref_data['Latitude'][ref_data['Layer'].isin(range(i))])
            last_max = np.max(ref_data['Latitude'][ref_data['Layer'].isin(range(i))])
            if last_min < min:
                min = last_min
            if last_max > max:
                max = last_max
        lat_min_max.append([min, max])

    # Longitude.
    for i in range(height): 
        min = np.min(ref_data['Longitude'][ref_data['Layer'] == i])
        max = np.max(ref_data['Longitude'][ref_data['Layer'] == i])
        if i > 0:
            last_min = np.min(ref_data['Longitude'][ref_data['Layer'].isin(range(i))])
            last_max = np.max(ref_data['Longitude'][ref_data['Layer'].isin(range(i))])
            if last_min < min:
                min = last_min
            if last_max > max:
                max = last_max
        long_min_max.append([min, max])
    
    # Filtering the layers so that data outside of the region where data was collected ("the cone")
    # is removed from the data frame.
    df_filtered = pd.DataFrame(columns=['Latitude', 'Longitude', 'Altitude', 'Value', 'Layer'])
    for index, row in df.iterrows():
        lat_min = lat_min_max[int(row['Layer'])][0]
        lat_max = lat_min_max[int(row['Layer'])][1]
        long_min = long_min_max[int(row['Layer'])][0]
        long_max = long_min_max[int(row['Layer'])][1]
        if (lat_min <= row['Latitude'] and lat_max >= row['Latitude']) and (long_min <= row['Longitude'] and long_max >= row['Longitude']):
            new_row = {'Latitude':row['Latitude'], 'Longitude':row['Longitude'], 'Altitude':row['Altitude'], 'Value':row['Value'], 'Layer':row['Layer']}
            df_filtered.loc[len(df_filtered)] = new_row

    # Return.
    return df_filtered
