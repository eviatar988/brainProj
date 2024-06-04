# %%
import math
import os.path as op
import os
from mne.datasets import sample
from patients_matrix import PatientsMatrix
from coherence_matrix import CoherenceMatrix
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import levene, gaussian_kde
from scipy.stats import anderson
import scipy.stats as stats
import patients_matrix

# %%
rest_lists_path = 'rest_lists'
film_lists_path = 'film_lists'


#read data from dict and return it
def load_data(path, file):
    loaded_file = np.load(op.join(path, file))
    return loaded_file['matrix_arr']

#reshaping the film data to fit the rest data
def data_fit(rest_path, film_path, file):
    rest_data = load_data(rest_path, file)
    rest_shape = rest_data.shape[0]
    film_data = load_data(film_path, file)
    film_shape = film_data.shape[0]
    film_data = film_data[range(0, film_shape, 2)]
    return rest_data, film_data[:rest_shape]

#return max values for rest and film
def read_file_max(rest_path, film_path, file):
    rest_data,film_data = data_fit(rest_path, film_path, file)

    return np.sort(rest_data, axis=1)[:, -100:], np.sort(film_data, axis=1)[:, -100:]


"""def read_file_film_max(path, file):
     loaded_file = np.load(op.join(path,file))
     data = loaded_file['matrix_arr']

     return np.sort(data, axis=1)[range(0, data.shape[0], 2), -100:]"""


#return random values from rest and film
def read_file_random(rest_path, film_path, file):
    rest_data,film_data = data_fit(rest_path, film_path, file)
    num_random_columns = 100
    # Randomly select column indices
    random_column_indices = np.random.choice(rest_data.shape[1], num_random_columns, replace=False)
    # Extract random columns
    return rest_data[:, random_column_indices], film_data[:, random_column_indices]

"""def read_file_film_random(path, file):
    loaded_file = np.load(op.join(path, file))
    data = loaded_file['matrix_arr']
    num_random_columns = 100
    # Randomly select column indices
    random_column_indices = np.random.choice(data.shape[1], num_random_columns, replace=False)
    temp = data[:, random_column_indices]
    # Extract random columns
    return temp[range(0, data.shape[0], 2), :]"""

#return features from rest and film
def feature_extract(rest_path, film_path, file):
    rest_data, film_data = data_fit(rest_path, film_path, file)
    rest_features = np.zeros((rest_data.shape[0], 4))
    for i in range(rest_data.shape[0]):
        rest_features[i, :] = [np.mean(rest_data[i]), np.std(rest_data[i]), np.min(rest_data[i]), np.max(rest_data[i])]
    film_features = np.zeros((film_data.shape[0], 4))
    for i in range(film_data.shape[0]):
        film_features[i, :] = [np.mean(film_data[i]), np.std(film_data[i]), np.min(film_data[i]), np.max(film_data[i])]
    return rest_features


"""def feature_extract_film(path, file):
    loaded_file = np.load(op.join(path, file))
    data = loaded_file['matrix_arr']
    features = np.zeros((int(data.shape[0]/2), 4))
    for i in range(0, int(data.shape[0]/2)):
        features[i, :] = [np.mean(data[i*2]), np.std(data[i*2]), np.min(data[i*2]), np.max(data[i*2])]
    return features"""

# calculate the max indices of the mean of rest and film data, return the data in those indices


def max_indices(rest_path, film_path, file):
    rest_data, film_data = data_fit(rest_path, film_path, file)
    rest_mean = np.mean(rest_data, axis=0)
    rest_indices = np.argsort(rest_mean)[-20:]
    film_mean = np.mean(film_data, axis=0)
    film_indices = np.argsort(film_mean)[-20:]
    indices = np.append(rest_indices, film_indices)
    return rest_data[:, indices], film_data[:, indices]


def data_extract(freq_type_rest, freq_type_film, bounds, sec_per_sample ,extract_func):
    rest_path = 'rest_data'
    patients = os.listdir(rest_path)
    film_path = 'film_data'

    first_p = bounds[0]
    last_p = bounds[1]
    rest_dict_path = op.join(rest_path, patients[first_p])

    film_dict_path = op.join(film_path, patients[first_p])

    file_path = f'{patients[first_p]},freq={freq_type_rest},sec_per_sample={sec_per_sample}.npz'
    rest_data, film_data = extract_func(rest_dict_path,film_dict_path,file_path)

    for i in range(first_p+1,last_p+1):
        rest_dict_path = op.join(rest_path, patients[i])

        film_dict_path = op.join(film_path, patients[i])

        file_path = f'{patients[i]},freq={freq_type_rest},sec_per_sample={sec_per_sample}.npz'
        temp_rest, temp_film = rest_data, film_data = extract_func(rest_dict_path,film_dict_path,file_path)

        rest_data = np.append(rest_data, temp_rest, axis=0)

        film_data = np.append(film_data, temp_film, axis=0)
    return rest_data, film_data
    





