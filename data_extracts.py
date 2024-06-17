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
import cv2

# %%
rest_lists_path = 'rest_lists'
film_lists_path = 'film_lists'


# read data from dict and return it
def load_data(path, file):
    loaded_file = np.load(op.join(path, file))
    return loaded_file['matrix_arr']


# reshaping the film data to fit the rest data
def data_fit(rest_path, film_path, rest_file, film_file):
    rest_data = load_data(rest_path, rest_file)
    rest_shape = rest_data.shape[0]
    film_data = load_data(film_path, film_file)
    film_shape = film_data.shape[0]
    film_data = film_data[range(0, film_shape, 2)]
    return rest_data, film_data[:rest_shape]


# return max values for rest and film
def read_file_max(rest_path, film_path, rest_file, film_file):
    rest_data, film_data = data_fit(rest_path, film_path, rest_file, film_file)

    return np.sort(rest_data, axis=1)[:, -100:], np.sort(film_data, axis=1)[:, -100:]


# return random values from rest and film
def read_file_random(rest_path, film_path, rest_file, film_file):
    rest_data, film_data = data_fit(rest_path, film_path, rest_file, film_file)
    num_random_columns = 100
    # Randomly select column indices
    random_column_indices = np.random.choice(rest_data.shape[1], num_random_columns, replace=False)
    # Extract random columns
    return rest_data[:, random_column_indices], film_data[:, random_column_indices]


# return features from rest and film
def feature_extract(rest_path, film_path, rest_file, film_file):
    rest_data, film_data = data_fit(rest_path, film_path, rest_file, film_file)
    rest_features = np.zeros((rest_data.shape[0], 4))
    for i in range(rest_data.shape[0]):
        rest_features[i, :] = [np.mean(rest_data[i]), np.std(rest_data[i]), np.min(rest_data[i]), np.max(rest_data[i])]
    film_features = np.zeros((film_data.shape[0], 4))
    for i in range(film_data.shape[0]):
        film_features[i, :] = [np.mean(film_data[i]), np.std(film_data[i]), np.min(film_data[i]), np.max(film_data[i])]
    return rest_features


# returning the indices with the most difference between film and rest
def max_diffrence_indices(rest_path, film_path, rest_file, film_file):
    rest_data, film_data = data_fit(rest_path, film_path, rest_file, film_file)
    rest_mean = np.mean(rest_data, axis=0)
    film_mean = np.mean(film_data, axis=0)
    indices = np.argsort(film_mean - rest_mean)[-100:]
    return rest_data[:, indices], film_data[:, indices]


# calculate the max indices of the mean of rest and film data, return the data in those indices
def max_indices(rest_path, film_path, rest_file, film_file):
    rest_data, film_data = data_fit(rest_path, film_path, rest_file, film_file)
    rest_mean = np.mean(rest_data, axis=0)
    # rest_indices = np.argsort(rest_mean)[-20:]
    film_mean = np.mean(film_data, axis=0)
    film_indices = np.argsort(film_mean)[-100:]
    # indices = np.append(rest_indices, film_indices)
    indices = film_indices
    return rest_data[:, indices], film_data[:, indices]


def matrix_transform(matrix_flat):
    data_len = matrix_flat.shape[0]
    matrix_len = int(math.sqrt(2 * matrix_flat.shape[1] + 1 / 4) + 1 / 2)
    matrix = np.zeros(shape=(data_len, matrix_len, matrix_len), dtype=float)
    for sec in range(data_len):
        index = 0
        for i in range(matrix_len):
            for j in range(i, matrix_len):
                if i == j:
                    matrix[sec, i, j] = 1
                else:
                    matrix[sec, i, j] = matrix_flat[sec, index]
                    matrix[sec, j, i] = matrix[sec, i, j]
                    index += 1
    return matrix


def matrix_resize(matrix):
    resized_matrix = np.zeros(shape=(matrix.shape[0], 30, 30))
    for i in range(matrix.shape[0]):
        resized_matrix[i, :, :] = cv2.resize(matrix[i, :, :], (30, 30), interpolation=cv2.INTER_AREA)
    return resized_matrix


def matrix_fit(rest_path, film_path, rest_file, film_file):
    rest_data, film_data = data_fit(rest_path, film_path, rest_file, film_file)
    rest_data = matrix_transform(rest_data)
    rest_data = matrix_resize(rest_data)
    rest_data = np.reshape(rest_data, (rest_data.shape[0], rest_data.shape[1], rest_data.shape[2], 1))

    film_data = matrix_transform(film_data)
    film_data = matrix_resize(film_data)
    film_data = np.reshape(film_data, (film_data.shape[0], film_data.shape[1], film_data.shape[2], 1))
    return rest_data, film_data


def data_extract(freq_type_rest, freq_type_film, bounds, extract_func):
    rest_path = 'rest_data'
    patients = os.listdir(rest_path)
    film_path = 'film_data'

    first_p = bounds[0]
    last_p = bounds[1]

    rest_dict = op.join(rest_path, patients[first_p])
    film_dict = op.join(film_path, patients[first_p])

    rest_file = f'{patients[first_p]},task=rest,freq={freq_type_rest}.npz'
    film_file = f'{patients[first_p]},task=film,freq={freq_type_film}.npz'

    rest_data, film_data = extract_func(rest_dict, film_dict, rest_file, film_file)

    for i in range(first_p + 1, last_p + 1):
        rest_dict = op.join(rest_path, patients[i])
        film_dict = op.join(film_path, patients[i])

        rest_file = f'{patients[i]},task=rest,freq={freq_type_rest}.npz'
        film_file = f'{patients[i]},task=film,freq={freq_type_film}.npz'
        temp_rest, temp_film = extract_func(rest_dict, film_dict, rest_file, film_file)

        rest_data = np.append(rest_data, temp_rest, axis=0)
        film_data = np.append(film_data, temp_film, axis=0)

    return rest_data, film_data
