# %%
import math
import os.path as op
import os
from mne.datasets import sample
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from patients_matrix import PatientsMatrix
import coherence_matrix
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



def load_data(path, file):
    
    '''
    Load the data from the file
    return rest_data, film_data arrys for single patient
    each array is a 2D array with shape (time, flat_matrix)
    '''
    
    loaded_file = np.load(op.join(path, file))
    return loaded_file['matrix_arr']


# 
def data_fit(rest_path, film_path, rest_file, film_file):
    '''
    reshaping the film data to fit the rest data.
    taking every second row matrix from the film data
    cutting the film data to the same size as the rest data
    '''
    rest_data = load_data(rest_path, rest_file)
    rest_shape = rest_data.shape[0]
    film_data = load_data(film_path, film_file)
    film_shape = film_data.shape[0]
    film_data = film_data[range(0, film_shape, 2)]
    return rest_data, film_data[:rest_shape]

def data_normalize(rest_data, film_data):
    rest_data =  np.apply_along_axis(lambda x: (x - np.mean(x)) / np.std(x), axis=1, arr= rest_data)
    film_data = np.apply_along_axis(lambda x: (x - np.mean(x)) / np.std(x), axis=1, arr= film_data_data)

def max_values(rest_path, film_path, rest_file, film_file):
    '''
    return the 100 max values for each second in the rest and film data
    '''
    rest_data, film_data = data_fit(rest_path, film_path, rest_file, film_file)
    rest_indices = np.sort(np.argsort(rest_data, axis=1)[:, -50:], axis=1)
    film_indices = np.sort(np.argsort(film_data, axis=1)[:, -50:], axis=1)
    rest_temp = np.zeros((rest_data.shape[0], 50))
    film_temp = np.zeros((rest_data.shape[0], 50))
    for i in range(rest_data.shape[0]):
        rest_temp[i, :] = rest_data[i, rest_indices[i]]
        film_temp[i, :] = film_data[i, film_indices[i]]
    return rest_temp, film_temp

def max_indices_mean(rest_path, film_path, rest_file, film_file):
    '''
    create a mean matrix from rest and film data for single patient
    choose the 100 max indices from the mean matrix
    return arrys with the values in those indices for each second
    '''
    rest_data, film_data = data_fit(rest_path, film_path, rest_file, film_file)
    rest_mean = np.mean(rest_data, axis=0) 
    film_mean = np.mean(film_data, axis=0)
    mean_matrix = np.add(rest_mean, film_mean)/2
    indices = np.argsort(mean_matrix)[-100:]
    return rest_data[:, indices], film_data[:, indices]


# returning the indices with the most difference between film and rest
def max_diffrence_indices(rest_path, film_path, rest_file, film_file):
    '''
    create a mean matrix from rest and film data for single patient
    choose the 100 indices with the most difference between the rest and film data
    return arrys with the values in those indices for each second
    
    '''
    rest_data, film_data = data_fit(rest_path, film_path, rest_file, film_file)
    rest_mean = np.mean(rest_data, axis=0)
    film_mean = np.mean(film_data, axis=0)
    indices = np.argsort(film_mean - rest_mean)[-100:]
    return rest_data[:, indices], film_data[:, indices]


# calculate the max indices of the mean of rest and film data, return the data in those indices
def max_indices(rest_path, film_path, rest_file, film_file):
    '''
    creat mean matrix for each rest and film data for single patient 
    find the 50 max indices in each mean matrix (50 from rest and 50 from film)
    return arrys with the values in those indices for each second
    '''
    rest_data, film_data = data_fit(rest_path, film_path, rest_file, film_file)
    rest_mean = np.mean(rest_data, axis=0)
    rest_indices = np.sort(np.argsort(rest_mean)[-50:])
    film_mean = np.mean(film_data, axis=0)
    film_indices = np.sort(np.argsort(film_mean)[-50:])
    indices = np.append(rest_indices, film_indices)
    return rest_data[:, rest_indices], film_data[:, film_indices]


def matrix_transform(matrix_flat):
    
    '''
    create a 2D matrix from the flat matrix
    '''
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
    '''
    resize the matrix to 30x30
    '''
    resized_matrix = np.zeros(shape=(matrix.shape[0], 30, 30))
    for i in range(matrix.shape[0]):
        resized_matrix[i, :, :] = cv2.resize(matrix[i, :, :], (30, 30), interpolation=cv2.INTER_AREA)
    return resized_matrix


def matrix_fit(rest_path, film_path, rest_file, film_file):
    '''
    reshape the rest and film data to gray scale 30x30x1 matrix
    
    '''
    rest_data, film_data = data_fit(rest_path, film_path, rest_file, film_file)
    rest_data = matrix_transform(rest_data)
    rest_data = matrix_resize(rest_data)
    rest_data = np.reshape(rest_data, (rest_data.shape[0], rest_data.shape[1], rest_data.shape[2], 1))

    film_data = matrix_transform(film_data)
    film_data = matrix_resize(film_data)
    film_data = np.reshape(film_data, (film_data.shape[0], film_data.shape[1], film_data.shape[2], 1))
    return rest_data, film_data


def data_extract(freq_type_rest, freq_type_film, bounds, extract_func):
    '''
    bound = patients 
    
    input: arry of patients indecies
    ouput: rest and film data for all patients in the bound 
    
    '''
    rest_path = 'rest_data'
    patients = os.listdir(rest_path)
    film_path = 'film_data'

    first_p = bounds[0]
    last_p = bounds[1]

    rest_dict = op.join(rest_path, patients[first_p])
    film_dict = op.join(film_path, patients[first_p])

    rest_file = f'{patients[first_p]},task=rest,freq={freq_type_rest},sec=3.npz'
    film_file = f'{patients[first_p]},task=film,freq={freq_type_film},sec=3.npz'

    rest_data, film_data = extract_func(rest_dict, film_dict, rest_file, film_file)

    for i in range(first_p + 1, last_p):
        rest_dict = op.join(rest_path, patients[i])
        film_dict = op.join(film_path, patients[i])

        rest_file = f'{patients[i]},task=rest,freq={freq_type_rest},sec=3.npz'
        film_file = f'{patients[i]},task=film,freq={freq_type_film},sec=3.npz'
        temp_rest, temp_film = extract_func(rest_dict, film_dict, rest_file, film_file)

        rest_data = np.append(rest_data, temp_rest, axis=0)
        film_data = np.append(film_data, temp_film, axis=0)

    return rest_data, film_data
