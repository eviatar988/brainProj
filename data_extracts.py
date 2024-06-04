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

def load_data(path,file):
    loaded_file = np.load(op.join(path, file))
    return loaded_file['matrix_arr']

def read_file_rest_max(rest_path, file):
    loaded_file = np.load(op.join(rest_path,file))
    data = loaded_file['matrix_arr']
     
    '''print(data[:,np.argsort(data,axis=1)[-5:]].shape)
    return data[:,np.argsort(data,axis=1)[-5:]]'''"testing stuff.py"
    # return 5 most correlated channels for each channel
    # print(data[:,np.argsort(data,axis=1)[:,-5:]].shape)
    return np.sort(data, axis=1)[:, -100:]


def read_file_film_max(path, file):
     loaded_file = np.load(op.join(path,file))
     data = loaded_file['matrix_arr']
     '''print(data[:,np.argsort(data,axis=1)[-5:]].shape)
     return data[:,np.argsort(data,axis=1)[-5:]]'''"testing stuff.py"
     # return 5 most correlated channels for each channel
     # print(data[:,np.argsort(data,axis=1)[:,-5:]].shape)
     return np.sort(data, axis=1)[range(0, data.shape[0], 2), -100:]



def read_file_rest_random(path, file):
    loaded_file = np.load(op.join(path, file))
    data = loaded_file['matrix_arr']
    num_random_columns = 100
    # Randomly select column indices
    random_column_indices = np.random.choice(data.shape[1], num_random_columns, replace=False)
    # Extract random columns
    return data[:, random_column_indices]

def read_file_film_random(path, file):
    loaded_file = np.load(op.join(path, file))
    data = loaded_file['matrix_arr']
    num_random_columns = 100
    # Randomly select column indices
    random_column_indices = np.random.choice(data.shape[1], num_random_columns, replace=False)
    temp = data[:, random_column_indices]
    # Extract random columns
    return temp[range(0, data.shape[0], 2), :]


def feature_extract_rest(path, file):
    loaded_file = np.load(op.join(path, file))
    data = loaded_file['matrix_arr']
    features = np.zeros((data.shape[0], 4))
    for i in range(data.shape[0]):
        features[i, :] = [np.mean(data[i]), np.std(data[i]), np.min(data[i]), np.max(data[i])]
    return features


def feature_extract_film(path, file):
    loaded_file = np.load(op.join(path, file))
    data = loaded_file['matrix_arr']
    features = np.zeros((int(data.shape[0]/2), 4))
    for i in range(0, int(data.shape[0]/2)):
        features[i, :] = [np.mean(data[i*2]), np.std(data[i*2]), np.min(data[i*2]), np.max(data[i*2])]
    return features


def last_elements_rest(path, file):
    loaded_file = np.load(op.join(path, file))
    data = loaded_file['matrix_arr']
    '''print(data[:,np.argsort(data,axis=1)[-5:]].shape)
    return data[:,np.argsort(data,axis=1)[-5:]]'''"testing stuff.py"
    # return 5 most correlated channels for each channel
    # print(data[:,np.argsort(data,axis=1)[:,-5:]].shape)
    return data[:, -100:]

def last_elemets_film(path,file):
    loaded_file = np.load(op.join(path, file))
    data = loaded_file['matrix_arr']
    '''print(data[:,np.argsort(data,axis=1)[-5:]].shape)
    return data[:,np.argsort(data,axis=1)[-5:]]'''"testing stuff.py"
    # return 5 most correlated channels for each channel
    # print(data[:,np.argsort(data,axis=1)[:,-5:]].shape)
    temp = data[:, -100:]
    return temp[range(0, data.shape[0], 2)]


def max_indices(rest_data, film_data):
    rest_data = np.mean(rest_data, axis=0)
    rest_indices = np.argsort(rest_data)[-20:]
    film_data = np.mean(film_data, axis=0)
    film_indices = np.argsort(film_data)[-20:]
    return np.append(rest_indices, film_indices)



def data_trasnform(freq_type_rest, freq_type_film ,first_p, last_p, rest_func, film_func):
    rest_path = 'rest_data'
    patients = os.listdir(rest_path)
    film_path = 'film_data'

    rest_data = rest_func(op.join(rest_path, patients[first_p]),
                               f'{patients[first_p]},freq={freq_type_rest},sec_per_sample={3}.npz')
    film_data = film_func(op.join(film_path, patients[first_p]),
                          f'{patients[first_p]},freq={freq_type_film},sec_per_sample={3}.npz')

    for i in range(first_p+1,last_p+1):
        temp = rest_func(op.join(rest_path, patients[i]),
                               f'{patients[i]},freq={freq_type_rest},sec_per_sample={3}.npz')
        rest_data = np.append(rest_data, temp, axis=0)
        temp = film_func(op.join(film_path, patients[i]),
                               f'{patients[i]},freq={freq_type_film},sec_per_sample={3}.npz')
        film_data = np.append(film_data, temp, axis=0)
    return rest_data, film_data
    





