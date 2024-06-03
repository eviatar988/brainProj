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



def read_file_film_random(path, file):
    return 1

def data_trasnform(freq_type_rest, freq_type_film ,first_p, last_p, rest_func, film_func):
    rest_path = 'rest_data'
    patients = os.listdir(rest_path)
    film_path = 'film_data'

    rest_data = rest_func(op.join(rest_path, patients[first_p]),
                               f'{patients[first_p]},freq={freq_type_rest},sec_per_sample={1}.npz')
    film_data = film_func(op.join(film_path, patients[first_p]),
                          f'{patients[first_p]},freq={freq_type_film},sec_per_sample={1}.npz')

    for i in range(first_p+1,last_p+1):
        temp = rest_func(op.join(rest_path, patients[i]),
                               f'{patients[i]},freq={freq_type_rest},sec_per_sample={1}.npz')
        rest_data = np.append(rest_data, temp, axis=0)
        temp = film_func(op.join(film_path, patients[i]),
                               f'{patients[i]},freq={freq_type_film},sec_per_sample={1}.npz')
        film_data = np.append(film_data, temp, axis=0)
    return rest_data, film_data
    





