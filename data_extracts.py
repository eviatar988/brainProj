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





# %%
def read_file_rest (file, rest_path):
     loaded_file = np.load(op.join(rest_path,file))
     data = loaded_file['matrix_arr']
     
     '''print(data[:,np.argsort(data,axis=1)[-5:]].shape)
     return data[:,np.argsort(data,axis=1)[-5:]]'''"testing stuff.py"
     # return 5 most correlated channels for each channel
     # print(data[:,np.argsort(data,axis=1)[:,-5:]].shape)
     return np.sort(data,axis=1)[:,-30:]
     

# %%
def read_file(path, file):
     loaded_file = np.load(op.join(path,file))
     data = loaded_file['matrix_arr']
     
     '''print(data[:,np.argsort(data,axis=1)[-5:]].shape)
     return data[:,np.argsort(data,axis=1)[-5:]]'''"testing stuff.py"
     # return 5 most correlated channels for each channel
     # print(data[:,np.argsort(data,axis=1)[:,-5:]].shape)
     return np.sort(data,axis=1)[range(0,data.shape[0],2),-30:]
     

# %%




def data_trasnform(freq_type_film, freq_type_rest):


    rest_path = 'rest_data'
    patients = os.listdir(rest_path)

    film_path = 'film_data'

    rest_data = read_file(op.join(rest_path, patients[0]),
                               f'{patients[0]},freq={freq_type_rest},sec_per_sample={3}.npz')
    film_data = read_file(op.join(film_path, patients[0]),
                          f'{patients[0]},freq={freq_type_rest},sec_per_sample={3}.npz')
    for i in range(1,len(patients)):
        temp = read_file(op.join(rest_path, patients[i]),
                               f'{patients[i]},freq={freq_type_rest},sec_per_sample={3}.npz')
        rest_data = np.append(rest_data, temp, axis=0)

        temp = read_file(op.join(film_path, patients[i]),
                               f'{patients[i]},freq={freq_type_rest},sec_per_sample={3}.npz')
        film_data = np.append(film_data, temp, axis=0)
        
    return rest_data, film_data
    





