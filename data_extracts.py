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

# %%
rest_lists_path = 'rest_lists'
film_lists_path = 'film_lists'



# %%
def read_file_rest (file, rest_path):
     loaded_file = np.load(op.join(rest_path,file))
     data = loaded_file['arr_rest']
     
     '''print(data[:,np.argsort(data,axis=1)[-5:]].shape)
     return data[:,np.argsort(data,axis=1)[-5:]]'''"testing stuff.py"
     # return 5 most correlated channels for each channel
     # print(data[:,np.argsort(data,axis=1)[:,-5:]].shape)
     return np.sort(data,axis=1)[:,-30:]
     

# %%
def read_file_film (file, rest_path):
     loaded_file = np.load(op.join(rest_path,file))
     data = loaded_file['arr_film']
     
     '''print(data[:,np.argsort(data,axis=1)[-5:]].shape)
     return data[:,np.argsort(data,axis=1)[-5:]]'''"testing stuff.py"
     # return 5 most correlated channels for each channel
     # print(data[:,np.argsort(data,axis=1)[:,-5:]].shape)
     return np.sort(data,axis=1)[range(0,data.shape[0],2),-30:]
     

# %%
def data_trasnform(freq_type_film, freq_type_rest):
    rest_path = rest_lists_path+freq_type_rest
    rest_files = os.listdir(rest_path)
    print(rest_files)
    
    film_path = film_lists_path+freq_type_film
    film_files = os.listdir(film_path)
    print(film_files)
    
    rest_data = read_file_rest(rest_files[0], rest_path)
    for i in range(1,len(rest_files)):  
        temp = read_file_rest(rest_files[i], rest_path)
        rest_data = np.append(rest_data, temp, axis=0)
    film_data = read_file_film(film_files[0], film_path)
    for i in range(1,len(film_files)):  
        temp = read_file_film(film_files[i], film_path)
        film_data = np.append(film_data, temp, axis=0)
        
    return rest_data, film_data
    





