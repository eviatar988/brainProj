import math
import os.path as op
import mne_bids
from mne.datasets import sample
import data_extracts
import ml_algorithms
from patients_matrix import PatientsMatrix
from coherence_matrix import CoherenceMatrix
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import levene, gaussian_kde
from scipy.stats import anderson
import scipy.stats as stats
import test1

bids_path = r"C:\Users\eyala\Documents\GitHub\brainProj\ds003688"
dataset = "ds003688"
subject = "07"
session = 'iemu'
datatype = 'ieeg'
acquisition = 'clinical'
suffix = 'ieeg'
run = '1'
rest_data_path = 'rest_data'
film_data_path = 'film_data'
freq_dict = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'low_gamma': (30, 70),
    'high_gamma': (70, 250)
}


def show_me_matrix(matrix_flat, name):
    matrix_len = int(math.sqrt(2*len(matrix_flat)+1/4)+1/2)
    matrix = np.zeros(shape=(matrix_len,matrix_len),dtype=float)
    index = 0
    for i in range(matrix_len):
        for j in range(i,matrix_len):
            if i == j:
                matrix[i, j] = 1
            else:
                matrix[i, j] = matrix_flat[index]
                matrix[j, i] = matrix_flat[index]
                index += 1
    temp = matrix.reshape(20, 20)
    plt.imshow(temp, cmap='viridis')
    plt.title(name)
    plt.colorbar()
    plt.show()


def readfile(task, patient, freq, sec_per_sample):
    task_dir_path = op.join(f'{task}_data' ,f'patient={patient}')
    data = np.load(op.join(task_dir_path, f'patient={patient},freq={freq},sec_per_sample={sec_per_sample}.npz'))
    return data['matrix_arr']

def get_bidsroot():
    return op.join(op.dirname(sample.data_path()), dataset)


def create_data():
        bids_root = get_bidsroot()
        patient_m = PatientsMatrix(bids_root, 1)
        patient_m.save_matrix_to_file()

def main():
    for i in range(43):
        try:
            freq = []
            for key in freq_dict.keys():
                rest_data, film_data = data_extracts.data_extract(key, key, (i, i), 3, data_extracts.max_indices)
                x_train, x_test, y_train, y_test = test1.data_split(rest_data, film_data)
                freq.append(ml_algorithms.svm_classifier(x_train, x_test, y_train, y_test))
            print(freq)
        except:
            print(i,'sucks')





if __name__ == '__main__':
    main()

