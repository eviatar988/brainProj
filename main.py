import math
import os.path as op
import mne_bids
from mne.datasets import sample
import data_extracts
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
    avg = np.zeros((44, 6))
    """print('svm single, 100 highest')
    for index in range(44):
        avg[index] = test1.svm_single(index, data_extracts.read_file_rest_max, data_extracts.read_file_film_max)
        print(f'patient: {index}:', avg[index])
    print("avg =",np.mean(avg,axis=0))
    print('svm all, 100 highest')
    print(test1.svm_all(data_extracts.read_file_rest_max, data_extracts.read_file_film_max))
    print('Rf single, 100 highest')
    for index in range(44):
        avg[index] = test1.random_forest_single(index, data_extracts.read_file_rest_max, data_extracts.read_file_film_max)
        print(f'patient: {index}:', avg[index])

    print("avg =", np.mean(avg, axis=0))
    print('RF all, 100 highest')
    print(test1.random_forest_all(data_extracts.read_file_rest_max, data_extracts.read_file_film_max))



    print('svm single, features')
    for index in range(44):
        avg[index] = test1.svm_single(index, data_extracts.feature_extract_rest, data_extracts.feature_extract_film)
        print(f'patient: {index}:', avg[index])
    print("avg =", np.mean(avg, axis=0))
    print('svm all, features')
    print(test1.svm_all(data_extracts.feature_extract_rest, data_extracts.feature_extract_film))
    print('RT single, features')
    for index in range(44):
        avg[index] = test1.random_forest_single(index, data_extracts.feature_extract_rest, data_extracts.feature_extract_film)
        print(f'patient: {index}:', avg[index])
    print("avg =", np.mean(avg, axis=0))
    print('RN all, features')
    print(test1.random_forest_all(data_extracts.feature_extract_rest, data_extracts.feature_extract_film))

    print('svm single, last_elements')
    for index in range(44):
        avg[index] = test1.svm_single(index, data_extracts.last_elements_rest, data_extracts.last_elemets_film)
        print(f'patient: {index}:', avg[index])
    print("avg =", np.mean(avg, axis=0))
    print('svm all, last_elements')
    print(test1.svm_all(data_extracts.last_elements_rest, data_extracts.last_elemets_film))
    print('RT single, last_elementss')
    for index in range(44):
        avg[index] = test1.random_forest_single(index, data_extracts.last_elements_rest, data_extracts.last_elemets_film)
        print(f'patient: {index}:', avg[index])
    print("avg =", np.mean(avg, axis=0))
    print('svm all, last_elements')
    print(test1.random_forest_all(data_extracts.last_elements_rest, data_extracts.last_elemets_film))
    """
    data = data_extracts.data_trasnform('low_gamma', 'low_gamma', 0, 0, data_extracts.load_data
                                 , data_extracts.load_data)
    rest_data = data[0]
    film_data = data[1]
    print(rest_data.shape)
    print(film_data.shape)
    rest_data = np.mean(rest_data, axis=0)
    sorted_indices = np.argsort(rest_data)
    print(rest_data[sorted_indices[-20:]])
    print(sorted_indices[-20:])


    film_data = np.mean(film_data, axis=0)
    sorted_indices = np.argsort(film_data)
    print(film_data[sorted_indices[-20:]])
    print(sorted_indices[-20:])
    print(rest_data[sorted_indices[-20:]])
    print(np.max)





if __name__ == '__main__':
    main()

