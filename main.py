import math
import os.path as op
from mne.datasets import sample
from patients_matrix import PatientsMatrix
from coherence_matrix import CoherenceMatrix
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import levene, gaussian_kde
from scipy.stats import anderson
import scipy.stats as stats

bids_path = r"C:\Users\eyala\Documents\GitHub\brainProj\ds003688"
dataset = "ds003688"
subject = "07"
session = 'iemu'
datatype = 'ieeg'
acquisition = 'clinical'
suffix = 'ieeg'
run = '1'
rest_lists_path = 'rest_lists'
film_lists_path = 'film_lists'
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
    plt.imshow(matrix, cmap='viridis')
    plt.title(name)
    plt.colorbar()
    plt.show()



def main():


    bids_root = op.join(op.dirname(sample.data_path()), dataset)
    patient_m = PatientsMatrix(bids_root, "low_gamma")
    patient_m.save_matrix_to_file()
    rest_mean_list = []
    film_mean_list = []
    matrixs = np.load(op.join(rest_lists_path + 'low_gamma', '17' + '_rest_matrixs.npz'))
    ars1 = matrixs['arr_rest']
    matrixs = np.load(op.join(film_lists_path + 'low_gamma', '17' + '_film_matrixs.npz'))
    ars2 = matrixs['arr_film']
    len1= len(ars1[0])
    len2 = len(ars2[0])
    show_me_matrix(ars1[10],52)
    count = 0
    for i in range(180):
        rest_mean_list.append(ars1[i][np.argsort(ars1[i])[-1:]])
        film_mean_list.append(ars2[i][np.argsort(ars2[i])[-1:]])
        if np.max(ars1[i]) < np.max(ars2[i]):
            count += 1

    print(count)
    print(np.mean(film_mean_list))
    print(np.mean(rest_mean_list))

if __name__ == '__main__':
    main()

