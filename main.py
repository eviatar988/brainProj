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
    'low gamma': (30, 70),
    'high gamma': (70, 250)
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
    patient_m = PatientsMatrix(bids_root, "beta")
    patient_m.save_matrix_to_file()
    """rest_mean_list = []
    film_mean_list = []
    matrixs = np.load(op.join(rest_lists_path + 'delta', '26' + '_rest_matrixs.npz'))
    ars1 = matrixs['arr_rest']
    matrixs = np.load(op.join(film_lists_path + 'alpha', '26' + '_film_matrixs.npz'))
    ars2 = matrixs['arr_film']
    show_me_matrix(ars2[120],500)"""
    """for i in range(100):
        rest_mean_list.append(np.var(ars1[i].flatten()))
        film_mean_list.append(np.var(ars2[i].flatten()))



# Calculate Pearson's correlation coefficient
    correlation, p_value = stats.pearsonr(rest_mean_list, film_mean_list)

    print("Pearson Correlation Coefficient:", correlation)
    print("p-value:", p_value)"""

if __name__ == '__main__':
    main()

