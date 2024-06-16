import math
import os
import os.path as op
import mne_bids
import scipy.stats
from mne.datasets import sample
import data_extracts
import ml_algorithms
import patients_matrix
from patients_matrix import PatientsMatrix
from coherence_matrix import CoherenceMatrix
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import levene, gaussian_kde
from scipy.stats import anderson
import scipy.stats as stats
import seaborn as sns
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

    plt.imshow(matrix, cmap='viridis')
    plt.title(name)
    plt.colorbar()
    plt.show()


def readfile(task, patient, freq):
    task_dir_path = op.join(f'{task}_data' ,f'patient={patient}')
    data = np.load(op.join(task_dir_path, f'patient={patient},task={task},freq={freq}.npz'))
    return data['matrix_arr']


def get_bidsroot():
    return op.join(op.dirname(sample.data_path()), dataset)


def create_data():
        bids_root = get_bidsroot()
        patient_m = PatientsMatrix(bids_root, 1)
        patient_m.save_matrix_to_file()

def main():

    rest_path = 'rest_data'
    patients = os.listdir(rest_path)
    film_path = 'film_data'

    for freq in freq_dict.keys():
        p_values = []
        rest_avg = []
        film_avg = []
        for i, patient in enumerate(patients):
            rest_data, film_data = data_extracts.data_extract(freq,freq,(i,i), data_extracts.max_indices)
            rest_data = np.mean(rest_data,axis=0)
            film_data = np.mean(film_data, axis=0)
            print('rest: ',rest_data)
            print('film: ',film_data)
            stat, pvalue = stats.ttest_ind(rest_data, film_data)
            p_values.append(pvalue)
        print(np.argmax(p_values))
        sns.boxplot(data=p_values)
        plt.show()

        """rest_avg.append(np.mean(readfile('rest', patient[-2:], freq),axis=0))
        film_avg.append(np.mean(readfile('film', patient[-2:], freq),axis=0))"""


    """ stat, pvalue = stats.ttest_ind(rest_avg, film_avg)
        p_values.append(pvalue)
    print(p_values)

    plt.figure(figsize=(7,7))
    plt.bar(list(freq_dict.keys()),p_values,)
    plt.xlabel('Frequency type')
    plt.ylabel('Pvalue')
    plt.title('paired t-test result for mean value of coherence for each patient')
    plt.show()"""







if __name__ == '__main__':
    main()

