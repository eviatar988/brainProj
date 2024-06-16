import math
import os
import os.path as op
import mne_bids
import scipy.stats
import sklearn.metrics
from mne.datasets import sample
from sklearn.metrics import accuracy_score
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
    print(len(patients))

    """pvalues = []
    for freq in freq_dict.keys():
        rest_avg = []
        film_avg = []

        for i, patient in enumerate(patients):
            rest_data, film_data = data_extracts.data_extract(freq, freq, (i, i), data_extracts.data_fit)
            rest_avg.append(np.mean(rest_data))
            film_avg.append(np.mean(film_data))
        stat, pvalue = stats.ttest_rel(rest_avg, film_avg)
        pvalues.append(pvalue)
    print(pvalues)
    plt.bar(freq_dict.keys(),pvalues)
    plt.title('t-test,pvalues for mean values of all the patients')
    plt.axhline(y=0.05, color='red', linestyle='--')
    plt.show()
    pvalues = []
    for freq in freq_dict.keys():
        rest_avg = []
        film_avg = []
        for i, patient in enumerate(patients):
            rest_data, film_data = data_extracts.data_extract(freq, freq, (i, i), data_extracts.data_fit)
            rest_avg.append(np.mean(rest_data))
            film_avg.append(np.mean(film_data))
        stat, pvalue = stats.f_oneway(rest_avg, film_avg)
        pvalues.append(pvalue)
    print(pvalues)
    plt.bar(freq_dict.keys(), pvalues)
    plt.title('f-test,pvalues for mean values of all the patients')
    plt.axhline(y=0.05, color='red', linestyle='--')
    plt.show()
    p_values_all = []
    for freq in freq_dict:
        p_values = []
        for i, patient in enumerate(patients):
            rest_data, film_data = data_extracts.data_extract(freq, freq, (i, i), data_extracts.data_fit)
            rest_data = np.mean(rest_data, axis=0)
            film_data = np.mean(film_data, axis=0)
            stat, pvalue = stats.ttest_rel(rest_data, film_data)
            p_values.append(pvalue)
        p_values = np.array(p_values)
        p_values_all.append(p_values)
    bars = []
    labels = []
    keys = list(freq_dict.keys())
    for i, p_values in enumerate(p_values_all):
        bars.append(np.sum(p_values < 0.05))
        bars.append(np.sum(p_values >= 0.05))
        labels.append(f'{keys[i]}:success')
        labels.append(f'{keys[i]}:failure')
    plt.figure(figsize=(15, 6))
    plt.ylabel('patients')
    plt.bar(labels, bars)
    plt.title(f'paired t-test success rate')
    plt.show()
    """
    svm_results = test1.pred_all_freqs(ml_algorithms.svm_classifier, data_extracts.max_indices)
    sns.boxplot(data=svm_results)
    plt.title(f'svm classifier')
    plt.show()
    rf_results = test1.pred_all_freqs(ml_algorithms.random_forest, data_extracts.max_indices)
    sns.boxplot(data=rf_results)
    plt.title(f'random forest')
    plt.show()
    svm_results = test1.pred_all_freqs(ml_algorithms.svm_classifier, data_extracts.max_diffrence_indices)
    print(svm_results)
    sns.boxplot(data=svm_results)
    plt.title(f'svm classifier')
    plt.show()
    rf_results = test1.pred_all_freqs(ml_algorithms.random_forest, data_extracts.max_diffrence_indices)
    sns.boxplot(data=rf_results)
    plt.title(f'random forest')
    plt.show()








if __name__ == '__main__':
    main()

