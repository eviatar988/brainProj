import math
import os
import os.path as op
import mne_bids
import scipy.stats
import sklearn.metrics
from mne.datasets import sample
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

import data_extracts
import ml_algorithms
import patients_matrix
import test2
from patients_matrix import PatientsMatrix
import coherence_matrix
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
    'high_gamma': (70, 250),
    'plv' : (5,5)
}


pvalues = []
for freq in freq_dict.keys():
    rest_avg = []
    film_avg = []

    for i in range(45):
        X_train, X_test, y_train, y_test = data_extracts.data_extract([i], data_extracts.max_values, freq, 1)
        rest_data = np.concatenate((X_train[y_train == 0], X_test[y_test == 0]))
        film_data = np.concatenate((X_train[y_train == 1], X_test[y_test == 1]))
        rest_avg.append(np.mean(rest_data))
        film_avg.append(np.mean(film_data))
        if (freq == 'beta'):
            print(np.mean(rest_data), np.mean(film_data))
    stat, pvalue = stats.ttest_rel(rest_avg, film_avg)
    pvalues.append(pvalue)
print(pvalues)
plt.bar(freq_dict.keys(), pvalues)
plt.title('t-test,pvalues for mean values of all the patients')
plt.axhline(y=0.05, color='red', linestyle='--')
plt.show()
pvalues = []
for freq in freq_dict.keys():
    rest_avg = []
    film_avg = []
    for i in range(45):
        X_train, X_test, y_train, y_test = data_extracts.data_extract([i],data_extracts.max_values,freq,1)
        rest_data = X_train[y_train == 0]
        film_data = X_train[y_train == 1]
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
"""
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


acc_al = []
for freq in freq_dict.keys():
    acc = test1.pred_single_frequency(ml_algorithms.random_forest, data_extracts.max_indices, freq)
    print(np.mean(acc))
    acc_al.append(acc)

plt.figure(figsize=(10, 6))  # Optional: Adjust figure size

plt.boxplot(acc_al, positions=[1, 2, 3, 4, 5, 6])  # Positions for the groups

# Optional: Add labels to x-axis
plt.xticks([1, 2, 3, 4, 5, 6], list(freq_dict.keys()))
plt.xlabel('Frequency ranges')
plt.ylabel('Accuracy')
plt.title('Boxplot Of Accuracies for single patient case,Random Forest')
plt.grid(True)

plt.show()

acc = test1.pred_all_frequencys(ml_algorithms.svm_classifier, data_extracts.max_diffrence_indices)
print(np.mean(acc))
sns.boxplot(acc)
plt.show()
"""


def calculate_plv(signal1, signal2):
    """
    Calculate the Phase-Locking Value (PLV) between two signals.

    Parameters:
        signal1 (numpy.ndarray): First signal.
        signal2 (numpy.ndarray): Second signal.

    Returns:
        float: PLV value between signal1 and signal2.
    """
    if len(signal1) != len(signal2):
        raise ValueError("Signals must have the same length.")

    # Apply Hilbert transform to get the analytic signal
    analytic_signal1 = hilbert(signal1)
    analytic_signal2 = hilbert(signal2)

    # Extract the phase from the analytic signals
    phase1 = np.angle(analytic_signal1)
    phase2 = np.angle(analytic_signal2)
    print(phase1)
    print(phase2)
    # Calculate the phase difference
    phase_diff = phase1 - phase2

    # Compute the PLV
    print(np.exp(1j * phase_diff))
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))

    return plv

print(calculate_plv(np.arange(10),np.ones(10)))