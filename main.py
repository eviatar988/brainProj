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
    data = np.load(op.join(task_dir_path, f'patient={patient},task={task},freq={freq},sec=3.npz'))
    return data['matrix_arr']


def get_bidsroot():
    return op.join(op.dirname(sample.data_path()), dataset)


def create_data():
        bids_root = get_bidsroot()
        patient_m = PatientsMatrix(bids_root, 3)
        patient_m.save_matrix_to_file()

def main():

    def compute_overlap():
        accuracy_per_freq = []
        for freq in freq_dict.keys():
            temp_Arr = []
            for i in range(45):   

                rest_indices,film_indices = data_extracts.data_extract(freq, freq, (i, i), data_extracts.get_max_indices_mean_matrix)
                
                #common indices
                common_indices = np.intersect1d(rest_indices, film_indices)
                common_indices_len = common_indices.shape[0]
                temp_Arr.append(common_indices_len)
            accuracy_per_freq.append(temp_Arr)    
        
        # boxplot of the common indices
        plt.figure(figsize=(10, 6))  # Optional: Adjust figure size
        plt.boxplot(accuracy_per_freq, positions=[1, 2, 3, 4, 5, 6])  # Positions for the groups
        # Optional: Add labels to x-axis
        plt.xticks([1, 2, 3, 4, 5, 6], list(freq_dict.keys()))
        plt.xlabel('Frequency ranges')
        plt.ylabel('Common indices')
        plt.title('Boxplot Of Common indices for single patient case over mean matrix')
        plt.grid(True)
        plt.show()
  
    
    accuracy_rest_indices = test1.pred_single_state_indices('rest',ml_algorithms.random_forest)
    accuracy_film_indices = test1.pred_single_state_indices('film',ml_algorithms.random_forest)
    
    #plot the accuracy of the rest and film indices
    plt.figure(figsize=(10, 6))  # Optional: Adjust figure size
    plt.boxplot([accuracy_rest_indices,accuracy_film_indices], positions=[1, 2])  # Positions for the groups
    # Optional: Add labels to x-axis
    plt.xticks([1, 2], ['rest', 'film'])
    plt.xlabel('State')
    plt.ylabel('Accuracy')
    plt.title('single state indices max values  - random forest')
    plt.grid(True)
    plt.show()
    
    
        
    rest_path = 'rest_data'
    patients = os.listdir(rest_path)
    film_path = 'film_data'
    '''print(len(patients))
    for freq in freq_dict.keys():
        print(freq+': ', test1.pred_all_patients(ml_algorithms.svm_classifier, data_extracts.max_values, freq))'''
        
    #print(test1.pred_all_patients_freqs(ml_algorithms.svm_classifier, data_extracts.max_indices))
    #print(test1.pred_all_patients_freqs(ml_algorithms.svm_classifier, data_extracts.max_indices))
    '''
    accuracy = []
    for freq in freq_dict.keys():
        single_freq_acc = []
        for patient in range(44):
            counter = 0
            rest_data, film_data = data_extracts.data_extract(freq, freq, (patient, patient), data_extracts.max_indices_mean)
            for sec in range(rest_data.shape[0]):
                rest_val = np.mean(rest_data[sec, :])
                film_val = np.mean(film_data[sec, :])
                if film_val > rest_val:
                    counter += 1
            patient_acc = counter / rest_data.shape[0]
            single_freq_acc.append(patient_acc)
        accuracy.append(single_freq_acc)
        
        
    plt.figure(figsize=(10, 6))  # Optional: Adjust figure size
    plt.boxplot(accuracy, positions=[1, 2, 3, 4, 5, 6])  # Positions for the groups
    # Optional: Add labels to x-axis
    plt.xticks([1, 2, 3, 4, 5, 6], list(freq_dict.keys()))
    plt.xlabel('Frequency ranges')
    plt.ylabel('Accuracy')
    plt.title('Boxplot Of Accuracies for single patient case')
    plt.grid(True)
    plt.show()"""
    
    
    
        
            
''' '''acc_al = []
    for freq in freq_dict.keys():
        acc = test1.pred_single_frequency(ml_algorithms.svm_classifier, data_extracts.max_indices, freq)
        print(np.mean(acc))
        acc_al.append(acc)

    plt.figure(figsize=(10, 6))  # Optional: Adjust figure size

    plt.boxplot(acc_al, positions=[1, 2, 3, 4, 5, 6])  # Positions for the groups

    # Optional: Add labels to x-axis
    plt.xticks([1, 2, 3, 4, 5, 6], list(freq_dict.keys()))
    plt.xlabel('Frequency ranges')
    plt.ylabel('Accuracy')
    plt.title('Boxplot Of Accuracies for single patient case,SVM')
    plt.grid(True)

    plt.show()

    for freq in freq_dict.keys():
        #  ?חוזרים פעמיים על אותו דבר
        acc = test1.pred_all_frequencys(ml_algorithms.svm_classifier, data_extracts.max_indices_mean)
        plt.boxplot(acc)
        plt.title(f"{freq}")
        plt.show()'''


if __name__ == '__main__':
    main()


