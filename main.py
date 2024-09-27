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
    'plv': (5,5)
}


def show_me_matrix(matrix_flat, name):
    matrix_len = int(math.sqrt(2*len(matrix_flat)+1/4)+1/2)
    matrix = np.zeros(shape=(matrix_len, matrix_len), dtype=float)
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
    data = np.load(op.join(task_dir_path, f'patient={patient},task={task},type={freq},sec=1.npz'))
    return data['matrix_arr']

def readfile_plv(task, patient):
    task_dir_path = op.join(f'{task}_data', f'patient={patient}')
    data = np.load(op.join(task_dir_path, f'patient={patient},task={task},type=plv,sec=1.npz'))
    return data['matrix_arr']

def get_bidsroot():
    return op.join(op.dirname(sample.data_path()), dataset)


def create_data(sec_per_sample, measurement):
        bids_root = get_bidsroot()
        patient_m = PatientsMatrix(bids_root, sec_per_sample)
        patient_m.save_matrix_to_file(measurement)


top_directory = 'C:\\Users\\Eyal Arad\\Documents\\GitHub\\brainProj\\rest_data'


# Function to rename files
def rename_files_in_directory(directory):
    # Walk through all directories and files
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if 'plv' in file_name:
            # Generate new file name
                new_name = file_name.replace('type=type=plv', 'type=plv')
                print(new_name)
                old_file_path = os.path.join(root, file_name)
                new_file_path = os.path.join(root, new_name)
                os.rename(old_file_path, new_file_path)
# Call the function on your directory
def main():
    # p_values_all = []
    # for freq in freq_dict:
    #     p_values = []
    #     for i in range(45):
    #         X_train, X_test, y_train, y_test = data_extracts.data_extract([i], data_extracts.raw_data, freq,1)
    #         rest_data = np.concatenate((X_train[y_train==0], X_test[y_test==0]))
    #         film_data = np.concatenate((X_train[y_train==1], X_test[y_test==1]))
    #         rest_data = np.mean(rest_data, axis=0)
    #         film_data = np.mean(film_data, axis=0)
    #         stat, pvalue = stats.ttest_rel(rest_data, film_data)
    #         p_values.append(pvalue)
    #     p_values = np.array(p_values)
    #     p_values_all.append(p_values)
    # bars = []
    # labels = []
    # keys = list(freq_dict.keys())
    # for i, p_values in enumerate(p_values_all):
    #     bars.append(np.sum(p_values < 0.05))
    #     bars.append(np.sum(p_values >= 0.05))
    #     labels.append(f'{keys[i]}:success')
    #     labels.append(f'{keys[i]}:failure')
    # plt.figure(figsize=(15, 6))
    # plt.ylabel('patients')
    # plt.bar(labels, bars)
    # plt.title(f'paired t-test success rate')
    # plt.show()
    """accuracy1 = test2.majority_vote_cross_eval(3, ml_algorithms.svm_classifier,data_extracts.max_indices,
                                         1, np.arange(45))
    print(accuracy1)
    accuracy2 = test2.majority_vote_cross_eval_single(3, ml_algorithms.svm_classifier,data_extracts.max_indices,
                                         1, np.arange(45))
    print(accuracy2)"""
    # acc_al = []
    # acc = test2.test_single_patient_majority_vote(ml_algorithms.random_forest, data_extracts.max_indices,1
    #                                     , np.arange(45))
    #
    # sns.boxplot(data=acc)
    # plt.title(f'Random Forest, single_patient, majority vote on coherence frequency values')
    # plt.ylabel('Accuracy')
    # plt.show()
    acc = test2.test_all_patients(ml_algorithms.svm_classifier, data_extracts.max_indices_film,'high_gamma' ,1
                                        , np.arange(45))
    print(acc)
    # sns.boxplot(data=acc)
    # plt.title(f'SVM classifier, single_patient, PLV measurements')
    # plt.ylabel('Accuracy')
    # plt.show()
    # print(test2.test_all_patients(ml_algorithms.svm_classifier, data_extracts.max_indices,'plv',
    #                                      1, np.arange(45),np.arange(45)))
    # accuracy4 = np.zeros(45)
    # for i in range(5):
    #     print(i)
    # accuracy4 += test2.test_single_patient(ml_algorithms.svm_classifier,data_extracts.max_indices,'plv',
                                         #1, np.arange(45))
    # print(accuracy4/5)
    # acc1 = 0.7897152818128996
    # acc2 = array = np.array([0.69113924, 0.93333333, 0.78974359, 1.0, 0.96153846, 0.58888889,
    #               0.68571429, 0.80833333, 0.82222222, 0.86666667, 0.83611111, 1.0,
    #               0.71842105, 0.73055556, 0.85897436, 0.99444444, 0.96153846, 0.96410256,
    #               0.81012658, 0.936, 0.87692308, 0.57222222, 0.59230769, 1.0,
    #               0.98421053, 0.98205128, 0.72658228, 0.6835443, 0.97692308, 0.6,
    #               0.99722222, 0.96153846, 0.76410256, 0.99487179, 0.82820513, 1.0,
    #               1.0, 0.94102564, 0.73611111, 0.98863636, 0.91794872, 1.0,
    #               0.9, 0.81012658, 0.57974684])
    # acc3 = 0.7919233004067403
    # acc4 = np.array([0.75443038, 0.96944444, 0.66666667, 1.0, 0.96153846, 0.64722222,
    #                0.81298701, 0.92777778, 0.67222222, 0.725, 0.89444444, 0.99444444,
    #                0.65263158, 0.79722222, 0.83846154, 0.97777778, 0.95897436, 0.99230769,
    #                0.74683544, 0.87733333, 0.9025641, 0.55833333, 0.68717949, 0.99506173,
    #                0.95526316, 0.95384615, 0.67088608, 0.68607595, 0.86153846, 0.60263158,
    #                0.96111111, 0.87179487, 0.86153846, 0.97948718, 0.8025641, 1.0,
    #                1.0, 0.84358974, 0.775, 0.99318182, 0.93589744, 0.98717949,
    #                0.86111111, 0.8556962, 0.64556962])
    # print(np.mean(acc2))
    # print(np.mean(acc4))
    # sns.boxplot(data=acc4)
    # plt.title(f'svm classifier, single_paitent, majority vote over freqs')
    # plt.show()
    """ef compute_single_state_indices_accuracy():
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
      
    # pred all patients freqs
    accuracy_randomForest = test1.pred_all_patients_freqs_2(ml_algorithms.random_forest, data_extracts.max_indices)
    accuracy_svm = test1.pred_all_patients_freqs_2(ml_algorithms.svm_classifier, data_extracts.max_indices)
    
    
    # todo: plot the single column accuracy
    
    print(accuracy_randomForest)
    print(accuracy_svm)
    
    # plot the accuracy of the random forest and svm
   
    plt.bar(['random forest', 'svm'], [np.mean(accuracy_randomForest), np.mean(accuracy_svm)])
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Random forest vs SVM - all patients - max indices')
    plt.show()""""""
       
   
    
    
    
    
    
    
 
    
    '''max_indices= test1.pred_all_patients_freqs(ml_algorithms.svm_classifier, data_extracts.max_indices)
    max_diffrence_indices= test1.pred_all_patients_freqs(ml_algorithms.svm_classifier, data_extracts.max_diffrence_indices)
    max_indices_rest= test1.pred_all_patients_freqs(ml_algorithms.svm_classifier, data_extracts.max_indices_rest)
    max_indices_film= test1.pred_all_patients_freqs(ml_algorithms.svm_classifier, data_extracts.max_indices_film)
'''
    #append all the accuracies to the list
    #all_patients_accuracy.append(max_indices_mean)
    #all_patients_accuracy.append(max_values)
    '''all_patients_accuracy.append(max_indices)
    all_patients_accuracy.append(max_diffrence_indices)
    all_patients_accuracy.append(max_indices_rest)
    all_patients_accuracy.append(max_indices_film)
    '''
   # print(all_patients_accuracy)
   # print(all_patients_accuracy.shape)
    #plot the accuracy use matplotlib
   
    
    
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
    
    
    
        
            
    ''' acc_al = []
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


