import os

import numpy as np
import shap
from lime import lime_tabular
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import ml_algorithms
import data_extracts

freq_dict = {
    'delta': (1, 3),
    'theta': (3, 5),
    'alpha': (5, 7),
    'beta': (7, 16),
    'low_gamma': (17, 36),
    'high_gamma': (36, 126)

}


def data_preparation(x_train , x_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    """pca = PCA(n_components=10)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)"""
    return x_train, x_test


def test_all_patients(model_type, func, data_type, sample_size, p_train, p_test=None):
    x_train, x_test, y_train, y_test = data_extracts.data_extract(p_train, func, data_type, sample_size)
    if p_test is not None:
        _, x_test, _, y_test = data_extracts.data_extract(p_test, data_extracts.max_values_test,data_type, sample_size)
    # if p_test is not None:
    #     _, x_test, _, y_test = data_extracts.data_extract(p_test, func, data_type, sample_size)
    x_train, x_test = data_preparation(x_train, x_test)
    model = model_type(x_train, y_train)
    y_pred = model.predict(x_test)

    return accuracy_score(y_test, y_pred)



def test_single_patient_majority_vote(model_type, func, sample_size, p_train, seed=42):
    accuracy = []
    for i in p_train:
        y_pred = []
        for data_type in list(freq_dict.keys())[1:6]:
            np.random.seed(seed)
            X_train, X_test, y_train, y_test = data_extracts.data_extract([i], func, data_type, sample_size)
            X_train, X_test = data_preparation(X_train, X_test)
            model = model_type(X_train, y_train)
            if len(y_pred) == 0:
                y_pred = model.predict(X_test)
            else:
                y_pred += model.predict(X_test)
        y_pred = np.where(y_pred > 2, 1, 0)
        accuracy.append(accuracy_score(y_test, y_pred))
    return accuracy

def test_single_patient(model_type, func, data_type,sample_size, p_train):
    accuracy = []
    for i in p_train:
        X_train, X_test, y_train, y_test = data_extracts.data_extract([i], func, data_type, sample_size)
        X_train, X_test = data_preparation(X_train, X_test)
        model = model_type(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
    return accuracy




def test_all_patient_majority_vote(model_type, func, sample_size, p_train, seed=42, p_test=None):
    '''
    this func evluates the model on each frequency range
    and in addition to that it evaluates the model on all the frequency ranges (majority voting)
    '''

    y_pred = []  # to store the sum of all predictions (for majority voting)
    for data_type in list(freq_dict.keys())[1:6]:
        np.random.seed(seed)
        x_train, x_test, y_train, y_test = data_extracts.data_extract(p_train, func, data_type, sample_size)
        if p_test is not None:
            _, x_test, _, y_test = data_extracts.data_extract(p_test, func, data_extracts.max_values_test, sample_size)
        x_train, x_test = data_preparation(x_train, x_test)
        model = model_type(x_train, y_train)
        if len(y_pred) == 0:
            y_pred = model.predict(x_test)
        else:
            y_pred += model.predict(x_test)
    y_pred = np.where(y_pred > 2, 1, 0)
    return accuracy_score(y_test, y_pred)

def majority_vote_cross_eval(eval_num, model_type, func, sample_size, p_train, p_test=None):
    accuracy = []
    for i in range(eval_num):
        print(i)
        seed = np.random.randint(0, 10000)  # Generate a new seed for each iteration
        np.random.seed(seed)
        accuracy.append(test_all_patient_majority_vote(model_type, func, sample_size, p_train, seed, p_test))
    return np.mean(np.array(accuracy))

def majority_vote_cross_eval_single(eval_num, model_type, func, sample_size, p_train, p_test=None):
    accuracy = np.zeros(45)
    for i in range(eval_num):
        print(i)
        seed = np.random.randint(0, 10000)  # Generate a new seed for each iteration
        np.random.seed(seed)
        accuracy += test_single_patient_majority_vote(model_type, func, sample_size, p_train, seed)
    return accuracy/5

#majority_vote_cross_eval(5, ml_algorithms.svm_classifier,data_extracts.max_indices,1, np.arange(45))