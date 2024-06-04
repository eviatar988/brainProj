import numpy as np
from sklearn.model_selection import train_test_split

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


def data_split(rest_data , film_data):
    rest_shape = rest_data.shape[0]
    X = np.append(film_data, rest_data, axis=0)
    y = np.zeros(rest_shape * 2)
    y[:rest_shape] = 1
    # Split data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def random_forest_all(rest_func, film_func):
    bounds = (0, 43)
    predictions = []
    for freq in freq_dict.keys():
        data = data_extracts.data_trasnform(freq, freq, bounds)
        X_train, X_test, y_train, y_test = data_split(data)
        predictions.append(ml_algorithms.random_forest(X_train, X_test, y_train, y_test))
    return predictions

def random_forest_single(index, rest_func , film_func):
    predictions = []
    for freq in freq_dict.keys():
        data = data_extracts.data_trasnform(freq, freq, index, index, rest_func, film_func)
        X_train, X_test, y_train, y_test = data_split(data)
        predictions.append(ml_algorithms.random_forest(X_train, X_test, y_train, y_test))
    return predictions

def svm_single(index, rest_func , film_func):
    predictions = []
    for freq in freq_dict.keys():
        data = data_extracts.data_trasnform(freq, freq, index, index, rest_func, film_func)
        X_train, X_test, y_train, y_test = data_split(data)
        predictions.append(ml_algorithms.svm_classifier(X_train, X_test, y_train, y_test))
    return predictions

def svm_all(rest_func , film_func):
    predictions = []
    for freq in freq_dict.keys():
        data = data_extracts.data_trasnform(freq, freq, 0, 43, rest_func, film_func)
        X_train, X_test, y_train, y_test = data_split(data)
        predictions.append(ml_algorithms.svm_classifier(X_train, X_test, y_train, y_test))
    return predictions