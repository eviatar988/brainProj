import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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


def data_split(rest_data, film_data):
    rest_shape = rest_data.shape[0]
    X = np.append(film_data, rest_data, axis=0)
    y = np.zeros(rest_shape * 2)
    y[:rest_shape] = 1
    # Split data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def pred_all_patients(model, func, freq):
    rest_data, film_data = data_extracts.data_extract(freq, freq, (0, 44), func)
    X_train, X_test, y_train, y_test = data_split(rest_data, film_data)
    y_pred = model(X_train, X_test, y_train, y_test)
    return accuracy_score(y_test, y_pred)


def pred_single_frequency(model_type, func, frequency):
    accuracy = []
    for i in range(45):
        rest_data, film_data = data_extracts.data_extract(frequency, frequency, (i, i), func)
        X_train, X_test, y_train, y_test = data_split(rest_data, film_data)
        model = model_type(X_train, X_test)
        y_pred = model.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
    return accuracy


def pred_all_frequencys(model_type, func):
    accuracy = []
    for i in range(45):
        y_pred = []
        for freq in list(freq_dict.keys())[1:6]:
            rest_data, film_data = data_extracts.data_extract(freq, freq, (i, i), func)
            X_train, X_test, y_train, y_test = data_split(rest_data, film_data)
            model = model_type(X_train, y_train)
            if len(y_pred) == 0:
                y_pred = model.predict(X_test)
            else:
                y_pred += model.predict(X_test)
        y_pred = np.where(y_pred > 2, 1, 0)
        accuracy.append(accuracy_score(y_test, y_pred))
    return accuracy
