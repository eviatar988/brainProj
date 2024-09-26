import os

import numpy as np
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


def data_split(rest_data, film_data):
    rest_shape = rest_data.shape[0]
    X = np.append(film_data, rest_data, axis=0)
    y = np.zeros(rest_shape * 2) # becase film and rest data are the same size
    y[:rest_shape] = 1 # 1 for film, 0 for rest, it is more accurate to use write 'film shape' instead of 'rest shape' but it is the same
    # Split data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def normalize_matrix(data):
    row_norms = np.linalg.norm(data, axis=1, keepdims=True)
    return data / row_norms

def data_preparation(x_train , x_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    """pca = PCA(n_components=10)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)"""
    return x_train, x_test

def pred_all_patients_plv(model_type, func):
    x_train, x_test, y_train, y_test = data_extracts.data_extract_plv((0, 30), func)
    _, x_test, _, y_test = data_extracts.data_extract_plv((30, 45), func)
    x_train, x_test = data_preparation(x_train, x_test)
    """rest_data, film_data = data_extracts.data_extract(freq, freq, (40, 45), func)
    x, x_test, x, y_test = data_split(rest_data, film_data)"""
    model = model_type(x_train, y_train)
    y_pred = model.predict(x_test)
    return accuracy_score(y_test, y_pred)

def pred_all_patients(model_type, func, freq):
    x_train, x_test, y_train, y_test = data_extracts.data_extract((0, 30), func, freq, freq)
    #_, x_test, _, y_test = data_extracts.data_extract((30, 45), func, freq, freq)
    x_train, x_test = data_preparation(x_train, x_test)
    model = model_type(x_train, y_train)
    y_pred = model.predict(x_test)
    return accuracy_score(y_test, y_pred)

def pred_all_patients_freqs(model_type, func, sample_size):
    '''
    this func evluates the model on each frequency range 
    and in addition to that it evaluates the model on all the frequency ranges (majority voting)
    '''
    
    y_pred = [] # to store the sum of all predictions (for majority voting)
    for freq in list(freq_dict.keys())[1:6]:
        print(freq)
        np.random.seed(42)
        x_train, x_test, y_train, y_test = data_extracts.data_extract((0, 30), func, sample_size)
        #_, x_test, _, y_test = data_extracts.data_extract((30, 45), func, freq, freq)
        x_train, x_test = data_preparation(x_train, x_test)
        model = model_type(x_train, y_train)
        if len(y_pred) == 0:
            y_pred = model.predict(x_test)
        else:
            y_pred += model.predict(x_test)
        print(accuracy_score(y_test, model.predict(x_test)))
    y_pred = np.where(y_pred > 2, 1, 0)
    return accuracy_score(y_test, y_pred)



def pred_single_plv(model_type, func):
    accuracy = []
    for i in range(45):
        X_train, X_test, y_train, y_test = data_extracts.data_extract((i, i), func,'plv')
        X_train, X_test = data_preparation(X_train, X_test)
        model = model_type(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
    return accuracy

def pred_single_frequency(model_type, func, frequency):
    '''
    pred for single patient and single frequency
    '''
    accuracy = []
    for i in range(45):
        X_train, X_test, y_train, y_test = data_extracts.data_extract((i, i), func, frequency)
        X_train, X_test = data_preparation(X_train, X_test)
        model = model_type(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
    return accuracy


def pred_all_frequencys(model_type, func):
    '''
    given accuracy for each patient using majority voting for all frequencies
    '''
    accuracy = []
    for i in range(45):
        y_pred = []
        for freq in list(freq_dict.keys())[1:6]:
            np.random.seed(42)
            X_train, X_test, y_train, y_test = data_extracts.data_extract((i, i), func, freq)
            X_train, X_test = data_preparation(X_train, X_test)
            model = model_type(X_train, y_train)
            if len(y_pred) == 0:
                y_pred = model.predict(X_test)
            else:
                y_pred += model.predict(X_test)
        y_pred = np.where(y_pred > 2, 1, 0)
        accuracy.append(accuracy_score(y_test, y_pred))
    return accuracy



def pred_single_state_indices(state,model_type):
    
    accuracy = []
    for i in range(45):
        y_pred = []
        for freq in list(freq_dict.keys())[1:6]:
            if state =='rest':
                rest_data,film_data =  data_extracts.data_extract(freq,freq,(i,i),data_extracts.max_indices_rest)
            if state =='film':
                rest_data,film_data = data_extracts.data_extract(freq,freq,(i,i),data_extracts.max_indices_film)
            
            X_train, X_test, y_train, y_test = data_split(rest_data, film_data)
            X_train, X_test = data_preparation(X_train, X_test)
            model = model_type(X_train, y_train)
            if len(y_pred) == 0:
                y_pred = model.predict(X_test)
            else:
                y_pred += model.predict(X_test)
                
        y_pred = np.where(y_pred > 2, 1, 0)
        accuracy.append(accuracy_score(y_test, y_pred))
    return accuracy


def pred_all_patients_freqs_2(model_type, func):
   
   # pick 8 random patients
    patients = os.listdir('rest_data')
    test_patients = np.random.choice(len(patients), 8, replace=False)
    
    
    y_pred = [] # to store the sum of all predictions (for majority voting)
    for freq in list(freq_dict.keys())[1:6]:
        print(freq)
        rest_data_train, film_data_train, rest_data_test, film_data_test = data_extracts.data_extract_2(freq, freq, test_patients, func)
 
        rest_shape_train = rest_data_train.shape[0]
        X_train = np.append(film_data_train, rest_data_train, axis=0)
        y_train = np.zeros(rest_shape_train * 2) # becase film and rest data are the same size
        y_train[:rest_shape_train] = 1
        
        #shuffle the data
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]
    
    
        X_test = np.append(film_data_test, rest_data_test, axis=0)
        rest_shape_test = rest_data_test.shape[0]
        y_test = np.zeros(rest_shape_test * 2)
        y_test[:rest_shape_test] = 1
        
        #shuffle the data
        indices = np.arange(X_test.shape[0])
        np.random.shuffle(indices)
        X_test = X_test[indices]
        y_test = y_test[indices]
        
        X_train, X_test = data_preparation(X_train, X_test)
        
        model = model_type(X_train, y_train)
        if len(y_pred) == 0:
            y_pred = model.predict(X_test)
        else:
            y_pred += model.predict(X_test)
        print(accuracy_score(y_test, model.predict(X_test)))
    y_pred = np.where(y_pred > 2, 1, 0)
    print('accuracy:',accuracy_score(y_test, y_pred))
    return accuracy_score(y_test, y_pred)
        