# %%
import math
import os.path as op
import os
from mne.datasets import sample
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from patients_matrix import PatientsMatrix
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import levene, gaussian_kde
from scipy.stats import anderson
import scipy.stats as stats
import cv2

# %%
rest_lists_path = 'rest_lists'
film_lists_path = 'film_lists'


def load_data(path, file):
    '''
    Load the data from the file
    return rest_data, film_data arrys for single patient
    each array is a 2D array with shape (time, flat_matrix)
    '''

    loaded_file = np.load(op.join(path, file))
    return loaded_file['matrix_arr']

def raw_data(rest_path, film_path, rest_file, film_file):
    rest_data, film_data = data_fit(rest_path, film_path, rest_file, film_file)
    return data_split(rest_data, film_data)
# 
def data_fit(rest_path, film_path, rest_file, film_file):
    '''
    reshaping the film data to fit the rest data.
    taking every second row matrix from the film data
    cutting the film data to the same size as the rest data
    '''
    rest_data = load_data(rest_path, rest_file)
    rest_shape = rest_data.shape[0]
    film_data = load_data(film_path, film_file)
    film_shape = film_data.shape[0]
    film_data = film_data[range(0, film_shape, 2)]
    return rest_data, film_data[:rest_shape]


def normalize_vector(vector):
    """
    Function to normalize a single vector.
    """
    norm = np.linalg.norm(vector)  # Calculate the magnitude of the vector
    if norm == 0:
        return vector  # Return the vector itself if the norm is 0 to avoid division by zero
    return vector / norm  # Divide each element by the magnitude


def normalize_array_of_vectors(array_of_vectors):
    """
    Function to normalize an array of vectors.
    """
    return np.array([normalize_vector(v) for v in array_of_vectors])


def max_values(rest_data, film_data):
    '''
    return the 100 max values for each second in the rest and film data
    '''
    rest_indices = np.sort(np.argsort(rest_data, axis=1)[:, -50:], axis=1)
    film_indices = np.sort(np.argsort(film_data, axis=1)[:, -50:], axis=1)
    rest_temp = np.zeros((rest_data.shape[0], 50))
    film_temp = np.zeros((rest_data.shape[0], 50))
    for i in range(rest_data.shape[0]):
        rest_temp[i, :] = rest_data[i, rest_indices[i]]
        film_temp[i, :] = film_data[i, film_indices[i]]
    X_train, X_test, y_train, y_test = data_split(rest_temp, film_temp)
    return X_train, X_test, y_train, y_test


def max_indices_mean(rest_data, film_data):
    '''
    create a mean matrix from rest and film data for single patient
    choose the 100 max indices from the mean matrix
    return arrys with the values in those indices for each second
    '''
    X_train, X_test, y_train, y_test = data_split(rest_data, film_data)
    rest_data = X_train[np.where(y_train == 0)[0]]
    film_data = X_train[np.where(y_train == 1)[0]]
    rest_mean = np.mean(rest_data, axis=0)
    film_mean = np.mean(film_data, axis=0)
    mean_matrix = np.add(rest_mean, film_mean) / 2
    indices = np.argsort(mean_matrix)[-100:]
    return X_train[:, indices], X_test[:, indices], y_train, y_test

def max_values_test(rest_data, film_data):
    '''
    return the 100 max values for each second in the rest and film data
    '''
    rest_indices = np.sort(np.argsort(rest_data, axis=1)[:, -50:], axis=1)
    film_indices = np.sort(np.argsort(film_data, axis=1)[:, -50:], axis=1)
    rest_temp = np.zeros((rest_data.shape[0], 100))
    film_temp = np.zeros((rest_data.shape[0], 100))
    for i in range(rest_data.shape[0]):
        rest_temp[i, :] = np.concatenate((rest_data[i, rest_indices[i]], rest_data[i, rest_indices[i]]))
        film_temp[i, :] = np.concatenate((film_data[i, film_indices[i]], film_data[i, film_indices[i]]))
    X_train, X_test, y_train, y_test = data_split(rest_temp, film_temp)
    return X_train, X_test, y_train, y_test

# returning the indices with the most difference between film and rest
def max_func_indices(rest_data, film_data):
    '''
    create a mean matrix from rest and film data for single patient
    choose the 100 indices with the most difference between the rest and film data
    return arrys with the values in those indices for each second
    '''
    X_train, X_test, y_train, y_test = data_split(rest_data, film_data)
    rest_data = X_train[np.where(y_train == 0)[0]]
    film_data = X_train[np.where(y_train == 1)[0]]
    rest_mean = np.mean(rest_data, axis=0)
    film_mean = np.mean(film_data, axis=0)
    max_array = [max(a, b) for a, b in zip(rest_mean, film_mean)]
    max_array = np.array(max_array)
    indices = np.argsort(max_array)[-100:]
    indices = sorted(indices)
    X_train = X_train[:, indices]
    X_test = X_test[:, indices]
    """X_train = normalize_array_of_vectors(X_train[:, indices])
    X_test = normalize_array_of_vectors(X_test[:, indices])"""
    return X_train, X_test, y_train, y_test


# calculate the max indices of the mean of rest and film data, return the data in those indices
def max_indices(rest_data, film_data):
    '''
    creat mean matrix for each rest and film data for single patient 
    find the 50 max indices in each mean matrix (50 from rest and 50 from film)
    return arrys with the values in those indices for each second
    '''
    X_train, X_test, y_train, y_test = data_split(rest_data, film_data)
    rest_data = X_train[np.where(y_train == 0)[0]]
    film_data = X_train[np.where(y_train == 1)[0]]
    rest_mean = np.mean(rest_data, axis=0)
    rest_indices = np.argsort(rest_mean)[-100:]
    film_mean = np.mean(film_data, axis=0)
    film_indices = np.argsort(film_mean)[-100:]
    common_elements = np.intersect1d(rest_indices[-50:], film_indices[-50:])
    while len(common_elements) > 0:
        for element in common_elements:
            if rest_mean[element] > film_mean[element]:
                film_indices = film_indices[film_indices != element]
            else:
                rest_indices = rest_indices[rest_indices != element]
        common_elements = np.intersect1d(rest_indices[-50:], film_indices[-50:])
    rest_indices = np.sort(rest_indices[-50:])
    film_indices = np.sort(film_indices[-50:])
    indices = np.concatenate((rest_indices, film_indices))
    X_temp = X_train[:, indices]
    # np.random.shuffle(indices)
    X_train = X_train[:, indices]
    X_test = X_test[:, indices]
    return X_train, X_test, y_train, y_test


"""
def matrix_transform(matrix_flat):
    
    '''
    create a 2D matrix from the flat matrix
    '''
    data_len = matrix_flat.shape[0]
    matrix_len = int(math.sqrt(2 * matrix_flat.shape[1] + 1 / 4) + 1 / 2)
    matrix = np.zeros(shape=(data_len, matrix_len, matrix_len), dtype=float)
    for sec in range(data_len):
        index = 0
        for i in range(matrix_len):
            for j in range(i, matrix_len):
                if i == j:
                    matrix[sec, i, j] = 1
                else:
                    matrix[sec, i, j] = matrix_flat[sec, index]
                    matrix[sec, j, i] = matrix[sec, i, j]
                    index += 1
    return matrix


def matrix_resize(matrix):
    '''
    resize the matrix to 30x30
    '''
    resized_matrix = np.zeros(shape=(matrix.shape[0], 30, 30))
    for i in range(matrix.shape[0]):
        resized_matrix[i, :, :] = cv2.resize(matrix[i, :, :], (30, 30), interpolation=cv2.INTER_AREA)
    return resized_matrix


def matrix_fit(rest_path, film_path, rest_file, film_file):
    '''
    reshape the rest and film data to gray scale 30x30x1 matrix
    
    '''
    rest_data, film_data = data_fit(rest_path, film_path, rest_file, film_file)
    rest_data = matrix_transform(rest_data)
    rest_data = matrix_resize(rest_data)
    rest_data = np.reshape(rest_data, (rest_data.shape[0], rest_data.shape[1], rest_data.shape[2], 1))

    film_data = matrix_transform(film_data)
    film_data = matrix_resize(film_data)
    film_data = np.reshape(film_data, (film_data.shape[0], film_data.shape[1], film_data.shape[2], 1))
    return rest_data, film_data"""


def max_indices_film(rest_data, film_data):
    '''
    return data only in film indices (for rest and film)
    '''
    X_train, X_test, y_train, y_test = data_split(rest_data, film_data)
    film_data = X_train[np.where(y_train == 1)[0]]
    film_mean = np.mean(film_data, axis=0)
    film_indices = np.sort(np.argsort(film_mean)[-100:])
    return X_train[:, film_indices], X_test[:, film_indices], y_train, y_test


def max_indices_rest(rest_data, film_data):
    '''
    creat mean matrix for each rest and film data for single patient 
    find the 50 max indices in each mean matrix (50 from rest and 50 from film)
    return arrys with the values in those indices for each second
    '''
    X_train, X_test, y_train, y_test = data_split(rest_data, film_data)
    rest_data = X_train[np.where(y_train == 0)[0]]
    rest_mean = np.mean(rest_data, axis=0)
    rest_indices = np.sort(np.argsort(rest_mean)[-100:])
    return X_train[:, rest_indices], X_test[:, rest_indices], y_train, y_test


def data_split(rest_data, film_data):
    rest_shape = rest_data.shape[0]
    X = np.append(film_data, rest_data, axis=0)
    y = np.zeros(rest_shape * 2)  # becase film and rest data are the same size
    y[:rest_shape] = 1  # 1 for film, 0 for rest, it is more accurate to use write 'film shape' instead of 'rest shape' but it is the same
    # Split data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def data_extract(bounds, extract_func, data_type, sample_size):
    '''
    bound = patients 
    
    input: arry of patients indecies
    ouput: rest and film data for all patients in the bound 
    
    '''
    rest_path = 'rest_data'
    patients = os.listdir(rest_path)
    film_path = 'film_data'

    first_p = bounds[0]

    rest_dict = op.join(rest_path, patients[first_p])  # rest path
    film_dict = op.join(film_path, patients[first_p])  # film path

    rest_file = f'{patients[first_p]},task=rest,type={data_type},sec={sample_size}.npz'
    film_file = f'{patients[first_p]},task=film,type={data_type},sec={sample_size}.npz'
    rest_data, film_data = data_fit(rest_dict, film_dict, rest_file, film_file)
    x_train, x_test, y_train, y_test = extract_func(rest_data, film_data)

    for i in bounds[1:]:
        rest_dict = op.join(rest_path, patients[i])
        film_dict = op.join(film_path, patients[i])

        rest_file = f'{patients[i]},task=rest,type={data_type},sec={sample_size}.npz'
        film_file = f'{patients[i]},task=film,type={data_type},sec={sample_size}.npz'
        rest_data, film_data = data_fit(rest_dict, film_dict, rest_file, film_file)
        x_train_t, x_test_t, y_train_t, y_test_t = extract_func(rest_data, film_data)

        x_train = np.append(x_train, x_train_t, axis=0)
        x_test = np.append(x_test, x_test_t, axis=0)
        y_train = np.append(y_train, y_train_t, axis=0)
        y_test = np.append(y_test, y_test_t, axis=0)
    indices = np.random.permutation(len(x_train))
    x_train = x_train[indices]
    y_train = y_train[indices]
    return x_train, x_test, y_train, y_test


def data_extract_plv(bounds, extract_func, freq_type_rest=None, freq_type_film=None):
    '''
    bound = patients

    input: arry of patients indecies
    ouput: rest and film data for all patients in the bound

    '''
    rest_path = 'rest_data'
    patients = os.listdir(rest_path)
    film_path = 'film_data'

    first_p = bounds[0]
    last_p = bounds[1]

    rest_dict = op.join(rest_path, patients[first_p])  # rest path
    film_dict = op.join(film_path, patients[first_p])  # film path

    rest_file = f'{patients[first_p]},task=rest,plv,sec=3.npz'
    film_file = f'{patients[first_p]},task=film,plv,sec=3.npz'

    X_train, X_test, y_train, y_test = extract_func(rest_dict, film_dict, rest_file, film_file)

    for i in range(first_p + 1, last_p):
        rest_dict = op.join(rest_path, patients[i])
        film_dict = op.join(film_path, patients[i])

        rest_file = f'{patients[i]},task=rest,plv,sec=3.npz'
        film_file = f'{patients[i]},task=film,plv,sec=3.npz'
        X_train_t, X_test_t, y_train_t, y_test_t = extract_func(rest_dict, film_dict, rest_file, film_file)

        X_train = np.append(X_train, X_train_t, axis=0)
        X_test = np.append(X_test, X_test_t, axis=0)
        y_train = np.append(y_train, y_train_t, axis=0)
        y_test = np.append(y_test, y_test_t, axis=0)

    return X_train, X_test, y_train, y_test


def data_extract_2(freq_type_rest, freq_type_film, test_patients, extract_func):
    '''
    data extract for all patients except the test patients
    
    '''
    rest_path = 'rest_data'
    patients = os.listdir(rest_path)
    film_path = 'film_data'

    rest_dict = op.join(rest_path, patients[0])  # rest path
    film_dict = op.join(film_path, patients[0])  # film path

    rest_file = f'{patients[0]},task=rest,freq={freq_type_rest}.npz'
    film_file = f'{patients[0]},task=film,freq={freq_type_film}.npz'

    rest_data, film_data = extract_func(rest_dict, film_dict, rest_file, film_file)

    for i in range(1, 44):
        if i in test_patients:
            continue
        rest_dict = op.join(rest_path, patients[i])
        film_dict = op.join(film_path, patients[i])

        rest_file = f'{patients[i]},task=rest,freq={freq_type_rest}.npz'
        film_file = f'{patients[i]},task=film,freq={freq_type_film}.npz'
        temp_rest, temp_film = extract_func(rest_dict, film_dict, rest_file, film_file)

        rest_data_train = np.append(rest_data, temp_rest, axis=0)
        film_data_train = np.append(film_data, temp_film, axis=0)

    for i in test_patients:
        rest_dict = op.join(rest_path, patients[i])
        film_dict = op.join(film_path, patients[i])

        rest_file = f'{patients[i]},task=rest,freq={freq_type_rest}.npz'
        film_file = f'{patients[i]},task=film,freq={freq_type_film}.npz'
        rest_data_test, film_data_test = extract_func(rest_dict, film_dict, rest_file, film_file)

        rest_data_test = np.append(rest_data_test, temp_rest, axis=0)
        film_data_test = np.append(film_data_test, temp_film, axis=0)

    return rest_data_train, film_data_train, rest_data_test, film_data_test

