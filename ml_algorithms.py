
import os.path as op
import mne_bids
from mne.datasets import sample
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import data_extracts
from patients_matrix import PatientsMatrix
import coherence_matrix
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import levene, gaussian_kde
from scipy.stats import anderson
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier


"""
class CustomCNNModel(Model):
    def __init__(self):
        super(CustomCNNModel, self).__init__()
        # Define layers in the constructor
        self.conv1 = layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(30, 30, 1))
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation="relu")
        self.dropout = layers.Dropout(0.2)
        self.dense2 = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        # Define the forward pass
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        if training:
            x = self.dropout(x, training=training)
        return self.dense2(x)


def cnn(x_train, x_test, y_train, y_test):
    x_train, x_test = x_train*256, x_test*256
    model = CustomCNNModel()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))
    y_pred = model.predict(x_test)
    y_pred = np.resize(y_pred, y_pred.shape[0])
    y_pred = np.where(y_pred > 0.5, 1, 0)
    return y_pred
"""


def random_forest(x_train, y_train):
    # create random forest model and train in on the data, return the trained model
    # x_train : nparray ,array that contains all the data
    # y_train : int[] ,array that contains the labels of the data
    model = RandomForestClassifier(n_estimators=100, max_depth=2)
    model.fit(x_train, y_train)
    return model


def svm_classifier(x_train, y_train):
    # create random forest model and train in on the data , return the trained model
    # x_train : nparray ,array that contains all the data
    # y_train : int[] ,array that contains the labels of the data
    model = SVC(kernel='rbf', gamma='scale', random_state=42, C=5, probability=True)
    model.fit(x_train, y_train)
    return model

