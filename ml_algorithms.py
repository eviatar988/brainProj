
import os.path as op
import mne_bids
from mne.datasets import sample
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from patients_matrix import PatientsMatrix
from coherence_matrix import CoherenceMatrix
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import levene, gaussian_kde
from scipy.stats import anderson
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier


def random_forest(x_train, x_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, max_depth=3)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return y_pred


def svm_classifier(x_train, x_test,y_train, y_test):
    svm_rbf = SVC(kernel='rbf', gamma='scale')
    svm_rbf.fit(x_train, y_train)
    y_pred = svm_rbf.predict(x_test)
    # Evaluate the classifier
    return y_pred