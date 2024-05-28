import sys
import warnings
import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
import io
from mne_bids import (BIDSPath, read_raw_bids, print_dir_tree, make_report, get_entity_vals)
import scipy
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from tqdm import tqdm
import typing
import data_extracts

X = data_extracts.data_trasnform('delta', 'delta')
y = np.zeros(X[0].shape[0]+X[1].shape[0])
y[:X[1].shape[0]] = 1
X = np.append(X[1], X[0], axis=0)

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(y_train)


svm = svm_rbf = SVC(kernel='rbf', gamma='scale', C=1)
svm.fit(X_train, y_train)

scores_rbf = cross_val_score(svm_rbf, X, y, cv=5)
print("RBF Kernel Accuracy: ", np.mean(scores_rbf))