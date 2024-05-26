import math
import os.path as op
import os
from mne.datasets import sample
from patients_matrix import PatientsMatrix
from coherence_matrix import CoherenceMatrix
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import levene, gaussian_kde
from scipy.stats import anderson
import scipy.stats as stats
import data_extracts
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load your data (replace with your data loading method)
X = data_extracts.data_trasnform('delta', 'delta')
y = np.zeros(X[0].shape[0]+X[1].shape[0])
y[:X[1].shape[0]] = 1
X = np.append(X[1], X[0], axis=0)
print(X.shape)

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Feature scaling (optional, but recommended)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the KNN model
knn = KNeighborsClassifier(n_neighbors=150, metric='euclidean')  # Adjust k and metric as needed

# Train the KNN model
knn.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test_scaled)

# Evaluate the model performance (replace with desired metrics)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")