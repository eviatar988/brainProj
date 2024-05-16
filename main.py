
import os.path as op
from mne.datasets import sample
from patients_matrix import PatientsMatrix
from coherence_matrix import CoherenceMatrix
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import levene, gaussian_kde
from scipy.stats import anderson

bids_path = r"C:\Users\eyala\Documents\GitHub\brainProj\ds003688"
dataset = "ds003688"
subject = "07"
session = 'iemu'
datatype = 'ieeg'
acquisition = 'clinical'
suffix = 'ieeg'
run = '1'
rest_lists_path = 'rest_lists'
film_lists_path = 'film_lists'
freq_dict = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'low gamma': (30, 70),
    'high gamma': (70, 250)
}

def show_me_matrix(matrix,name):
    plt.imshow(matrix, cmap='viridis', )
    plt.title(name)
    plt.colorbar()
    plt.show()

def main():

    bids_root = op.join(op.dirname(sample.data_path()), dataset)
    patient_m = PatientsMatrix(bids_root, "delta")
    patient_m.save_matrix_to_file()


if __name__ == '__main__':
    main()

