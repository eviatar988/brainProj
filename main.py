
import os.path as op
from mne.datasets import sample
from patients_matrix import PatientsMatrix
from coherence_matrix import CoherenceMatrix
from matplotlib import pyplot as plt
import numpy as np

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


def main():
    bids_root = op.join(op.dirname(sample.data_path()), dataset)
    data = np.load(op.join(film_lists_path, '02' + '_film_matrixs.npz'))
    film_matrix = data['arr_film']
    plt.imshow(film_matrix[0], cmap='viridis')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    main()

