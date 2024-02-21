
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



def main():
    bids_root = op.join(op.dirname(sample.data_path()), dataset)
    p1 = PatientsMatrix(bids_root)
    p1.save_matrix_to_file()


if __name__ == '__main__':
    main()

