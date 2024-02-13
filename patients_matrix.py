import mne_bids
import numpy as np
import pandas as pd
import os
import os.path as op
import pickle
import matplotlib.pyplot as plt
from coherence_matrix import CoherenceMatrix
from mne_bids import (BIDSPath, read_raw_bids, print_dir_tree, make_report, get_entity_vals)
import scipy
import mne


# object hold all the matrix's of both films and rest for all patients

save_filename = 'coherence_matrixs.npz'


class PatientsMatrix:
    def __init__(self, bids_root):
        self.bids_root = bids_root
        self.all_film_matrix = []  # list of all the film matrix
        self.all_rest_matrix = []  # list of all the rest matrix
        patients = self.get_patients()
        for patient in patients:
            if patient == "04":# for testing
                break
            print("patient: ".join(patient))
            self.add_patient(sub=patient)

    def add_patient(self, sub):
        rest_matrix_list = CoherenceMatrix(self.bids_root, sub, "rest")
        film_matrix_list = CoherenceMatrix(self.bids_root, sub, "film")

        if rest_matrix_list.matrix_list is not None:
            self.all_rest_matrix.extend(rest_matrix_list.get_matrix_list())

        if film_matrix_list.matrix_list is not None:
            self.all_film_matrix.extend(film_matrix_list.get_matrix_list())

    def get_patients(self):
        patients_list = []

        participants_path = op.join(self.bids_root, 'participants.tsv')

        # read participants.tsv file
        participants = pd.read_csv(participants_path, sep='\t')
        print(len(participants))
        # creat array of strings like 01, 02, 03, 04, 05, 06, 07, 08, 09, 10...
        for i in range(1, len(participants) + 1):
            if i < 10:
                patients_list.append('0' + str(i))
            else:
                patients_list.append(str(i))
        return patients_list
        # print(patients_list)8 j

    def save_matrix_to_file(self):
        np.savez(save_filename, arr_rest=np.array(self.all_rest_matrix, dtype=object),
                 arr_film=np.array(self.all_rest_matrix, dtype=object), allow_pickle=True)

    def get_rest_matrix_list(self):
        return self.all_rest_matrix

    def get_film_matrix_list(self):
        return self.all_film_matrix
