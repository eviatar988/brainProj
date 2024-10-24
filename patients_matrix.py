import multiprocessing

import numpy as np
import pandas as pd
import os
import os.path as op

import bids_extract
import connectivity_matrix
from tqdm import tqdm
import threading


# object hold all the matrix's of both films and rest for all patients
signal_len = 3
save_filename = 'coherence_matrixs.npz'
rest_data_path = 'rest_data'
film_data_path = 'film_data'
freq_dict = {
    'delta': (1, 3),
    'theta': (3, 5),
    'alpha': (5, 7),
    'beta': (7, 16),
    'low_gamma': (16, 36),
    'high_gamma': (36, 126)
}

class PatientsMatrix:
    # class which is used to create matrices for all patients in our data and save in a compressed file.
    #
    def __init__(self, dataset, sec_per_sample=1):
        self.sec_per_sample = int(sec_per_sample)
        self.bids_root = bids_extract.get_bidsroot(dataset)
        self.all_film_matrix = []  # list of all the film matrix
        self.all_rest_matrix = []  # list of all the rest matrix
        self.patients = self.get_patients()

    def get_patients(self):# get all the patients in the dataset
        patients_list = []
        participants_path = op.join(self.bids_root, 'participants.tsv')

        # read participants.tsv file
        participants = pd.read_csv(participants_path, sep='\t')
        # creat array of strings like 01, 02, 03, 04, 05, 06, 07, 08, 09, 10...
        for i in range(1, len(participants) + 1):
            if i < 10:
                patients_list.append('0' + str(i))
            else:
                patients_list.append(str(i))
        return patients_list

    def patient_thread_coherence(self, patient):
        if not os.path.isdir(op.join(rest_data_path)):
            os.mkdir(op.join(rest_data_path))
        if not os.path.isdir(op.join(film_data_path)):
            os.mkdir(op.join(film_data_path))
        task = connectivity_matrix.coherence_calc
        rest_matrix_list, film_matrix_list = connectivity_matrix.create_matrix_list(self.bids_root, patient,
                                                                                 self.sec_per_sample)

        if rest_matrix_list is None or film_matrix_list is None:
            print('No ecog samples for this patient')
            return

        rest_matrix_list = np.array(rest_matrix_list, dtype=float)
        film_matrix_list = np.array(film_matrix_list, dtype=float)
        print(rest_matrix_list.shape, film_matrix_list.shape)
        for key in freq_dict:
            rest_dir_path = op.join(rest_data_path, f'patient={patient}')
            film_dir_path = op.join(film_data_path, f'patient={patient}')

            if not os.path.isdir(rest_dir_path):
                os.mkdir(rest_dir_path)
            if not os.path.isdir(film_dir_path):
                os.mkdir(film_dir_path)
            lower_Bound = freq_dict[key][0]
            upper_Bound = freq_dict[key][1]

            np.savez(op.join(rest_dir_path, f'patient={patient},task=rest,freq={key},sec={signal_len}.npz'),
                     matrix_arr=np.mean(rest_matrix_list[:, :, lower_Bound:upper_Bound], axis=2), allow_pickle=True)
            np.savez(op.join(film_dir_path, f'patient={patient},task=film,freq={key},sec={signal_len}.npz'),
                     matrix_arr=np.mean(film_matrix_list[:, :, lower_Bound:upper_Bound], axis=2), allow_pickle=True)
            print(patient,' complete')

    def patient_thread_plv(self, patient):
        if not os.path.isdir(op.join(rest_data_path)):
            os.mkdir(op.join(rest_data_path))
        if not os.path.isdir(op.join(film_data_path)):
            os.mkdir(op.join(film_data_path))
        rest_matrix_list, film_matrix_list = connectivity_matrix.create_matrix_list(self.bids_root, patient,
                                                                                 self.sec_per_sample,
                                                                                task=connectivity_matrix.calculate_plv)
        if rest_matrix_list is None or film_matrix_list is None:
            print('No ecog samples for this patient')
            return
        rest_matrix_list = np.array(rest_matrix_list, dtype=float)
        film_matrix_list = np.array(film_matrix_list, dtype=float)
        rest_dir_path = op.join(rest_data_path, f'patient={patient}')
        film_dir_path = op.join(film_data_path, f'patient={patient}')
        if not os.path.isdir(rest_dir_path):
            os.mkdir(rest_dir_path)
        if not os.path.isdir(film_dir_path):
            os.mkdir(film_dir_path)
        np.savez(op.join(rest_dir_path, f'patient={patient},task=rest,plv,sec={self.sec_per_sample}.npz'),
                 matrix_arr=rest_matrix_list, allow_pickle=True)
        np.savez(op.join(film_dir_path, f'patient={patient},task=film,plv,sec={self.sec_per_sample}.npz'),
                 matrix_arr=film_matrix_list, allow_pickle=True)
        print(patient, ' complete')

    def save_matrix_to_file(self, measurement='Coherence'):

        proc_arr = []
        if measurement == 'PLV':
            for i in self.get_patients():
                self.patient_thread_plv(i)
        if measurement == 'Coherence':
            for i in self.get_patients():
                self.patient_thread_coherence(i)


