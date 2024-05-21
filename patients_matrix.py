import numpy as np
import pandas as pd
import os.path as op
from coherence_matrix import CoherenceMatrix
from tqdm import tqdm
import threading


# object hold all the matrix's of both films and rest for all patients

save_filename = 'coherence_matrixs.npz'
rest_lists_path = 'rest_lists'
film_lists_path = 'film_lists'
freq_dict = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'low_gamma': (30, 70),
    'high_gamma': (70, 250)
}

class PatientsMatrix:
    def __init__(self, bids_root, freq_type):
        self.freq_type = freq_type
        self.bids_root = bids_root
        self.all_film_matrix = []  # list of all the film matrix
        self.all_rest_matrix = []  # list of all the rest matrix
        self.patients = self.get_patients()
        '''for patient in patients:
            self.add_patient(sub=patient)'''

    """def add_patient(self, sub):  # add all the matrix's of the patient to the list
        rest_matrix_list = CoherenceMatrix(self.bids_root, sub, "rest")
        film_matrix_list = CoherenceMatrix(self.bids_root, sub, "film")

        if rest_matrix_list.matrix_list is not None:
            self.all_rest_matrix.extend(rest_matrix_list.get_matrix_list())

        if film_matrix_list.matrix_list is not None:
            self.all_film_matrix.extend(film_matrix_list.get_matrix_list())"""

    def get_patients(self): # get all the patients in the dataset
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


    def flat_matrix(self, matrix):
        flat_matrix = []
        for i in range(matrix.shape[0]):
            for j in range(i, matrix.shape[1]):
                flat_matrix.append(matrix[i, j])




    def patient_thread(self, patient):

        rest_matrix_list = CoherenceMatrix(self.bids_root, patient, "rest")
        film_matrix_list = CoherenceMatrix(self.bids_root, patient, "film")

        rest_matrix_list.create_matrix_list()
        film_matrix_list.create_matrix_list()
        if rest_matrix_list.matrix_list is not None and film_matrix_list.matrix_list is not None:
            return
        else:
            print('No coherence')
        rest_matrix_list = np.array(rest_matrix_list.get_matrix_list(),dtype=float)
        film_matrix_list = np.array(film_matrix_list.get_matrix_list(),dtype=float)
        for key in freq_dict:
            lower_Bound = freq_dict[key][0]
            upper_Bound = freq_dict[key][1]
            np.savez(op.join(rest_lists_path + key, patient + key + '_rest_matrixs.npz'),
                     arr_rest=np.mean(rest_matrix_list[:, :, lower_Bound:upper_Bound], axis=2), allow_pickle=True)
            np.savez(op.join(film_lists_path + key, patient + key + '_film_matrixs.npz'),
                     arr_rest=np.mean(rest_matrix_list[:, :, lower_Bound:upper_Bound], axis=2), allow_pickle=True)



    def save_matrix_to_file(self):
        """np.savez(save_filename, arr_rest=np.array(self.all_rest_matrix, dtype=object),
                 arr_film=np.array(self.all_rest_matrix, dtype=object), allow_pickle=True)"""

   # -------new code for saving the matrix's to file-------
        for patient in self.get_patients():
            print(patient)
            rest_matrix_list = CoherenceMatrix(self.bids_root, patient, "rest")
            film_matrix_list = CoherenceMatrix(self.bids_root, patient, "film")

            rest_matrix_list.create_matrix_list()
            film_matrix_list.create_matrix_list()
            if rest_matrix_list.matrix_list is not None or film_matrix_list.matrix_list is not None:
                rest_matrix_list = np.array(rest_matrix_list.get_matrix_list(), dtype=float)
                film_matrix_list = np.array(film_matrix_list.get_matrix_list(), dtype=float)
                for key in freq_dict:
                    lower_bound = freq_dict[key][0]
                    upper_bound = freq_dict[key][1]
                    np.savez(op.join(rest_lists_path + key, patient + '_rest_matrixs.npz'),
                             matrix_arr=np.mean(rest_matrix_list[:, :, lower_bound:upper_bound], axis=2),
                             allow_pickle=True)
                    np.savez(op.join(film_lists_path + key, patient + '_film_matrixs.npz'),
                             matrix_arr=np.mean(film_matrix_list[:, :, lower_bound:upper_bound], axis=2),
                             allow_pickle=True)
            else:
                print('No coherence')
    def get_rest_matrix_list(self):
        return self.all_rest_matrix

    def get_film_matrix_list(self):
        return self.all_film_matrix
