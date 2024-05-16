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


    def patient_thread(self, patient):

        rest_matrix_list = CoherenceMatrix(self.bids_root, patient, "rest", self.freq_type)
        film_matrix_list = CoherenceMatrix(self.bids_root, patient, "film", self.freq_type)

        rest_matrix_list.create_matrix_list()
        film_matrix_list.create_matrix_list()

        if rest_matrix_list.matrix_list is not None and film_matrix_list.matrix_list is not None:
            np.savez(op.join(rest_lists_path + self.freq_type, patient + '_rest_matrixs.npz'),
                     arr_rest=np.array(rest_matrix_list.get_matrix_list(), dtype=float), allow_pickle=True)
            np.savez(op.join(film_lists_path + self.freq_type, patient + '_film_matrixs.npz'),
                     arr_film=np.array(film_matrix_list.get_matrix_list(), dtype=float), allow_pickle=True)
        else:
            print('No coherence')


    def save_matrix_to_file(self):
        """np.savez(save_filename, arr_rest=np.array(self.all_rest_matrix, dtype=object),
                 arr_film=np.array(self.all_rest_matrix, dtype=object), allow_pickle=True)"""

   # -------new code for saving the matrix's to file-------
        for patient in self.get_patients()[25:]:
            print(patient)
            p_thread = threading.Thread(target=self.patient_thread,args=(patient,))
            p_thread.start()
            p_thread.join()

    def get_rest_matrix_list(self):
        return self.all_rest_matrix

    def get_film_matrix_list(self):
        return self.all_film_matrix
