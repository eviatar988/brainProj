
import mne_bids
import numpy as np
import pandas as pd
import os
import os.path as op

import matplotlib.pyplot as plt
from coherence_matrix import CoherenceMatrix
from mne_bids import (BIDSPath, read_raw_bids, print_dir_tree, make_report, get_entity_vals)
import scipy
import pandas as pd
import mne


class PatientsMatrix:
    def __init__(self, bids_root ):
        self.bids_root = bids_root
        self.all_film_matrix =[] #list of all the film matrix 
        self.all_rest_matrix= [] #list of all the rest matrix
    
    
    def add_patient(self, sub):
        rest_matrix_list = CoherenceMatrix(self.bids_path,sub, "rest")  
        film_matrix_list = CoherenceMatrix(self.bids_path,sub, "film")
        
        if rest_matrix_list is not None:
            self.all_rest_matrix.append(rest_matrix_list)
        
        if film_matrix_list is not None:
            self.all_film_matrix.append(film_matrix_list)
            
    
    def get_patients(self):
        patients_list = []
        
        participants_path = op.join(self.bids_root, 'participants.tsv')
        
        # read participants.tsv file
        participants = pd.read_csv(participants_path, sep='\t')
        print(len(participants))
        
        #creat array of strings like 01, 02, 03, 04, 05, 06, 07, 08, 09, 10...
        for i in range(1, len(participants)+1):
            if i < 10:
                patients_list.append('0'+str(i))
            else:
                patients_list.append(str(i))
                
        #print(patients_list)
    
    # TODO: complete the last function...
    # func that for every patient in the list, apply the function add_patient
    #fill the all_film_matrix and all_rest_matrix
    def fill_all_matrix(self): 
        for patient in self.patients_list:
            self.add_patient(patient)
    
             
        
       
        
    

