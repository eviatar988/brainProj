# this class is used to extract the raw data from the bids dataset

import mne_bids
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from mne_bids import (BIDSPath, read_raw_bids, print_dir_tree, make_report, get_entity_vals)
import scipy

import seaborn as sns
import mne

# define the parameters for the raw data extraction

session = 'iemu' 
datatype = 'ieeg' 
acquisition = 'clinical'
suffix = 'ieeg'
task_f = 'film'
task_r = 'rest'
sub = '05'
run = '1'
exten = '.vhdr'

# extract the raw data from the bids dataset
def extract(bids_root):
    bids_path = BIDSPath(root=bids_root, subject=sub, session=session, task=task_f, run=run,
                         datatype=datatype, acquisition=acquisition, suffix=suffix, extension=exten)
    return mne_bids.read_raw_bids(bids_path)




# this class is used to extract the raw data from the bids dataset
class bidsExtract:
    def __init__(self, bids_root):
        self.bids_root = bids_root
        self.raw_film = extract(bids_root)

    def get_raw(self):
        return self.raw_film
