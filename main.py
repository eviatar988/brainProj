
from scipy.signal import medfilt
import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
import openneuro
from mne_bids import (BIDSPath, read_raw_bids, print_dir_tree, make_report, get_entity_vals)
import scipy
from scipy.signal import csd, coherence
import matplotlib.pyplot as plt

from mne.datasets import sample
import seaborn as sns
import mne
from bids_extract import bidsExtract
from coherence_matrix import CoherenceMatrix

bids_path = r"C:\Users\eyala\Documents\GitHub\brainProj\ds003688"
dataset = "ds003688"
subject = "07"
session = 'iemu'
datatype = 'ieeg'
acquisition = 'clinical'
suffix = 'ieeg'
run = '1'




def main():


    # Download one subject's data from each dataset
    bids_root = op.join(op.dirname(sample.data_path()), dataset)
    print(bids_root)

    path = BIDSPath(root=bids_root, subject="10", session=session, task='rest', run=run,
                    datatype=datatype, acquisition=acquisition, suffix=suffix)
    raw = read_raw_bids(path, verbose=None)
    raw.load_data()
    raw.set_eeg_reference()
    raw.notch_filter(np.arange(50,251,50))
    raw.compute_psd().plot()
    psd_try = raw.compute_psd()
    signal,time = psd_try.get
    
   
    #calculate the cohernce between F01 AND F21.
    
    
if __name__ == '__main__': # if we're running file directly and not importing it
    main()

