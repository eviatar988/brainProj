

import numpy as np
import pandas as pd
import os
import os.path as op
import matplotlib.pyplot as plt
import openneuro
from mne_bids import (BIDSPath, read_raw_bids, print_dir_tree, make_report, get_entity_vals)
import scipy
from mne.datasets import sample
import seaborn as sns
import mne
from bids_extract import bidsExtract

bids_path = r"C:\Users\eyala\Documents\GitHub\brainProj\ds003688"
dataset = "ds003688"
subject = "02"

def main():
 

    # Download one subject's data from each dataset
    bids_root = op.join(op.dirname(sample.data_path()), dataset)
    print(bids_root)
    test = bidsExtract(bids_root)
    raw = test.get_raw()
    #raw.plot() # plot the raw data
    #print(raw.info)
    raw_selection = raw["F01", 0:]
    x1=raw_selection[1]
    y1=raw_selection[0].T
    
    raw_selection = raw["F21", 0:]
    x2=raw_selection[1]
    y2=raw_selection[0].T
    
    f = y1[0]
    g = y2[0]

    
    #creat cohernce matrix between F01 AND F21
    f, Cxy = scipy.signal.coherence(f, g, fs=512, nperseg=1024)
    plt.semilogy(f, Cxy)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Coherence')
    plt.show()
    
    
    
    
    
if __name__ == '__main__':
    main()

