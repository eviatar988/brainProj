

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
subject = "07"

def main():
 

    # Download one subject's data from each dataset
    bids_root = op.join(op.dirname(sample.data_path()), dataset)
    test = bidsExtract(bids_root)
    raw = test.get_raw()
    all_channels = tuple(zip(raw.ch_names, raw.get_channel_types()))
    channels = [x[0] for x in all_channels if x[1] == "ecog"]
    x,time = raw[channels[0], :]
    time = time[-1]
    time = int(time)/1
    sample_rate = raw.info["sfreq"]
    print(time)

    
   
    
    #calculate the cohernce between F01 AND F21.
    
    
    
    
    
    
    
if __name__ == '__main__': # if we're running file directly and not importing it
    main()

