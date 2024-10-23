# this class is used to extract the raw data from the bids dataset
import io
import sys

import mne_bids
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mne_bids import (BIDSPath, read_raw_bids, print_dir_tree, make_report, get_entity_vals)
import scipy
import seaborn as sns
import mne
import os.path as op
from mne.datasets import sample
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

#return the path to the dataset:
def get_bidsroot(dataset):
    return op.join(op.dirname(sample.data_path()), dataset)



def find_run(directory_path, run_path):
    for filename in os.listdir(directory_path):
        if filename.startswith(run_path):
            return filename[filename.find('run') + 4]
    return None


# creating bidspath
def get_bids_path(bids_root, sub, task):
    iemu_path = op.join(bids_root, 'sub-' + sub, 'ses-iemu', "ieeg")
    run_path = "".join(('sub-' + sub, '_ses-iemu_task-', task))
    if op.isdir(iemu_path):  # if this directory dosent exist their wont be any ecog data
        run_num = find_run(iemu_path, run_path)
        if run_num is not None:  # if the patient dosent have any runs
            bids_path = BIDSPath(root=bids_root, subject=sub, session=session, task=task, run=run_num,
                                 datatype=datatype, acquisition=acquisition, suffix=suffix)
            return bids_path
        else:
            return None
    else:
        return None


def handle_channels(bids_rest, bids_film):
    if bids_rest is None or bids_film is None:
        return None, None, None, None

    dummy_output = io.StringIO()  # preventing the print from the bids
    # Save the current stdout
    original_stdout = sys.stdout
    # Redirect stdout to the dummy object
    sys.stdout = dummy_output
    # Now call the function that produces output
    raw_rest = read_raw_bids(bids_rest)
    raw_film = read_raw_bids(bids_film)
    bad_channels = set(raw_rest.info['bads']).union(raw_film.info['bads'])
    bad_channels = list(bad_channels)
    # Restore the original stdout
    sys.stdout = original_stdout

    # preprocessing
    raw_rest.load_data()
    for channel in bad_channels:  # removing bad channels
        if channel in raw_rest.ch_names:
            raw_rest.drop_channels(channel)
        if channel in raw_film.ch_names:
            raw_film.drop_channels(channel)

    raw_rest.notch_filter(np.arange(50, 251, 50))
    raw_rest.set_eeg_reference()
    sfreq_rest = raw_rest.info['sfreq']
    raw_film.load_data()
    raw_film.notch_filter(np.arange(50, 251, 50))
    raw_film.set_eeg_reference()
    sfreq_film = raw_film.info['sfreq']
    if 'ecog' in raw_rest.get_channel_types():
        return (raw_rest.pick(picks=['ecog'], exclude='bads').get_data(),
                raw_film.pick(picks=['ecog'], exclude='bads').get_data(), int(sfreq_rest), int(sfreq_film))
    else:
        return None, None, None, None
