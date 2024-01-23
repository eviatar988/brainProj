
import mne_bids
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from mne_bids import (BIDSPath, read_raw_bids, print_dir_tree, make_report, get_entity_vals)
import scipy

import seaborn as sns
import mne

session = 'iemu'
datatype = 'ieeg'
acquisition = 'clinical'
suffix = 'ieeg'
run = '1'
exten = '.vhdr'




def split_signal(signal, duration=1):
    x1, y1 = signal
    sample_rate = 1 / (x1[1] - x1[0])  # Assuming uniform sampling rate
    print(sample_rate)
    # Calculate the number of samples for the specified duration
    samples_per_subsignal = int(duration * sample_rate)

    # Calculate the total number of full sub-signals
    num_full_subsignals = len(x1) // samples_per_subsignal

    # Split the signal into sub-signals
    subsignals = [
        (x1[i*samples_per_subsignal:(i+1)*samples_per_subsignal],
         y1[i*samples_per_subsignal:(i+1)*samples_per_subsignal])
        for i in range(num_full_subsignals)
    ]

    return subsignals



def get_raw(bids_root, sub, task):
    bids_path = BIDSPath(root=bids_root, subject=sub, session=session, task=task, run=run,
                         datatype=datatype, acquisition=acquisition, suffix=suffix, extension=exten)
    return mne_bids.read_raw_bids(bids_path)



def get_channels(raw):
    all_channels = tuple(zip(raw.ch_names, raw.get_channel_types()))
    channels = [x[0] for x in all_channels if x[1] == "ecog"]
    if len(channels) == 0:
        return []
    bad_channels = raw.info["bads"]
    for i in bad_channels:
        if i in channels: channels.remove(i)
    return channels


def create_matrix(sec,freq,channels,raw):
    signals = []
    for i in channels:
        signals.append(raw[i, sec*freq:(sec+1)*freq][0])


def create_matrix_list(sub, task, bids_root):
    bids_path = BIDSPath(root=bids_root, subject=sub, session=session, task=task, run=run,
                         datatype=datatype, acquisition=acquisition, suffix=suffix, extension=exten)
    raw = mne_bids.read_raw_bids(bids_path, verbose=None)
    channels = get_channels(raw)
    if len(channels) == 0:
        return None
    sample_rate = raw.info["sfreq"]
    x, time = raw[channels[0], :]
    time = int(time[-1])
    time = int(time / 1)
    matrixs = []
    for i in range(time):
        create_matrix(i,sample_rate,channels,raw)
    return matrixs





class CoherenceMatrix:
    def __init__(self, sub_tag, task, bids_root):
        self.matrix_list = create_matrix_list(sub_tag, task)

