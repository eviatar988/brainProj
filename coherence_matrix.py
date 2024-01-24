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
        (x1[i * samples_per_subsignal:(i + 1) * samples_per_subsignal],
         y1[i * samples_per_subsignal:(i + 1) * samples_per_subsignal])
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
    bad_channels = raw.info['bads']
    print(bad_channels)
    for i in bad_channels:
        if i in channels:
            channels.remove(i)
    print(channels)
    return channels


def coherence_calc(signal_1, signal_2, freq):
    f, coherence = scipy.signal.coherence(signal_1, signal_2, fs=freq, nperseg=freq/2)
    return np.mean(coherence)


# creating coherence matrix between each channel ( for each second)
def create_matrix(sec, freq, channels, raw):
    matrix = np.empty((len(channels), len(channels)))
    signals = []
    for i in channels:
        signals.append(raw[i, sec * freq:(sec + 1) * freq][0])
    lenz = len(signals)
    for i in range(lenz):
        for j in range(i, lenz):
            matrix[i][j] = coherence_calc(signals[i], signals[j], freq)
            matrix[j][i] = matrix[i][j]
    return matrix


def create_matrix_list(sub, task, bids_path):
   
    raw = mne_bids.read_raw_bids(bids_path, verbose=None)
    channels = get_channels(raw)
    if len(channels) == 0:
        return None
    sample_rate = raw.info["sfreq"]
    x, time = raw[channels[0], :]
    time = int(time[-1])
    time = int(time / 1)
    matrix_list = []
    for i in range(time):
        print("this is the sec num: " + str(i))
        if i == 5:
            break
        matrix_list.append(create_matrix(i, sample_rate, channels, raw))
    return matrix_list


class CoherenceMatrix:
    def __init__(self, sub_tag, task, bids_root):
        self.tag = sub_tag
        self.task = task
        self.bids_root = bids_root
        bids_path = BIDSPath(root=bids_root, subject=sub_tag, session=session, task=task, run=run,
                                  datatype=datatype, acquisition=acquisition, suffix=suffix, extension=exten)
        self.matrix_list = create_matrix_list(sub_tag, task ,bids_path)

    def get_root(self):
        return self.bids_root
    
    def get_tag(self):
        return self.tag
    
    
    def get_task(self):
        return self.task
    
    
    def show_matrix(self, index):
        plt.imshow(self.matrix_list[index], cmap='viridis')
        plt.colorbar()
        plt.show()