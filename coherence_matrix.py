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


def get_bids_path(bids_root, sub, task):
    
    # complet deal with exepcion -------------------------------------------------------------------------------------------
    
    try:
        bids_path = BIDSPath(root=bids_root, subject=sub, session=session, task=task, run=run,
                             datatype=datatype, acquisition=acquisition, suffix=suffix, extension=exten)
    except:
        bids_path = BIDSPath(root=bids_root, subject=sub, session=session, task=task, run='2',
                             datatype=datatype, acquisition=acquisition, suffix=suffix, extension=exten)
    return bids_path


def get_channels(raw):
    raw.set_eeg_reference()
    raw.notch_filter(np.arange(50,251,50))
    channels = raw.pick(picks="ecog",exclude="bads")
    return channels.get_data()



def coherence_calc(signal_1, signal_2, freq):
    f, coherence = scipy.signal.coherence(signal_1, signal_2, fs=freq, nperseg=freq/2)
    return np.mean(coherence)


# creating coherence matrix between each channel ( for each second)
def create_matrix(sec, freq, channels):
    channel_count = len(channels)
    matrix = np.empty(channel_count, channel_count)
    sec = int(sec)
    freq = int(freq)
    for row in range(channel_count):
        for col in range(row, channel_count):
            matrix[row][col] = coherence_calc(channels[row][sec*freq:(sec+1)*freq],
                                              channels[col][sec*freq:(sec+1)*freq], freq)
            matrix[col][row] = matrix[row][col]
    return matrix


def create_matrix_list(sub, task, bids_path):
    raw = mne_bids.read_raw_bids(bids_path, verbose=None)
    raw.load_data()
    channels = get_channels(raw)
    if len(channels) == 0:
        return None
    sample_rate = raw.info["sfreq"]
    time = int(len(channels[0]))/sample_rate
    time = int(time / 1)
    matrix_list = []
    for sec in range(time):#remove after testing
        print("this is the sec num: " + str(sec))
        if sec == 2:
            break
        matrix_list.append(create_matrix(sec, sample_rate, channels))
    return matrix_list


class CoherenceMatrix:
    def __init__(self, bids_root, sub_tag, task):
        self.sub = sub_tag
        self.task = task
        self.bids_root = bids_root
        self.bids_path = get_bids_path(bids_root, sub_tag, task)
        self.matrix_list = create_matrix_list(sub_tag, task, self.bids_path)

    def get_root(self):
        return self.bids_root
    
    def get_sub(self):
        return self.sub
    
    
    def get_task(self):
        return self.task
    
    
    def show_matrix(self, index):
        plt.imshow(self.matrix_list[index], cmap='viridis')
        plt.colorbar()
        plt.show()