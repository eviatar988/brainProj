import sys
import warnings
import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
import io
from mne_bids import (BIDSPath, read_raw_bids, print_dir_tree, make_report, get_entity_vals)
import scipy
from tqdm import tqdm

# Ignore RuntimeWarnings

freq_dict = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'low_gamma': (30, 70),
    'high_ gamma': (70, 250)
}
session = 'iemu'
datatype = 'ieeg'
acquisition = 'clinical'
suffix = 'ieeg'
run = '1'
exten = '.vhdr'

"""
# find the number of run for a patient task, find the file and take the run number from its name
def find_run(directory_path, run_path):
    for filename in os.listdir(directory_path):
        if filename.startswith(run_path):
            return filename[filename.find('run')+4]
    return None


def get_bids_path(bids_root, sub, task):
    iemu_path = op.join(bids_root, 'sub-'+sub, 'ses-iemu',"ieeg")
    run_path = "".join(('sub-'+sub, '_ses-iemu_task-', task))
    if op.isdir(iemu_path):
        run_num = find_run(iemu_path, run_path)
        if run_num is not None:
            bids_path = BIDSPath(root=bids_root, subject=sub, session=session, task=task, run=run_num,
                        datatype=datatype, acquisition=acquisition, suffix=suffix)
            return bids_path
        else:
            return None
    else:
        return None



def get_channels(raw):
    raw.load_data()
    raw.set_eeg_reference()
    raw.notch_filter(np.arange(50, 251, 50))
    if 'ecog' in raw.get_channel_types():
        channels = raw.pick(picks=['ecog'], exclude='bads')
        return channels.get_data()
    else:
        return None

# channels = raw.pick(picks='ecog' , exclude='bads')


def coherence_calc(signal_1, signal_2, freq, freq_type):
    bounds = freq_dict.get(freq_type)
    f, coherence = scipy.signal.coherence(signal_1, signal_2, fs=freq, nperseg=freq / 2)
    return np.mean(coherence)


# creating coherence matrix between each channel ( for each second)
def create_matrix(freq, channels, freq_type):
    channel_count = len(channels)
    matrix = np.empty((channel_count, channel_count))
    freq = int(freq)
    for row in (range(channel_count)):
        for col in range(row, channel_count):
            matrix[row, col] = coherence_calc(channels[row], channels[col], freq, freq_type)
            matrix[col, row] = matrix[row, col]
    return matrix


def raw_handler(bids_path):
    dummy_output = io.StringIO()
    # Save the current stdout
    original_stdout = sys.stdout
    # Redirect stdout to the dummy object
    sys.stdout = dummy_output
    # Now call the function that produces output
    raw = read_raw_bids(bids_path)
    sample_rate = raw.info['sfreq']
    channels = get_channels(raw)
    # Restore the original stdout
    sys.stdout = original_stdout
    return channels, sample_rate


def create_matrix_list(bids_path, freq_type):
    channels, sample_rate = raw_handler(bids_path)
    if channels is None:
        return None
    time = int(len(channels[0])) / sample_rate
    time = int(time / 1)
    matrix_list = []
    for sec in tqdm(range(time)):  # remove after testing
        matrix_list.append(create_matrix(sample_rate, channels[:, int(sec*sample_rate):int((sec+1)*sample_rate)]
                                         , freq_type))
    return matrix_list
"""


def find_run(directory_path, run_path):
    for filename in os.listdir(directory_path):
        if filename.startswith(run_path):
            return filename[filename.find('run') + 4]
    return None


def get_bids_path(bids_root, sub, task):
    iemu_path = op.join(bids_root, 'sub-' + sub, 'ses-iemu', "ieeg")
    run_path = "".join(('sub-' + sub, '_ses-iemu_task-', task))
    if op.isdir(iemu_path):
        run_num = find_run(iemu_path, run_path)
        if run_num is not None:
            bids_path = BIDSPath(root=bids_root, subject=sub, session=session, task=task, run=run_num,
                                 datatype=datatype, acquisition=acquisition, suffix=suffix)
            return bids_path
        else:
            return None
    else:
        return None


class CoherenceMatrix:

    def __init__(self, bids_root, sub_tag, task, freq_type):
        self.bids_path = get_bids_path(bids_root, sub_tag, task)
        self.freq_type = freq_type
        self.sample_freq = -1
        self.channels = None
        self.handle_channels()
        self.matrix_list = None

    def get_matrix_list(self):
        return self.matrix_list

    def show_matrix(self, index):
        if self.matrix_list is not None:
            plt.imshow(self.matrix_list[index], cmap='viridis')
            plt.colorbar()
            plt.show()

    def handle_channels(self):
        if self.bids_path is None:
            return
        dummy_output = io.StringIO()
        # Save the current stdout
        original_stdout = sys.stdout
        # Redirect stdout to the dummy object
        sys.stdout = dummy_output
        # Now call the function that produces output
        raw = read_raw_bids(self.bids_path)
        # Restore the original stdout
        sys.stdout = original_stdout

        # preprocessing
        raw.load_data()
        raw.set_eeg_reference()
        raw.notch_filter(np.arange(50, 251, 50))
        sfreq = raw.info['sfreq']
        if 'ecog' in raw.get_channel_types():
            self.channels = raw.pick(picks=['ecog'], exclude='bads').get_data()
            self.sample_freq = sfreq

    # channels = raw.pick(picks='ecog' , exclude='bads')

    def coherence_calc(self, index1, index2, sec):
        if self.channels is None:
            return
        freq_bounds = freq_dict.get(self.freq_type)
        freq = int(self.sample_freq)
        time_bound = sec*freq
        channel1 = self.channels[index1]
        channel2 = self.channels[index2]
        f, coherence = scipy.signal.coherence(channel1[time_bound:time_bound+freq], channel2[time_bound:time_bound+freq]
                                              , fs=self.sample_freq, nperseg=self.sample_freq / 2)
        return np.mean(coherence[freq_bounds[0]:freq_bounds[1]])

    # creating coherence matrix between each channel ( for each second)
    def create_matrix(self, sec):
        channel_count = len(self.channels)
        matrix = []
        for row in (range(channel_count)):
            for col in range(row+1, channel_count):
                matrix.append(self.coherence_calc(row, col, sec))
        return np.array(matrix, dtype= float)

    def create_matrix_list(self):
        print('creating matrixs')
        if self.channels is None:
            return
        time = int(len(self.channels[0])) / self.sample_freq
        time = int(time / 1)
        matrix_list = []
        for sec in tqdm(range(time)):  # remove after testing
            matrix_list.append(self.create_matrix(sec))
        self.matrix_list = matrix_list
