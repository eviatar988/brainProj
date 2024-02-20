import sys
import warnings
import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
import io
from mne_bids import (BIDSPath, read_raw_bids, print_dir_tree, make_report, get_entity_vals)
import scipy

# Ignore RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


session = 'iemu'
datatype = 'ieeg'
acquisition = 'clinical'
suffix = 'ieeg'
run = '1'
exten = '.vhdr'


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


def coherence_calc(signal_1, signal_2, freq):
    f, coherence = scipy.signal.coherence(signal_1, signal_2, fs=freq, nperseg=freq / 2)
    return np.mean(coherence)


# creating coherence matrix between each channel ( for each second)
def create_matrix(sec, freq, channels):
    channel_count = len(channels)
    matrix = np.empty((channel_count, channel_count))
    sec = int(sec)
    freq = int(freq)
    for row in (range(channel_count)):
        for col in range(row, channel_count):
            matrix[row][col] = coherence_calc(channels[row][sec * freq:(sec + 1) * freq],
                                              channels[col][sec * freq:(sec + 1) * freq], freq)
            matrix[col][row] = matrix[row][col]
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


def create_matrix_list(bids_path):
    channels, sample_rate = raw_handler(bids_path)
    if channels is None:
        return None
    time = int(len(channels[0])) / sample_rate
    time = int(time / 1)
    matrix_list = []
    for sec in range(time):  # remove after testing
        matrix_list.append(create_matrix(sec, sample_rate, channels))
    return matrix_list


class CoherenceMatrix:
    def __init__(self, bids_root, sub_tag, task):
        self.sub = sub_tag
        self.task = task
        self.bids_root = bids_root
        self.bids_path = get_bids_path(bids_root, sub_tag, task)
        self.matrix_list = None
        if self.bids_path is not None:
            self.matrix_list = create_matrix_list(self.bids_path)

    def get_root(self):
        return self.bids_root

    def get_sub(self):
        return self.sub

    def get_task(self):
        return self.task

    def get_matrix_list(self):
        return self.matrix_list

    def show_matrix(self, index):
        plt.imshow(self.matrix_list[index], cmap='viridis')
        plt.colorbar()
        plt.show()
