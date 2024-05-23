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
import typing

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

    def __init__(self, bids_root: str, sub_tag: str, task: str, sec_per_sample: int):
        self.bids_path = get_bids_path(bids_root, sub_tag, task)
        self.sample_freq = -1
        self.sec_per_sample = int(sec_per_sample)
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

    def coherence_calc(self, index1: int, index2: int, sec: int):
        if self.channels is None:
            return
        freq = int(self.sample_freq)
        lower_bound = int(sec*freq*self.sec_per_sample)
        upper_bound = int(lower_bound + freq*self.sec_per_sample)
        channel1 = self.channels[index1]
        channel2 = self.channels[index2]
        f, coherence = scipy.signal.coherence(channel1[lower_bound:upper_bound],
                                              channel2[lower_bound:upper_bound+freq]
                                              , fs=self.sample_freq, nperseg=self.sample_freq / 2)
        return coherence

    # creating coherence matrix between each channel ( for each second)
    def create_matrix(self, sec):
        channel_count = len(self.channels)
        matrix = []
        for row in (range(channel_count)):
            for col in range(row+1, channel_count):
                matrix.append(self.coherence_calc(row, col, sec))
        return matrix

    def create_matrix_list(self):
        if self.channels is None:
            return
        time = int(len(self.channels[0])) / self.sample_freq
        time = int(time / self.sec_per_sample)
        matrix_list = []
        for sec in tqdm(range(time)):  # remove after testing
            matrix_list.append(self.create_matrix(sec))
        self.matrix_list = matrix_list
