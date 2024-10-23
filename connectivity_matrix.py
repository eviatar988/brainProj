import sys
import warnings
import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
import io
from mne_bids import (BIDSPath, read_raw_bids, print_dir_tree, make_report, get_entity_vals)
import scipy
from scipy.signal import hilbert
from scipy.signal import coherence
from tqdm import tqdm
import typing

import bids_extract

# Ignore RuntimeWarnings
session = 'iemu'
datatype = 'ieeg'
acquisition = 'clinical'
suffix = 'ieeg'
run = '1'
exten = '.vhdr'



def coherence_calc(channel1, channel2, sfreq, signal_len):
    freq = int(sfreq)
    f, coherence_value = coherence(channel1, channel2, fs=sfreq, nperseg=signal_len * freq / 2)
    return coherence_value

def calculate_plv(signal1, signal2, sfreq, signal_len):
    """
    Calculate the Phase-Locking Value (PLV) between two signals.

    Parameters:
        signal1 (numpy.ndarray): First signal.
        signal2 (numpy.ndarray): Second signal.

    Returns:
        float: PLV value between signal1 and signal2.
    """

    # Apply Hilbert transform to get the analytic signal
    analytic_signal1 = hilbert(signal1)
    analytic_signal2 = hilbert(signal2)

    # Extract the phase from the analytic signals
    phase1 = np.unwrap(np.angle(analytic_signal1))
    phase2 = np.unwrap(np.angle(analytic_signal2))
    # Calculate the phase difference
    phase_diff = phase1 - phase2

    # Compute the PLV
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))

    return plv


def create_matrix(channels, sfreq, signal_len, task=coherence_calc):
    channel_count = len(channels)
    matrix = []
    for row in (range(channel_count)):
        for col in range(row + 1, channel_count):
            matrix.append(task(channels[row], channels[col], sfreq, signal_len))
    return matrix



def create_matrix_list(bids_root: str, sub_tag: str, signal_len, task=coherence_calc):
    bids_path_rest = bids_extract.get_bids_path(bids_root, sub_tag, 'rest')
    bids_path_film = bids_extract.get_bids_path(bids_root, sub_tag, 'film')
    rest_channels, film_channels, rest_sfreq, film_sfreq = bids_extract.handle_channels(bids_path_rest, bids_path_film)

    if rest_channels is None or film_channels is None:  # for the case that a patient have non or only one type of data
        return None, None
    time = int(len(rest_channels[0])) / rest_sfreq
    time = time / signal_len
    time = int(time)
    matrix_list_rest = []
    for sec in tqdm(range(time)):  # remove after testing
        lower_bound = signal_len * sec * rest_sfreq
        upper_bound = (signal_len*sec + signal_len) * rest_sfreq
        matrix_list_rest.append(
            create_matrix(rest_channels[:, lower_bound:upper_bound], rest_sfreq, signal_len, task))

    time = int(len(film_channels[0])) / film_sfreq
    time = time/signal_len
    time = int(time)
    matrix_list_film = []
    for sec in tqdm(range(time)):
        lower_bound = signal_len * sec * film_sfreq
        upper_bound = (signal_len * sec + signal_len) * film_sfreq
        matrix_list_film.append(
            create_matrix(film_channels[:, lower_bound:upper_bound], film_sfreq, signal_len, task))
    return matrix_list_rest, matrix_list_film



