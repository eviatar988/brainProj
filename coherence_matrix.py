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

session = 'iemu'
datatype = 'ieeg'
acquisition = 'clinical'
suffix = 'ieeg'
run = '1'
exten = '.vhdr'


# find the number of run on data
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


def coherence_calc(channel1, channel2, sfreq):
    freq = int(sfreq)
    f, coherence = scipy.signal.coherence(channel1, channel2, fs=sfreq, nperseg= 3 * freq / 2)
    return coherence


def create_matrix(channels, sfreq):
    channel_count = len(channels)
    matrix = []
    for row in (range(channel_count)):
        for col in range(row + 1, channel_count):
            matrix.append(coherence_calc(channels[row], channels[col], sfreq))
    return matrix


def create_matrix_list(bids_root: str, sub_tag: str):
    bids_path_rest = get_bids_path(bids_root, sub_tag, 'rest')
    bids_path_film = get_bids_path(bids_root, sub_tag, 'film')
    rest_channels, film_channels, rest_sfreq, film_sfreq = handle_channels(bids_path_rest, bids_path_film)

    if rest_channels is None or film_channels is None:  # for the case that a patient have non or only one type of data
        return None, None
    time = int(len(rest_channels[0])) / rest_sfreq
    time = time / 3
    time = int(time)
    matrix_list_rest = []
    for sec in tqdm(range(time)):  # remove after testing
        matrix_list_rest.append(
            create_matrix(rest_channels[:, 3*sec * rest_sfreq:(3*sec + 3) * rest_sfreq], rest_sfreq))

    time = int(len(film_channels[0])) / film_sfreq
    time = time/3
    time = int(time)
    matrix_list_film = []
    for sec in tqdm(range(time)):  # remove after testing
        matrix_list_film.append(
            create_matrix(film_channels[:, 3*sec * film_sfreq:(3*sec + 3) * film_sfreq], film_sfreq))
    return matrix_list_rest, matrix_list_film



