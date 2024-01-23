
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


def signal_handler(bids_root, sub, task):
    bids_path = BIDSPath(root=bids_root, subject=sub, session=session, task=task, run=run,
                         datatype=datatype, acquisition=acquisition, suffix=suffix, extension=exten)
    return mne_bids.read_raw_bids(bids_path)


def create_matrix(sub, task, bids_root):
    bids_path = BIDSPath(root=bids_root, subject=sub, session=session, task=task, run=run,
                         datatype=datatype, acquisition=acquisition, suffix=suffix, extension=exten)
    raw = mne_bids.read_raw_bids(bids_path)

    matrix = []



class CoherenceMatrix:
    def __init__(self, sub_tag, task, bids_root):
        self.matrix = create_matrix(sub_tag, task)

