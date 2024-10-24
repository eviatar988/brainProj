import numpy as np
import pandas as pd
import os
import os.path as op
import matplotlib.pyplot as plt
import openneuro
from mne_bids import (BIDSPath, read_raw_bids, print_dir_tree, make_report, get_entity_vals)
import scipy
from mne.datasets import sample
import seaborn as sns
import mne

dataset = "ds003688"
subject = '10'


# Download one subject's data from each dataset
def download_data_openneuro(dataset, subject=None):
    bids_root = op.join(op.dirname(sample.data_path()), dataset)
    if not op.isdir(bids_root):
        os.makedirs(bids_root)
    if subject is not None:
        openneuro.download(dataset=dataset, target_dir=bids_root, include=[f"sub-{subject}"])
    else:
        openneuro.download(dataset=dataset, target_dir=bids_root)