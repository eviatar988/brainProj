
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
from bids_extract import bidsExtract

bids_path = r"C:\Users\eyala\Documents\GitHub\brainProj\ds003688"


def main():
    dataset = "ds003688"
    subject = "02"

    # Download one subject's data from each dataset
    bids_root = op.join(op.dirname(sample.data_path()), dataset)
    if not op.isdir(bids_root):
        os.makedirs(bids_root)

    openneuro.download(dataset=dataset, target_dir=bids_root, include=[f"sub-{subject}"])

    print_dir_tree(bids_root, max_depth=4)
if __name__ == '__main__':
    main()

