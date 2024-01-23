
import mne_bids
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from mne_bids import (BIDSPath, read_raw_bids, print_dir_tree, make_report, get_entity_vals)
import scipy

import seaborn as sns
import mne


class Patient:
    def __init__(self, name):
        pass
