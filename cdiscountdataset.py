#
# Code for processing the .bson files for use elsewhere.
#
# Functions adapted from various Kaggle kernels:
#   https://www.kaggle.com/inversion/processing-bson-files
#   https://www.kaggle.com/humananalog/keras-generator-for-reading-directly-from-bson
#

import io
import bson
import numpy as np
import pandas as pd
from os.path import isfile
from torch.utils.data import Dataset


class CDiscountDataSet(Dataset):
    """
    Extend Torch Dataset to place nice with other functionality.
    """

    def __init__(self, filepath=None):
        if not isfile(filepath):
            raise FileNotFoundError("Provided file to constructor does not exist.")





def demo():
    """
    Demonstrate the primary functions in this file.
    """
    pass


if __name__ == "__main__":
    # Demo if train_example.bson is present.
    try:
        f = open("train_example.bson", 'rb')
        demo()

    except FileNotFoundError:
        print("Whoops, train_example.bson not found in immediate directory.")
        print("Exiting demo.")
