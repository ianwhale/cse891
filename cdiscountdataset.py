#
# Code for processing the .bson files for use elsewhere.
#
# Functions adapted from various Kaggle kernel:
#   https://www.kaggle.com/humananalog/keras-generator-for-reading-directly-from-bson
#

import io
import bson
import struct
import numpy as np
import pandas as pd
from os.path import isfile
from collections import defaultdict
from torch.utils.data import Dataset


class CDiscountDataSet(Dataset):
    """
    Extend Torch Dataset base class to play nice with other Torch functionality.
    The Dataset subclasses only need to implement __len__ and __getitem__.
    The DataLoader takes care of everything else like batching.
    """

    CATEGORY_NAMES_PATH = 'category_names.csv'
    OFFSETS_PATH = 'offsets.csv'

    def __init__(self, bsonpath, categories_path=None):
        """
        Constructor.
        Creates the files necessary for random indexing across the bson file.
        :param bsonpath: filepath to train.bson or test.bson.
        """
        if not isfile(bsonpath):
            raise FileNotFoundError("CDiscountDataSet __init__ given nonexistent bson filepath.")

        if not categories_path:
            categories_path = self.CATEGORY_NAMES_PATH

        if not isfile(categories_path):
            raise FileNotFoundError("CDiscountDataSet __init__ can't find categories csv.")

        self.cat2idx = {}  # Given category, transform to index.
        self.idx2cat = {}  # Given index, transform to category (not really used, but here for possible use later).
        self.make_category_tables(categories_path)

        self.num_categories = len(self.cat2idx)  # Number of categories. Used for one-hot encoding.

        self.offsets = None  # Csv loaded into memory as a pandas dataframe.
        self.get_offsets(bsonpath)

        self.data = None  # Actual bson data file pointer (doesn't load into memory).

    def make_category_tables(self, categories_path):
        """
        Set up self.cat2idx and self.idx2cat with the categories
        :param categories_path:
        :return:
        """
        categories_df = pd.read_csv(categories_path, index_col="category_id")
        categories_df["category_idx"] = pd.Series(range(len(categories_df)), index=categories_df.index)

        for itr in categories_df.itertuples():
            category_id = itr[0]
            category_idx = itr[4]

            self.cat2idx[category_id] = category_idx
            self.idx2cat[category_idx] = category_id

    def get_offsets(self, bsonpath):
        """
        Create the offset dictionary and csv for random access into the bson file.
        Saves file as offsets.csv.
        For the full train.bson file, this will take several minutes.
        :param bsonpath: filepath to train.bson or test.bson.
        """
        if isfile(self.OFFSETS_PATH):
            self.offsets = pd.read_csv(self.OFFSETS_PATH)

        rows = {}

        with open(bsonpath, "rb") as f:
            offset = 0

            while True:
                item_length_bytes = f.read(4)

                if len(item_length_bytes) == 0:
                    break

                length = struct.unpack("<i", item_length_bytes)[0]

                f.seek(offset)
                item_data = f.read(length)

                assert len(item_data) == length, "Assert failed, something went wrong with reading data length."

                item = bson.BSON(item_data).decode()
                product_id = item["_id"]
                num_imgs = len(item["imgs"])
                category = item["category_id"]

                row = [num_imgs, offset, length, category]
                rows[product_id] = row

                offset += length
                f.seek(offset)

        columns = ["num_imgs", "offset", "length", "category_id"]

        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "product_id"
        df.columns = columns
        df.sort_index(inplace=True)

        df.to_csv(self.OFFSETS_PATH)

        self.offsets = df


def demo():
    """
    Demonstrate the primary functions in this file.
    """
    ds = CDiscountDataSet('train_example.bson')

    print("Total number of categories: {:d}".format(len(ds.cat2idx)))


if __name__ == "__main__":
    # Demo if train_example.bson is present.
    if isfile("train_example.bson"):
        demo()

    else:
        print("Whoops, train_example.bson not found in immediate directory.")
        print("Exiting demo.")
