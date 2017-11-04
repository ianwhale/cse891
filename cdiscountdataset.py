#
# Code for processing the .bson files for use elsewhere.
#
# Functions adapted from Kaggle kernel:
#   https://www.kaggle.com/humananalog/keras-generator-for-reading-directly-from-bson
#

import io
import bson
import torch
import struct
import threading
import numpy as np
import pandas as pd
from PIL import Image
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
    OFFSETS_PATH = '_offsets.csv'
    INDEXES_PATH = '_indexes.csv'

    def __init__(self, bsonpath, training=True, categories_path=None, transform=None):
        """
        Constructor.
        Creates the files necessary for random indexing across the bson file.
        :param bsonpath: filepath to train.bson or test.bson.
        :param training: is this the training dataset?
        :param categories_path: path to category_names.csv.
        :param transform: pytorch transformation.
        """
        if not isfile(bsonpath):
            raise FileNotFoundError("CDiscountDataSet __init__ given nonexistent bson filepath.")

        if not categories_path:
            categories_path = self.CATEGORY_NAMES_PATH

        if not isfile(categories_path):
            raise FileNotFoundError("CDiscountDataSet __init__ can't find categories csv.")

        self.training = training
        self.transform = transform

        self.cat2idx = {}  # Given category, transform to index.
        self.idx2cat = {}  # Given index, transform to category (not really used, but here for possible use later).
        self.make_category_tables(categories_path)

        self.num_categories = len(self.cat2idx)  # Number of categories. Used for one-hot encoding.

        self.offsets = None  # Csv loaded into memory as a pandas dataframe.
        self.get_offsets(bsonpath)

        self.indexes = None  # Csv loaded into memory as a pandas dataframe.
        self.get_indexes()

        self.file = open(bsonpath, "rb")  # Actual bson data file pointer (doesn't load into memory).

        self.lock = threading.Lock()  # Lock since Pytorch uses multithreading (maybe?).

    def __len__(self):
        """
        Total number of images in the data set.
        :return: int
        """
        return len(self.indexes)

    def __getitem__(self, item):
        """
        Random access to an image and its label
        :param item: integer, index of desired image.
        :return: tuple, (image, label) with types (PIL.Image, np.array)
        """
        # Only need to protect when we're accessing the file and dataframes.
        with self.lock:
            index_row = self.indexes.iloc[item]
            product_id = index_row["product_id"]
            offset_row = self.offsets.loc[product_id]

            # Read product's data from BSON file.
            self.file.seek(offset_row["offset"])
            item_data = self.file.read(offset_row["length"])

        # Get the image and label.
        item = bson.BSON(item_data).decode()
        img_idx = index_row["img_idx"]
        bson_img = io.BytesIO(item["imgs"][img_idx]["picture"])

        # Create the image object.
        image = Image.open(bson_img)
        label = np.zeros(self.num_categories, dtype=np.int64)  # PyTorch insists on longs for whatever reason...
        label[index_row["category_idx"]] = 1  # One-hot encoding.

        if self.transform:
            image = self.transform(image)

        return image, torch.from_numpy(label)

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
        if self.training and isfile("training" + self.OFFSETS_PATH):
            print("Found stored offsets! Loading from csv...")
            self.offsets = pd.read_csv("training" + self.OFFSETS_PATH, index_col=0)
            return

        if not self.training and isfile("testing" + self.OFFSETS_PATH):
            print("Found stored offsets! Loading from csv...")
            self.offsets = pd.read_csv("testing" + self.OFFSETS_PATH, index_col=0)
            return

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

                assert len(item_data) == length, "Something went wrong with reading data length."

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

        df.to_csv(("training" if self.training else "testing") + self.OFFSETS_PATH)

        self.offsets = df

    def get_indexes(self):
        """
        Get the indexes that will actually be used.

        TODO: Doing some more sophisticated choice here, rather than choosing everything.
              Possibilities:
                    - Balance class sizes.
                    - Remove some smallest % of classes.
        """
        if self.training and isfile("training" + self.INDEXES_PATH):
            print("Found stored indexes! Loading from csv...")
            self.indexes = pd.read_csv("training" + self.INDEXES_PATH, index_col=0)
            return

        if not self.training and isfile("testing" + self.INDEXES_PATH):
            print("Found stored indexes! Loading from csv...")
            self.indexes = pd.read_csv("testing" + self.INDEXES_PATH, index_col=0)
            return

        category_dict = defaultdict(list)  # Product ids belonging to each category.

        for itr in self.offsets.itertuples():
            category_dict[itr[4]].append(itr[0])

        index_list = []
        for category_id, product_ids in category_dict.items():
            category_idx = self.cat2idx[category_id]

            for product_id in product_ids:
                row = [product_id, category_idx]

                for img_idx in range(self.offsets.loc[product_id, "num_imgs"]):
                    index_list.append(row + [img_idx])

        columns = ["product_id", "category_idx", "img_idx"]
        index_df = pd.DataFrame(index_list, columns=columns)

        index_df.to_csv(("training" if self.training else "testing") + self.INDEXES_PATH)

        self.indexes = index_df


def demo():
    """
    Demonstrate the primary functions in this file.
    """
    ds = CDiscountDataSet('train_example.bson')

    print("Total number of categories: {:d}".format(len(ds.cat2idx)))
    print("Number of images: {:d}".format(len(ds.indexes)))

    print("\nAn example of the offsets dataframe: ")
    print(ds.offsets.iloc[0])

    print("\nAn example of the indexes dataframe: ")
    print(ds.indexes.iloc[0])

    print("\nDo a random access into the dataset: ")
    print(ds[5])

    for i in range(len(ds)):
        # Make sure everything can be indexed without error.
        ds[i]

    # Use ds[i][0].show() to look at the image if you want.


if __name__ == "__main__":
    # Demo if train_example.bson is present.
    if isfile("train_example.bson"):
        demo()

    else:
        print("Whoops, train_example.bson not found in immediate directory.")
        print("Exiting demo.")
