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
from math import ceil
from torch.optim import lr_scheduler
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

    def __init__(self, bsonpath,
                 categories_path=None,
                 transform=None,
                 category_level=3,
                 trim_classes=None):
        """
        Constructor.
        Creates the files necessary for random indexing across the bson file.
        :param bsonpath: filepath to train.bson or test.bson.
        :param categories_path: path to category_names.csv.
        :param transform: pytorch transformation.
        :param category_level: category level to classify at. Options:
            - 3: All three categories (biggest task, 5270 classes).
            - 2: Second category.
            - 1: First category (smallest task, around 50 classes).
        :param trim_classes: int or None, if int, all classes with less than X examples are remove, all those with
            more are trimmed to have only X.
        """
        if not isfile(bsonpath):
            raise FileNotFoundError("CDiscountDataSet __init__ given nonexistent bson filepath.")

        if not categories_path:
            categories_path = self.CATEGORY_NAMES_PATH

        if not isfile(categories_path):
            raise FileNotFoundError("CDiscountDataSet __init__ can't find categories csv.")

        self.transform = transform

        self.category_level = category_level  # This all made sense at the time...
        self.level2class = {}  # Given a category level index, transform to a output class label.
        self.idx2level = {}    # Given an index, transform to a category level index.
        self.cat2idx = {}      # Given category, transform to index.
        self.idx2cat = {}      # Given index, transform to category (not really used, but here for possible use later).
        self.make_category_tables(categories_path)

        self.offsets = None  # Csv loaded into memory as a pandas dataframe.
        self.get_offsets(bsonpath)

        self.trim_classes = trim_classes
        self.indexes = None  # Csv loaded into memory as a pandas dataframe.
        self.get_indexes()

        self.build_level2class()

        self.file = open(bsonpath, "rb")  # Actual bson data file pointer (doesn't load into memory).

        self.lock = threading.Lock()  # Lock since Pytorch uses multithreading (maybe?).

        self.num_categories = max(self.level2class.values()) + 1

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
        # label = np.zeros(self.num_categories, dtype=np.int64)  # PyTorch insists on longs for whatever reason...
        # label[index_row["category_idx"]] = 1  # One-hot encoding.

        if self.transform:
            image = self.transform(image)

        return image, self.level2class[self.idx2level[index_row["category_idx"]]]

    def make_category_tables(self, categories_path):
        """
        Set up self.cat2idx and self.idx2cat with the categories
        :param categories_path:
        :return:
        """
        categories_df = pd.read_csv(categories_path, index_col="category_id")
        categories_df["category_idx"] = pd.Series(range(len(categories_df)), index=categories_df.index)

        counter = 0
        category_numbers = {}
        for itr in categories_df.itertuples():
            category_id = itr[0]
            category_idx = itr[4]

            self.cat2idx[category_id] = category_idx
            self.idx2cat[category_idx] = category_id

            # A category may have the same name at the 3rd level only as some other category, but different names
            # at levels 1 and 2, so we build a tuple of length 1, 2, or 3 depending on the level to distinguish them.
            # E.g.: we have could have a "light" in automotive, car parts, light but also have a "light" in
            #       home & garden, interior, light.
            category_tuple = tuple(itr[i + 1] for i in range(self.category_level))

            if category_tuple not in category_numbers:
                category_numbers[category_tuple] = counter
                counter += 1

            self.idx2level[category_idx] = category_numbers[category_tuple]

    def build_level2class(self):
        """
        Clases may have been removed due to the new trimming processes, so we have to build a output class dictionary.
        """
        seen = 0
        for row in self.indexes.itertuples():
            category = row[2]

            if category not in self.level2class:
                self.level2class[self.idx2level[category]] = seen
                seen += 1

    def get_offsets(self, bsonpath):
        """
        Create the offset dictionary and csv for random access into the bson file.
        Saves file as offsets.csv.
        For the full train.bson file, this will take several minutes.
        :param bsonpath: filepath to train.bson or test.bson.
        """
        if isfile("training" + self.OFFSETS_PATH):
            print("Found stored offsets! Loading from csv...")
            self.offsets = pd.read_csv("training" + self.OFFSETS_PATH, index_col=0)
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

        df.to_csv("training" + self.OFFSETS_PATH)

        self.offsets = df

    def trim_categories(self, category_dict):
        """
        Remove the categories that are too small and sample from the ones that are too big.
        :param category_dict: maps category number to list of product ids.
        :return: dict
        """
        counts = defaultdict(int)
        level2cat = defaultdict(list)

        for key, value in category_dict.items():
            level = self.idx2level[self.cat2idx[key]]
            counts[level] += len(value)

            level2cat[level].append(key)

        to_remove = []

        for key, value in category_dict.items():
            level = self.idx2level[self.cat2idx[key]]
            count = counts[level]

            if count < self.trim_classes:
                # Remove all the categories associated with that class level, there are too few examples.
                to_remove += level2cat[level]

            if count > self.trim_classes:
                # This is a little complicated since we only know the sum of the values is over the trim limit.
                # Take an equal number of examples from each category, but if there aren't enough just take them all.
                # If we do the second option, we have to update how many examples we need from the other classes.
                cats = level2cat[level]
                num_cats = len(level2cat[level])
                total_examples = 0
                cats_used = 0
                examples_needed = ceil(self.trim_classes / num_cats)

                # Sort the categories by how many ids they have.
                # We want the smallest ones first to be able to compensate for them being possibly too small.
                cats.sort(key=lambda x: len(category_dict[x]))

                for cat in cats:
                    cats_used += 1
                    ids = category_dict[cat]

                    if len(ids) < examples_needed:
                        total_examples += len(ids)
                        examples_needed = ceil((self.trim_classes - total_examples) / (num_cats - cats_used))

                    else:
                        category_dict[cat] = np.random.choice(ids, examples_needed, replace=False)
                        total_examples += examples_needed

            # In the unlikely event we have examples self.trim_classes examples, we don't do anything.

        for key in to_remove:
            category_dict.pop(key, None)

        return category_dict

    def get_indexes(self):
        """
        Get the indexes that will actually be used.
        """
        if not self.trim_classes and isfile("training" + self.INDEXES_PATH):
            print("Found stored indexes! Loading from csv...")
            self.indexes = pd.read_csv("training" + self.INDEXES_PATH, index_col=0)
            return

        if self.trim_classes and isfile("training_" + str(self.trim_classes) + self.INDEXES_PATH):
            print("Found stored indexes with trimmed classes! Loading from csv...")
            self.indexes = pd.read_csv("training_" + str(self.trim_classes) + self.INDEXES_PATH, index_col=0)
            return

        category_dict = defaultdict(list)  # Product ids belonging to each category.

        for itr in self.offsets.itertuples():
            category_dict[itr[4]].append(itr[0])

        if self.trim_classes:
            category_dict = self.trim_categories(category_dict)

        index_list = []
        for category_id, product_ids in category_dict.items():
            category_idx = self.cat2idx[category_id]

            for product_id in product_ids:
                row = [product_id, category_idx]

                for img_idx in range(self.offsets.loc[product_id, "num_imgs"]):
                    index_list.append(row + [img_idx])

        columns = ["product_id", "category_idx", "img_idx"]
        index_df = pd.DataFrame(index_list, columns=columns)

        index_df.to_csv("training" + ("_" + str(self.trim_classes) if self.trim_classes else "") + self.INDEXES_PATH)

        self.indexes = index_df


def demo():
    from os import remove
    """
    Demonstrate the primary functions in this file.
    """
    ds = CDiscountDataSet('train_example.bson', category_level=1)

    print("Total number of output classes: {:d}".format(ds.num_categories))
    print("Number of images: {:d}".format(len(ds)))

    print("\nAn example of the offsets dataframe: ")
    print(ds.offsets.iloc[0])

    print("\nAn example of the indexes dataframe: ")
    print(ds.indexes.iloc[0])

    print("\nDo a random access into the dataset: ")
    print(ds[0])

    for i in range(len(ds)):
        # Make sure everything can be indexed without error.
        ds[i]

    # Use ds[i][0].show() to look at the image if you want.

    # Clean up testing files that were created as they'll mess things up if we try to construct on the actual datasets.
    remove('training' + ds.OFFSETS_PATH)
    remove('training' + ds.INDEXES_PATH)


if __name__ == "__main__":
    # Demo if train_example.bson is present.
    if isfile("train_example.bson"):
        demo()

    else:
        print("Whoops, train_example.bson not found in immediate directory.")
        print("Exiting demo.")
