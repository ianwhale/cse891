import io
import bson
import torchvision
from PIL import Image
from torch.utils.data import Dataset


class DoubleDataSet(Dataset):
    def __init__(self, fp1, fp2, which):
        """
        Constructor. Loads everything into memory, so be warned.
        :param fp1: filepath to first bson dataset.
        :param fp2: filepath to second bson dataset.
        """

        fps = None

        self.which = which

        if which == "flat":
            fps = [fp1, fp2]
            self.num_classes = 40

        elif which == "binary":
            fps = [fp1, fp2]
            self.num_classes = 2

        elif which == "meuble":
            fps = [fp1]
            self.num_classes = 20

        elif which == "electronique":
            fps = [fp2]
            self.num_classes = 20

        self.class_indexes = [[], []]  # [0] gives class 0 indices, [1] give class 1 indicies.
        # Pass this to a sampler in the DataLoader.

        self.data, (self.binary_labels, self.categorical_labels) = self.parse_files(fps)
        self.transform = torchvision.transforms.ToTensor()

    def __getitem__(self, item):
        """
        Index into the dataset.
        :param item: index of desired item.
        :return: img, label
        """
        img = self.transform(Image.open(io.BytesIO(self.data[item])))

        if self.which in ["flat", "meuble", "electronique"]:
            return img, self.categorical_labels[item]

        elif self.which in ["binary"]:
            return img, self.binary_labels[item]

        else:
            print("bad which")
            exit(-1)

    def __len__(self):
        return len(self.data)

    def parse_files(self, fps):
        """
        Generate the data and label lists.
        :param fp1s: list of file paths.
        :return:
        """
        data = []
        labels = []

        count = 0
        for i, fp in enumerate(fps):
            bson_data = bson.decode_file_iter(open(fp, 'rb'))

            fp_labels = []  # Labels for a specific file.
            indexes = []

            for entry in bson_data:
                indexes.append(count)
                count += 1

                data.append(entry["image"])
                fp_labels.append(entry["category"])

            self.class_indexes[i] = indexes
            labels.append(fp_labels)

        return data, DoubleDataSet.parse_labels(labels)

    @staticmethod
    def parse_labels(label_list):
        """
        Remap labels to that they are binary and categorical appropriatley.
        :param label_list:
        :return:
        """
        binary_labels = [i for i, lis in enumerate(label_list) for __ in lis]

        label_list = [lab for lis in label_list for lab in lis]

        category_counts = {}
        count = 0

        category_labels = []
        for label in label_list:
            if label in category_counts:
                category_labels.append(category_counts[label])

            else:
                category_counts[label] = count
                count += 1

                category_labels.append(category_counts[label])

        return binary_labels, category_labels


def demo():
    furniture = "./data/meuble_data.bson"
    electronics = "./data/electronique_data.bson"

    set = DoubleDataSet(furniture, electronics)

    print("Running asserts to make sure data is properly formatted...")

    for i in set.class_indexes[0]:
        img, lvl_1, lvl_3 = set[i]

        assert lvl_1 == 0
        assert 0 <= lvl_3 <= 19

    for i in set.class_indexes[1]:
        img, lvl_1, lvl_3 = set[i]

        assert lvl_1 == 1
        assert 20 <= lvl_3 <= 39

    print("Everything is ok.")


if __name__ == "__main__":
    demo()
