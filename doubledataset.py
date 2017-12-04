import io
import bson
import torchvision
from PIL import Image
from torch.utils.data import Dataset


class DoubleDataSet(Dataset):
    def __init__(self, fp1, fp2):
        """
        Constructor. Loads everything into memory, so be warned.
        :param fp1: filepath to first bson dataset.
        :param fp2: filepath to second bson dataset.
        """
        self.class_indexes = [[], []]  # [0] gives class 0 indices, [1] give class 1 indicies.
        # Pass this to a sampler in the DataLoader.

        self.data, (self.binary_labels, self.categorical_labels) = self.parse_files(fp1, fp2)
        self.transform = torchvision.transforms.ToTensor()

    def __getitem__(self, item):
        """
        Index into the dataset.
        :param item: index of desired item.
        :return: img, label
        """
        img = self.transform(Image.open(io.BytesIO(self.data[item])))

        return img, self.binary_labels[item], self.categorical_labels[item]

    def __len__(self):
        return len(self.data)

    def parse_files(self, fp1, fp2):
        """
        Generate the data and label lists.
        :param fp1: file path 1.
        :param fp2: file path 2.
        :return:
        """
        data = []
        labels = []

        count = 0
        for i, fp in enumerate([fp1, fp2]):
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
        binary_labels = [0 for _ in label_list[0]] + [1 for _ in label_list[1]]

        label_list = label_list[0] + label_list[1]

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
