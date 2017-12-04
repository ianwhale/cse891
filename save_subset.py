#
# Saves a subset of the data to a .pt file.
# Allows the choice of which level 1 category to choose.
# Files are opened and manipulated just like the are in CDiscountDataSet.__getitem__
#


import io
import bson
import torchvision
import pandas as p
import multiprocessing as mp
from PIL import Image
from collections import defaultdict


NCORE = 4


def save_subset(category_dict, name, fp, num_images=15000):
    """
    Saves the images and level 1 and 3 labels to a file.

    Produces 3 files.
        1:

    :param category_dict: dict, level 1 category ids. All the ids of the level 1 category.
    :param name: name of the category.
    :param num_images: number of images to save in total.
    """
    transform = torchvision.transforms.Compose([
                    torchvision.transforms.Scale(224),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                    torchvision.transforms.ToPILImage()
                    ])

    def process(q):
        """
        Worker process. Writes the transformed image to a file.
        :param q: multiprocessing queue.
        """
        while True:  # Continuously consume data.
            entry = q.get()

            if entry is None:
                break

            data_f = open(name + "_data.bson", "ab")

            for img in entry['imgs']:
                img = Image.open(io.BytesIO(img['picture']))
                img = transform(img)

                img_bytes = io.BytesIO()
                img.save(img_bytes, format="JPEG")

                data_f.write(bson.BSON.encode(
                    {
                        'category': entry['category_id'],
                        'image': img_bytes.getvalue()
                    }))

            data_f.close()

    data = bson.decode_file_iter(open(fp, 'rb'))

    q = mp.Queue(maxsize=NCORE)
    pool = mp.Pool(NCORE, initializer=process, initargs=(q, ))

    # Create file here and close it to remove possibility of processes racing to create it.
    f = open(name + '_data.bson', 'wb')
    f.close()

    images = 0

    category_counts = defaultdict(int)
    per_cat_threshold = 500

    for entry in data:
        if entry['category_id'] in category_dict:

            if images > num_images:  # We've collected ~15,000 images.
                break

            if category_counts[entry['category_id']] > per_cat_threshold:  # We've collected enough from this category.
                continue

            q.put(entry)

            images += len(entry['imgs'])
            category_counts[entry['category_id']] += len(entry['imgs'])

    for _ in range(NCORE):
        q.put(None)

    pool.close()
    pool.join()


def name_to_id_dict(name):
    """
    Turn a category name into an dictionary of category ids that belong to that name.
    I.e. if we chose MEUBLE (furniture), we'd want ids 1000012439 through 1000019516.
    :param name:
    :return:
    """
    d = {}

    frame = p.read_csv('category_names.csv')

    for _, row in frame.iterrows():
        if row[1] == name:
            d[row[0]] = True

    return d


def demo():
    # cat = "MEUBLE"  # Furniture.
    # cat = "ELECTRONIQUE"  # Electronics.

    cat = "TELEPHONIE - GPS"  # This is used for example, the two we actually need are not in train_example.bson.

    d = name_to_id_dict(cat)
    save_subset(d, cat, 'train_example.bson')

    data = bson.decode_file_iter(open(cat + "_data.bson", "rb"))

    to_tensor = torchvision.transforms.ToTensor()

    for item in data:
        print(item)  # Item in dictionary form.

        # Demonstrate how read and convert an image.
        img = io.BytesIO(item['image'])
        img = Image.open(img)
        img = to_tensor(img)

        print(img)


if __name__ == "__main__":
    demo()
