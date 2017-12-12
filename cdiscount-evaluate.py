
# load the four trained models
# and evaluate the classification accuracies of our
# two different approaches


import time
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from random import shuffle
from torch.autograd import Variable
from cdiscountdataset import CDiscountDataSet
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from collections import defaultdict

from doubledataset import DoubleDataSet as DDS



def get_split_indices(n, split):
    idx = [i for i in range(n)]
    split = int(n * split)

    shuffle(idx)

    return idx[0: n - split], idx[n - split:]

def load_data():
    batch_size = 256
    validation_split = 0.25

    furniture = "./data/meuble_data.bson"
    electronics = "./data/electronique_data.bson"

    tset = DDS(furniture, electronics, "flat")

    num_classes = tset.num_classes

    train_idx, valid_idx = get_split_indices(len(tset), validation_split)

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(tset, sampler=train_sampler, batch_size=batch_size)
    valid_loader = DataLoader(tset, sampler=valid_sampler, batch_size=batch_size)

    print("Done.")

    return valid_loader, batch_size

def classify_flat(flat_model, valid_loader):
    cuda = True

    if cuda:
        flat_model = flat_model.cuda()

    flat_model.train(False)

    running_corrects = 0

    for data in valid_loader:
        inputs, labels = data

        if cuda:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())

        else:
            inputs = Variable(inputs)
            labels = Variable(labels)

        outputs = flat_model(inputs)
        _, preds = torch.max(outputs.data, 1)

        running_corrects += torch.sum(preds == labels.data)


    epoch_acc = running_corrects / len(valid_loader.dataset)

    print("Flat Acc: {:.4f}".format(epoch_acc))

    return epoch_acc

def classify_hier(
        binary_model,
        meuble_model,
        electronique_model,
        valid_loader
    ):
    cuda = True

    if cuda:
        binary_model = binary_model.cuda()
        meuble_model = meuble_model.cuda()
        electronique_model = electronique_model.cuda()

    running_corrects = 0

    for data in valid_loader:
        inputs, labels = data

        if cuda:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())

        else:
            inputs = Variable(inputs)
            labels = Variable(labels)

        # run all three models on input data...
        binary_outputs = binary_model(inputs)
        _, binary_preds = torch.max(binary_outputs.data, 1)

        meuble_outputs = meuble_model(inputs)
        _, meuble_preds = torch.max(meuble_outputs.data, 1)

        electronique_outputs = electronique_model(inputs)
        _, electronique_preds = torch.max(electronique_outputs.data, 1)

        # create filters based on binary prediction
        meuble_filter = (binary_preds == 0)
        electronique_filter = (binary_preds == 1)

        # check correctness of results after binary prediction
        running_corrects += torch.sum(meuble_preds[meuble_filter] == labels.data[meuble_filter])
        # add 20 to get electronic indices to correct range
        # i.e. from 0-19 to 20-39
        running_corrects += torch.sum((electronique_preds[electronique_filter] + 20) == labels.data[electronique_filter])

    epoch_acc = running_corrects / len(valid_loader.dataset)

    print("Hier Acc: {:.4f}".format(epoch_acc))

    return epoch_acc

if __name__ == "__main__":

    models = dict()
    mlist = ["flat", "meuble", "electronique", "binary"]

    models["flat"] = "out/flat-false/flat-false-flat-model.pt"
    models["meuble"] = "out/meuble-false/meuble-false-meuble-model.pt"
    models["electronique"] = "out/electronique-false/electronique-false-electronique-model.pt"
    models["binary"] = "out/binary-false/binary-false-binary-model.pt"


    valid_loader, batch_size = load_data()

    classify_flat(torch.load(models["flat"]), valid_loader)

    classify_hier(
            torch.load(models["binary"]),
            torch.load(models["meuble"]),
            torch.load(models["electronique"]),
            valid_loader
        )
