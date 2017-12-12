#!/usr/bin/python3

# This script loads logs from the training process to make graphs of train/test accuracy vs epoch and train/test loss vs epoch.
# These graphs are spit out into the out directory.

import scipy.io as sio
import sys
import numpy as np
import json, codecs
import os
import matplotlib.pyplot as plt

def plot_training(datas, labels, ylab, legend, title, slug, mul, val_train):

    # draw and save loss vs epoch graph
    fig, ax = plt.subplots(nrows=1, ncols=1)

    for dat, lab in zip(datas, labels):

        dat=dat[val_train::2]
        p2 = ax.plot(
                np.array(range(len(dat)))*mul,
                dat,
                '-',
                label=lab
            )

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

    ax.set_xlabel('Training Epoch')
    ax.set_ylabel(ylab)
    ax.set_title(title)

    os.makedirs('out', exist_ok=True)
    fig.savefig('out/' + slug + '.pdf')
    plt.close(fig)

dat_binary_false_accs = json.load(open("out/binary-false/binary-false-binary-accs.json"))
dat_binary_false_losses = json.load(open("out/binary-false/binary-false-binary-losses.json"))

dat_flat_false_accs = json.load(open("out/flat-false/flat-false-flat-accs.json"))
dat_flat_false_losses = json.load(open("out/flat-false/flat-false-flat-losses.json"))

dat_meuble_false_accs = json.load(open("out/meuble-false/meuble-false-meuble-accs.json"))
dat_meuble_false_losses = json.load(open("out/meuble-false/meuble-false-meuble-losses.json"))

dat_electronique_false_accs = json.load(open("out/electronique-false/electronique-false-electronique-accs.json"))
dat_electronique_false_losses = json.load(open("out/electronique-false/electronique-false-electronique-losses.json"))

dat_binary_true_accs = json.load(open("out/binary-true/binary-true-binary-accs.json"))
dat_binary_true_losses = json.load(open("out/binary-true/binary-true-binary-losses.json"))

dat_flat_true_accs = json.load(open("out/flat-true/flat-true-flat-accs.json"))
dat_flat_true_losses = json.load(open("out/flat-true/flat-true-flat-losses.json"))

dat_meuble_true_accs = json.load(open("out/meuble-true/meuble-true-meuble-accs.json"))
dat_meuble_true_losses = json.load(open("out/meuble-true/meuble-true-meuble-losses.json"))

dat_electronique_true_accs = json.load(open("out/electronique-true/electronique-true-electronique-accs.json"))
dat_electronique_true_losses = json.load(open("out/electronique-true/electronique-true-electronique-losses.json"))


plot_training([dat_binary_false_accs, dat_flat_false_accs, dat_meuble_false_accs, dat_electronique_false_accs], ["Binary", "Flat", "Furniture", "Electronics"], "Accuracy", True, "Model Validation Accuracy During Training Without Fine Tuning", "false_accs_val", 1, 1)

plot_training([dat_binary_true_accs, dat_flat_true_accs, dat_meuble_true_accs, dat_electronique_true_accs], ["Binary", "Flat", "Furniture", "Electronics"], "Accuracy", True, "Model Validation Accuracy During Training With Fine Tuning", "true_accs_val", 1, 1)

plot_training([dat_binary_false_losses, dat_flat_false_losses, dat_meuble_false_losses, dat_electronique_false_losses], ["Binary", "Flat", "Furniture", "Electronics"], "Loss", True, "Model Validation Loss During Training Without Fine Tuning", "false_losses_val", 1, 1)

plot_training([dat_binary_true_losses, dat_flat_true_losses, dat_meuble_true_losses, dat_electronique_true_losses], ["Binary", "Flat", "Furniture", "Electronics"], "Loss", True, "Model Validation Loss During Training With Fine Tuning", "true_losses_val", 1, 1)

plot_training([dat_binary_false_accs, dat_flat_false_accs, dat_meuble_false_accs, dat_electronique_false_accs], ["Binary", "Flat", "Furniture", "Electronics"], "Accuracy", True, "Model Training Accuracy During Training Without Fine Tuning", "false_accs_train", 1, 0)

plot_training([dat_binary_true_accs, dat_flat_true_accs, dat_meuble_true_accs, dat_electronique_true_accs], ["Binary", "Flat", "Furniture", "Electronics"], "Accuracy", True, "Model Training Accuracy During Training With Fine Tuning", "true_accs_train", 1, 0)

plot_training([dat_binary_false_losses, dat_flat_false_losses, dat_meuble_false_losses, dat_electronique_false_losses], ["Binary", "Flat", "Furniture", "Electronics"], "Loss", True, "Model Training Loss During Training Without Fine Tuning", "false_losses_train", 1, 0)

plot_training([dat_binary_true_losses, dat_flat_true_losses, dat_meuble_true_losses, dat_electronique_true_losses], ["Binary", "Flat", "Furniture", "Electronics"], "Loss", True, "Model Training Loss During Training With Fine Tuning", "true_losses_train", 1, 0)
