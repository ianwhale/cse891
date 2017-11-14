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

def get_split_indices(n, split):
    idx = [i for i in range(n)]
    split = int(n * split)

    shuffle(idx)

    return idx[0: n - split], idx[n - split:]


def classify():
    batch_size = 256
    cuda = True
    validation_split = 0.25
    fine_tune = False

    print("Constructing train_loader...")

    dataset = CDiscountDataSet('train.bson',
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.Scale(224),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                    ]),
                category_level=1,
                trim_classes=50000
            )

    train_idx, valid_idx = get_split_indices(len(dataset), validation_split)

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)
    valid_loader = DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size)

    dataloaders = {
            "train": train_loader,
            "val": valid_loader
    }

    print("Done.")

    num_classes = train_loader.dataset.num_categories

    model = torchvision.models.resnet152(pretrained=True)

    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False

    num_filters = model.fc.in_features
    model.fc = nn.Linear(num_filters, num_classes)  # Resize output layer.

    learning_rate = 5e-2
    momentum = 0.9
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=learning_rate, momentum=momentum)

    epochs = 10
    log_interval = 200

    print("Loading model to Cuda...")

    if cuda:
        criterion, model = criterion.cuda(), model.cuda()

    print("Done.")

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    print("Output classes: {}".format(train_loader.dataset.num_categories))

    # Code adapated from: http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0

    log_interval = 100 # Report stats every 100 minibatches.

    for epoch in range(epochs):
        print("Epoch {} / {}".format(epoch, epochs - 1))
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                exp_lr_scheduler.step()
                model.train(True)

            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            counter = 0

            for data in dataloaders[phase]:
                inputs, labels = data

                if cuda:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())

                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

                if phase == "train" and counter != 0 and counter % log_interval == 0:
                    print("Training loss at iteration {}: {:.4f}".format(counter,
                        running_loss / (counter * batch_size)))
                    print("Training accuracy at iteration {}: {:.4f}".format(counter,
                        running_corrects / (counter * batch_size)))


                counter += 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            print("{} Loss: {:.4f}  Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "val" and epoch_acc > best_acc:
                bast_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since

    print("Training completed in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))
    print("Best validation accuracy: {:4f}".format(best_acc))


if __name__ == "__main__":
    classify()

