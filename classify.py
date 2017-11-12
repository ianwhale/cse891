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
    batch_size = 1
    cuda = False
    validation_split = 0.25

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

    print("Done.")

    num_classes = train_loader.dataset.num_categories

    model = torchvision.models.resnet18(pretrained=True)
    num_filters = model.fc.in_features
    model.fc = nn.Linear(num_filters, num_classes)  # Resize output layer.

    learning_rate = 1e-2
    momentum = 0.9
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    epochs = 10
    log_interval = 200

    print("Loading model to Cuda...")

    if cuda:
        criterion, model = criterion.cuda(), model.cuda()

    print("Done.")

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Code adapated from: http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    for epoch in range(epochs):
        model.train()
        correct = 0

        exp_lr_scheduler.step()

        for batch_idx, (data, target) in enumerate(train_loader):
            print("Allocating data.")

            if cuda:
                data, target = data.cuda(), target.cuda()

            data, target = Variable(data), Variable(target)

            print(data)
            print(target)

            optimizer.zero_grad()

            outputs = model(data)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, target)

            loss.backward()
            optimizer.step()

            correct += torch.sum(preds == target.data)

            if batch_idx % log_interval == 0:
                print("Train epoch: {}. [{} / {}] \t Loss: {:.6f}".format(epoch,
                    batch_idx * len(data), len(train_idx), loss.data[0]))

        print("\n Train epoch: {}. \t Accuracy: {:.6f}".format(epoch,
                        100. * correct / len(train_idx)))

        model.eval()
        correct = 0

        for batch_idx, (data, target) in enumerate(valid_loader):
            if cuda:
                data, target = data.cuda(), target.cuda()

            data, target = Variable(data), Variable(target)

            optimizer.zero_grad()

            outputs = model(data)
            _, pred = torch.max(outputs.data, 1)

            correct += torch.sum(preds == target.data)

        print("\n Validation epoch: {}. \t Accuracy: {:.6f}".format(epoch,
            100. * correct / len(valid_idx)))

if __name__ == "__main__":
    classify()

