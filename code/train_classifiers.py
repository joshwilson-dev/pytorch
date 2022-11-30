# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
import time
import os
import copy
import json
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import PIL
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.model_selection import train_test_split

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Hierarchical Classification Training", add_help=add_help)

    parser.add_argument("--data-root", default="datasets/bird-mask-new/classifier_dataset/train/artificial", type=str, help="path to dir containing regions")
    parser.add_argument("--output-root", default="models/temp/", type=str, help="path to top level of regional dataset")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--batchsize", default=20, type=int, help="Batch size")
    parser.add_argument("--lr-steps", default=[1, 2, 3], nargs="+", type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)")
    parser.add_argument("--epochs", default=1000, type=int, help="Epochs")
    parser.add_argument('--patience', default = 2, type = int, help = "How many epochs without improvement before action?")
    parser.add_argument("--printfreq", default=5, type=int, help="After how many batches do you want to print the loss")
    return parser

def create_model(num_classes, device):
    model = models.resnet101(weights='ResNet101_Weights.DEFAULT')
    features = model.fc.in_features
    model.fc = nn.Linear(features, num_classes)
    model = model.to(device)
    return model

class CustomDataloader(Dataset):
    def __init__(self, image_paths, class_to_idx, labels, transform):
        self.transform = transform
        self.image_paths = image_paths
        self.labels = labels
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = PIL.Image.open(image_path).convert('RGB')
        label = torch.tensor(self.labels[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, label

def get_transform(train):
    trans = []
    if train:
        # trans.append(transforms.RandomResizedCrop(size = 224, scale = (0.70, 1.0), ratio = (1.0, 1.0)))
        trans.append(transforms.RandomHorizontalFlip(0.5))
        # trans.append(transforms.RandomRotation(degrees=(0, 360)))
    trans.append(transforms.ToTensor())
    trans.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return transforms.Compose(trans)

def train_model(model, output_path, train_loader, val_loader, train_size, val_size, device, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    writer = SummaryWriter(log_dir = output_path)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_since_improvement = 0
    lr_steps = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            progress = 0

            # Iterate over data.
            loader = train_loader if phase == "train" else val_loader
            dataset_size = train_size if phase == "train" else val_size
            for inputs, labels in loader:
                progress += 1
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if progress % args.printfreq == 0:
                    print("LR: {}   Image: {}/{}    Loss: {}".format(optimizer.param_groups[0]["lr"], progress * args.batchsize, dataset_size, running_loss/(progress * args.batchsize)))

            epoch_loss = running_loss / dataset_size
            # AP instead of accuracy
            epoch_acc = running_corrects.double() / dataset_size

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                writer.add_scalar("Acc", epoch_acc, epoch)
                if epoch_acc > best_acc:
                    print("The model improved this epoch")
                    best_acc = epoch_acc
                    epochs_since_improvement = 0
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), os.path.join(output_path, "model_best_state_dict.pth"))
                else:
                    epochs_since_improvement += 1
                    print("The model has not improved for {} epochs...".format(epochs_since_improvement))
        if epochs_since_improvement == args.patience:
            print("{} epochs without improvement...".format(args.patience))
            # load best model wts
            model.load_state_dict(best_model_wts)
            # decrease the learning rate, unless at last lr, then stop
            if lr_steps < len(args.lr_steps):
                print("Decreasing learning rate, step {}/{}".format(lr_steps, len(args.lr_steps)))
                scheduler.step()
                lr_steps += 1
                epochs_since_improvement = 0
            else:
                writer.flush()
                break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(output_path, "model_best_state_dict.pth"))
    return model

def main(args):
    device = torch.device(args.device)
    # walk through dataset directory and get levels
    for root, dirs, files in os.walk(args.data_root):
        for file in files:
            # if file.endswith("json"):
            if "Africa.json" in file:
                file_path = os.path.join(root, file)
                with open(file_path) as anns:
                    dataset = json.load(anns)
                if len(dataset["images"]) > 0:
                    # check if model directory exists, if not create it
                    region = os.path.splitext(file)[0]
                    output_path = os.path.join(args.output_root, region)
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    # create train and val datasets
                    image_paths = dataset["images"]
                    image_paths = [os.path.join(args.data_root, image_path) for image_path in image_paths]
                    labels = dataset["labels"]
                    # remove classes with only 1 occurance
                    classes, counts = np.unique(labels, return_counts=True)
                    classes = classes[np.where(counts > 1)]
                    labels, image_paths = zip(*[(a,b) for a,b in zip(labels,image_paths) if a in classes])
                    # save index_to_class for use when predicting
                    idx_to_class = {i:j for i, j in enumerate(classes)}
                    class_to_idx = {value:key for key,value in idx_to_class.items()}
                    labels = [class_to_idx[labels[i]] for i in range(len(labels))]
                    with open(os.path.join(output_path, 'index_to_class.json'), 'w') as f:
                        json.dump(idx_to_class, f, indent = 2)
                    # no point producing a model if there's only one class
                    if len(idx_to_class) > 1:
                        # split data into val and train
                        image_paths_train, image_paths_val, labels_train, labels_val = train_test_split(image_paths, labels, test_size=0.15, stratify=labels, random_state=42)

                        train_size = len(image_paths_train)
                        val_size = len(image_paths_val)

                        dataset_train = CustomDataloader(image_paths_train, class_to_idx, labels_train, get_transform(train=True))
                        dataset_val = CustomDataloader(image_paths_val, class_to_idx, labels_val, get_transform(train=False))

                        train_loader = DataLoader(dataset_train, batch_size=min(len(image_paths_train), args.batchsize))
                        val_loader = DataLoader(dataset_val, batch_size=min(len(image_paths_val), args.batchsize), shuffle=False)
                        # create model
                        num_classes = len(classes)
                        model = create_model(num_classes, device)
                        # train model
                        criterion = nn.CrossEntropyLoss()
                        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
                        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
                        print("Training {} classifier".format(region))
                        train_model(model, output_path, train_loader, val_loader, train_size, val_size, device, criterion, optimizer, scheduler, args.epochs)
                    
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)