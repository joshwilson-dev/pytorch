# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import time
import os
import copy
import json
from torch.utils.data import Dataset
import PIL
from torch.utils.data import DataLoader

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Hierarchical Classification Training", add_help=add_help)

    parser.add_argument("--data-root", default="../../datasets/bird-classifier/", type=str, help="path to dir containing ")
    parser.add_argument("--output-root", default="../../models/temp/", type=str, help="path to top level of hierarchical dataset")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--batchsize", default=20, type=int, help="Batch size")
    parser.add_argument("--epochs", default=5, type=int, help="Epochs")
    parser.add_argument("--printfreq", default=5, type=int, help="After how many batches do you want to print the loss")
    return parser

def train_model(model, train_loader, val_loader, train_size, val_size, device, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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
                    print("Image: {}/{}    Loss: {}".format(progress, dataset_size / args.batchsize, running_loss/(progress * args.batchsize)))
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def create_model(num_classes, device):
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    features = model.fc.in_features
    model.fc = nn.Linear(features, num_classes)
    model = model.to(device)
    return model

class CustomDataloader(Dataset):
    def __init__(self, image_paths, class_to_idx, labels, transform):
        self.transform = transform
        self.image_paths = image_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = PIL.Image.open(image_path).convert('RGB')
        label = torch.tensor(self.class_to_idx[self.labels[index]])
        if self.transform is not None:
            image = self.transform(image)
        return image, label

def get_transform(train):
    trans = []
    if train:
        trans.append(transforms.RandomResizedCrop(224))
        trans.append(transforms.RandomHorizontalFlip(0.5))
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    else:
        trans.append(transforms.Resize(256))
        trans.append(transforms.CenterCrop(224))
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return transforms.Compose(trans)

def main(args):
    device = torch.device(args.device)
    # walk through dataset directory and get levels
    for classifier in next(os.walk(args.data_root))[1]:
        label_path = os.path.join(args.data_root, classifier, "dataset.json")
        with open(label_path) as anns:
            dataset = json.load(anns)
        # create train and val datasets
        image_paths = dataset["images"]
        image_paths = [os.path.join(args.data_root, "images", image_path) for image_path in image_paths]
        labels = dataset["labels"]
        classes = list(set(labels))
        idx_to_class = {i:j for i, j in enumerate(classes)}
        class_to_idx = {value:key for key,value in idx_to_class.items()}
        # save idx_to_class fo use when predicting
        output_path = args.output_root + classifier
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        with open(os.path.join(output_path, 'index_to_class.txt'), 'w') as f:
            json.dump(idx_to_class, f)
        dataset_train = CustomDataloader(image_paths, class_to_idx, labels, get_transform(train=True))
        dataset_val = CustomDataloader(image_paths, class_to_idx, labels, get_transform(train=False))
        train_size = int(0.85 * len(dataset_train))
        val_size = len(dataset_train) - train_size
        train_dataset, _ = torch.utils.data.random_split(dataset_train, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        _, val_dataset = torch.utils.data.random_split(dataset_val, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        num_classes = len(list(set(dataset["labels"])))
        model = create_model(num_classes, device)
        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        print("Training {} detector".format(classifier))
        trained_model = train_model(model, train_loader, val_loader, train_size, val_size, device, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=args.epochs)
        torch.save(trained_model.state_dict(), os.path.join(output_path, "model_final_state_dict.pth"))

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)