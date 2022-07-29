'''Pytorch dataset loading script.
'''

import os
import cv2
from PIL import Image
from torch.utils.data import Dataset
from level_dict import hierarchy
from helper import read_meta
import json


class LoadDataset(Dataset):
    '''Reads the given csv file and loads the data.
    '''

    def __init__(self, root, image_size=32, image_depth=3, return_label=True, transform=None):
        '''Init param.
        '''
        self.root = root

        with open(os.path.join(self.root,"dataset.json")) as anns:
            self.dictionary = json.load(anns)
        
        self.image_size = image_size
        self.image_depth = image_depth
        self.return_label = return_label
        self.transform = transform

    def __len__(self):
        '''Returns the total amount of data.
        '''
        return len(self.dictionary)

    def __getitem__(self, idx):
        '''Returns a single item.
        '''
        image_path, image, order, species = None, None, None, None
        if self.return_label:
            image_path = self.dictionary["images"][idx]
            order = self.dictionary["labels"][idx][2]
            species  = self.dictionary["labels"][idx][5]
        else:
            image_path = self.dictionary["images"][idx]

        if self.image_depth == 1:
            image = cv2.imread(image_path, 0)
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.image_size != 32:
            cv2.resize(image, (self.image_size, self.image_size))


        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        if self.return_label:
            return {
                'image':image/255.0,
                'label_1': order,
                'label_2': species
            }
        else:
            return {
                'image':image
            }