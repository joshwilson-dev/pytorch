################
#### Header ####
################

# Title: Crop Dataset
# Author: Josh Wilson
# Date: 29-07-2022
# Description: 
# This script takes an object detection dataset and crops the images
# and annotations to create an image classification dataset 

###############
#### Setup ####
###############
import json
import os
from pickle import FALSE, TRUE
from PIL import Image
import math
import hashlib
import tkinter
from tkinter import filedialog
from tkinter import messagebox
import piexif

#################
#### Content ####
#################

# create function for user to select dir
root = tkinter.Tk()
root.withdraw()

def search_for_file_path ():
    currdir = os.getcwd()
    tempdir = filedialog.askdirectory(
        parent=root,
        initialdir=currdir,
        title='Please select a directory')
    if len(tempdir) > 0:
        print ("You chose: %s" % tempdir)
    return tempdir

file_path_variable = search_for_file_path()

save_dir = "./img_cls_dataset/"

# did the user select a dir or cancel?
if len(file_path_variable) > 0:
    # confirm dir with user
    check = messagebox.askquestion(
        "CONFIRM",
        "Are you sure you want to create a image classification dataset from the files in:\n" + file_path_variable)
    if check =="yes":
        os.chdir(file_path_variable)
        # check if save directory exists and if no create one 
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # walk through image files and crop
        dataset = {"images": [], "labels": []}
        for file in os.listdir():
            if file.endswith(".json"):
                # read annotation file
                annotation_file = open(file)
                image_file = os.path.splitext(file)[0] + '.JPG'
                annotations = json.load(annotation_file)
                image = Image.open(image_file, mode="r")
                for i in range(len(annotations["shapes"])):
                    label = annotations["shapes"][i]["label"]
                    taxa = label.split('-')
                    points = annotations["shapes"][i]["points"]
                    box_x1 = points[0][0]
                    box_x2 = points[1][0]
                    box_y1 = points[0][1]
                    box_y2 = points[1][1]

                    box_left = min(box_x1, box_x2)
                    box_right = max(box_x1, box_x2)
                    box_top = min(box_y1, box_y2)
                    box_bottom = max(box_y1, box_y2)

                    instance = image.crop((box_left, box_top, box_right, box_bottom))
                    exif_dict = piexif.load(image.info["exif"])
                    exif_bytes = piexif.dump(exif_dict)
                    instance = instance.save(save_dir + "instance.JPG", exif = exif_bytes)
                    hash_md5 = hashlib.md5()
                    with open(save_dir + "instance.JPG", "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_md5.update(chunk)
                    output = hash_md5.hexdigest()
                    image_name = output + ".JPG"
                    dataset['images'].append(image_name)
                    dataset['labels'].append(taxa)
                    image_output = save_dir + image_name
                    os.rename(os.path.join(save_dir + "instance.JPG"), image_output)
        with open(os.path.join(save_dir, 'dataset.json'), 'w') as file:
            json.dump(dataset, file, indent=2)