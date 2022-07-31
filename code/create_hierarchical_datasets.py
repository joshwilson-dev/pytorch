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

# did the user select a dir or cancel?
if len(file_path_variable) > 0:
    # confirm dir with user
    check = messagebox.askquestion(
        "CONFIRM",
        "Are you sure you want to create a image classification dataset from the files in:\n" + file_path_variable)
    if check =="yes":
        os.chdir(file_path_variable)
        # walk through image files and crop
        for file in os.listdir():
            if file.endswith(".json"):
                # read annotation file
                annotation_file = open(file)
                image_file = os.path.splitext(file)[0] + '.JPG'
                annotations = json.load(annotation_file)
                image = Image.open(image_file, mode="r")
                for index1 in range(len(annotations["shapes"])):
                    # crop image to instance
                    points = annotations["shapes"][index1]["points"]
                    box_x1 = points[0][0]
                    box_x2 = points[1][0]
                    box_y1 = points[0][1]
                    box_y2 = points[1][1]
                    box_left = min(box_x1, box_x2)
                    box_right = max(box_x1, box_x2)
                    box_top = min(box_y1, box_y2)
                    box_bottom = max(box_y1, box_y2)
                    instance = image.crop((box_left, box_top, box_right, box_bottom))

                    # determine image hash
                    md5hash = hashlib.md5(instance.tobytes()).hexdigest()
                    image_name = md5hash + ".JPG"

                    # get image exif
                    exif_dict = piexif.load(image.info["exif"])
                    exif_bytes = piexif.dump(exif_dict)

                    # copy instance and annotation into relveant hierarchy dir
                    label = annotations["shapes"][index1]["label"]
                    levels = label.split('-') 
                    for index2 in range(1, len(levels)):
                        level_dir = levels[index2]
                        if level_dir != "unknown":
                            x = index2
                            while x > 1:
                                level_dir = os.path.join(levels[x - 1], level_dir)
                                x += -1
                            if not os.path.exists(level_dir):
                                os.makedirs(level_dir)
                            # save instance to top level, we only need annotation file
                            # in each hierarchical dirs, not actual images
                            if index2 == 1:
                                instance.save(os.path.join(level_dir, image_name), exif = exif_bytes)
                            # check if annotation file already exists otherwise create it
                            dataset_path = os.path.join(level_dir, "dataset.json")
                            if not os.path.exists(dataset_path):
                                instance_annotations = {"images": [], "labels": []}
                            else:
                                with open(dataset_path, 'r') as instance_annotation_file:
                                    instance_annotations = json.load(instance_annotation_file)
                            # add new annotation to annotation file
                            instance_annotations["images"].append(os.path.join("../" * (index2 - 1), image_name))
                            instance_annotations["labels"].append(levels[index2])
                            instance_annotations = json.dumps(instance_annotations, indent=2)
                            with open(dataset_path, "w") as instance_annotation_file:
                                instance_annotation_file.write(instance_annotations)
