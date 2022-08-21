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
import numpy

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

def background_crop(image, boxes, size = 100, min_scale = 0.5, max_scale = 2.0, min_aspect_ratio = 0.5, max_aspect_ratio = 2.0, sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], trials = 40):
        orig_w, orig_h = image.size
        while True:
            for _ in range(trials):
                # check the aspect ratio limitations
                r = min_scale + (max_scale - min_scale) * numpy.random.rand(2)
                new_w = int(size * r[0])
                new_h = int(size * r[1])
                aspect_ratio = new_w / new_h
                if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
                    continue
                # check for 0 area crops
                r = numpy.random.rand(2)
                left = int((orig_w - new_w) * r[0])
                top = int((orig_h - new_h) * r[1])
                right = left + new_w
                bottom = top + new_h
                if left == right or top == bottom:
                    continue
                # check for any valid boxes with corners within the crop area
                for index1 in [0, 2]:
                    for index2 in [1, 3]:
                        cx = numpy.array([item[index1] for item in boxes])
                        cy = numpy.array([item[index2] for item in boxes])
                        is_within_crop_area = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                        if is_within_crop_area.any():
                            continue
                background = image.crop((left, top, left + new_w, top + new_h))
                return background

# did the user select a dir or cancel?
if len(file_path_variable) > 0:
    # confirm dir with user
    check = messagebox.askquestion(
        "CONFIRM",
        "Are you sure you want to create an hierarchical image classification dataset from the files in:\n" + file_path_variable)
    if check =="yes":
        os.chdir(file_path_variable)
        root = "hierarchical"
        levels = ["order", "family", "genus", "species", "age"]
        labels_dict = {}
        for index in range(len(levels)):
            labels_dict[index] = {"images": [], "labels": []}
        if not os.path.exists(root):
            os.makedirs(root)
        if not os.path.exists(os.path.join(root, "images")):
            os.makedirs(os.path.join(root, "images"))
        for level in levels:
            if not os.path.exists(os.path.join(root, level)):
                os.makedirs(os.path.join(root, level))
        # walk through image files and crop
        for file in os.listdir():
            if file.endswith(".json") and file != "dataset.json":
                print("Creating hierarchcial classifier data for", file)
                # read annotation file
                annotation_file = open(file)
                image_file = os.path.splitext(file)[0] + '.JPG'
                annotations = json.load(annotation_file)
                image = Image.open(image_file, mode="r")
                boxes = []
                for shape in annotations["shapes"]:
                    # crop image to instance
                    box_x1 = shape["points"][0][0]
                    box_x2 = shape["points"][1][0]
                    box_y1 = shape["points"][0][1]
                    box_y2 = shape["points"][1][1]
                    box_left = min(box_x1, box_x2)
                    box_right = max(box_x1, box_x2)
                    box_top = min(box_y1, box_y2)
                    box_bottom = max(box_y1, box_y2)
                    boxes.append([box_left, box_top, box_right, box_bottom])
                    instance = image.crop((box_left, box_top, box_right, box_bottom))

                    # determine image hash
                    md5hash = hashlib.md5(instance.tobytes()).hexdigest()
                    image_name = md5hash + ".JPG"

                    # get image exif
                    exif_dict = piexif.load(image.info["exif"])
                    exif_bytes = piexif.dump(exif_dict)
                    #TODO update exif image size

                    # copy instances into location
                    instance.save(os.path.join(root, "images", image_name), exif=exif_bytes)

                    # copy instance and annotation into relveant hierarchy dir
                    for level_index in range(len(levels)):
                        label = shape["label"].split("_")[level_index + 2]
                        if label != "unknown":
                            # add new annotation to annotation file
                            labels_dict[level_index]["images"].append(image_name)
                            labels_dict[level_index]["labels"].append(label)
                # take a background crop from each image and save to every level
                for _ in range(5):
                    background = background_crop(image, boxes)
                    md5hash = hashlib.md5(background.tobytes()).hexdigest()
                    background_name = md5hash + ".JPG"
                    exif_dict = piexif.load(background.info["exif"])
                    exif_bytes = piexif.dump(exif_dict)
                    for level_index in range(len(levels)):
                        background.save(os.path.join(root, "images", background_name), exif=exif_bytes)
                        labels_dict[level_index]["images"].append(background_name)
                        labels_dict[level_index]["labels"].append("background")

        for level_index in range(len(levels)):
            dataset_path = os.path.join(root, levels[level_index], "dataset.json")
            instance_annotations = json.dumps(labels_dict[level_index], indent=2)
            with open(dataset_path, "w") as instance_annotation_file:
                instance_annotation_file.write(instance_annotations)