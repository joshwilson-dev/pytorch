################
#### Header ####
################

# Title: Crop Dataset
# Author: Josh Wilson
# Date: 07-07-2022
# Description: 
# This script crops the dataset images and labels in the selected directory
# into the smallest equal size less than max crop size  

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
import numpy as np

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

max_crop_size = 650
save_dir = "./crops/"

# did the user select a dir or cancel?
if len(file_path_variable) > 0:
    # confirm dir with user
    check = messagebox.askquestion(
        "CONFIRM",
        "Are you sure you want to create a dataset from the files in:\n" + file_path_variable)
    if check =="yes":
        os.chdir(file_path_variable)
        # check if save directory exists and if no create one 
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # walk through image files and crop
        for file in os.listdir():
            if file.endswith(".JPG"):
                # check if image is labelled
                annotation_name = os.path.splitext(file)[0] + '.json'
                if os.path.exists(annotation_name):
                    print("Cropping", file)
                    image = Image.open(file, mode="r")
                    width, height = image.size
                    n_crops_width = math.ceil(width / max_crop_size)
                    n_crops_height = math.ceil(height / max_crop_size)
                    crop_width = width / n_crops_width
                    crop_height = height / n_crops_height

                    for i in range(0, n_crops_width):
                        for a in range(0, n_crops_height):
                            left = i * crop_width
                            right = (i + 1) * crop_width
                            top = a * crop_height
                            bottom = (a + 1) * crop_height
                            with open(annotation_name, 'r') as annotation:
                                data = json.load(annotation)
                                data["imageHeight"] = crop_height
                                data["imageWidth"] = crop_width
                                # copy labels in annotation file create boxes and labels
                                restart = TRUE
                                boxes_kept = 0
                                while restart == TRUE and boxes_kept != len(data["shapes"]):
                                    for b in range(boxes_kept, len(data["shapes"])):
                                        box_x1 = min(x[0] for x in data["shapes"][b]["points"])
                                        box_x2 = max(x[0] for x in data["shapes"][b]["points"])
                                        box_y1 = min(x[1] for x in data["shapes"][b]["points"])
                                        box_y2 = max(x[1] for x in data["shapes"][b]["points"])

                                        box_left = min(box_x1, box_x2)
                                        box_right = max(box_x1, box_x2)
                                        box_top = min(box_y1, box_y2)
                                        box_bottom = max(box_y1, box_y2)
                                        box_width = box_right - box_left
                                        box_height = box_bottom - box_top
                                        box_area = box_width * box_height

                                        if box_left > right or box_right < left or \
                                            box_top > bottom or box_bottom < top:
                                            del data["shapes"][b]
                                            restart = TRUE
                                            break
                                        else:
                                            if box_left < left: box_left = left
                                            if box_right > right: box_right = right
                                            if box_top < top: box_top = top
                                            if box_bottom > bottom: box_bottom = bottom
                                            new_box_width = box_right - box_left
                                            new_box_height = box_bottom - box_top
                                            new_box_area = new_box_width * new_box_height
                                            if new_box_area < box_area:
                                                # blackout part of image
                                                topleft = (round(box_left), round(box_top))
                                                black_box = Image.new("RGB", (round(new_box_width), round(new_box_height)))
                                                image.paste(black_box, topleft)
                                                del data["shapes"][b]
                                                restart = TRUE
                                                break
                                            else:
                                                for index in range(len(data["shapes"][b]["points"])):
                                                    data["shapes"][b]["points"][index][0] += - left
                                                    data["shapes"][b]["points"][index][1] += - top
                                                restart = FALSE
                                                boxes_kept += 1
                                if len(data["shapes"]) > 0:
                                    # load exif data
                                    exif_dict = piexif.load(image.info["exif"])
                                    exif_bytes = piexif.dump(exif_dict)
                                    image_crop = image.crop((left, top, right, bottom))
                                    md5hash = hashlib.md5(image_crop.tobytes()).hexdigest()
                                    image_crop = image_crop.save(save_dir + md5hash + ".JPG", exif = exif_bytes)
                                    annotation_output = save_dir + md5hash + ".json"
                                    data["imagePath"] = md5hash + ".JPG"
                                    with open(annotation_output, 'w') as new_annotation:
                                        json.dump(data, new_annotation, indent=2)