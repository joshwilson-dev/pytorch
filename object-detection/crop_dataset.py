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

max_crop_size = 800
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
                    im = Image.open(file, mode="r")
                    width, height = im.size
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
                                restart = TRUE
                                boxes_kept = 0
                                while restart == TRUE and boxes_kept != len(data["shapes"]):
                                    for b in range(boxes_kept, len(data["shapes"])):
                                        box_x1 = data["shapes"][b]["points"][0][0]
                                        box_x2 = data["shapes"][b]["points"][1][0]
                                        box_y1 = data["shapes"][b]["points"][0][1]
                                        box_y2 = data["shapes"][b]["points"][1][1]

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
                                            if new_box_area < 0.3 * box_area:
                                                # blackout part of image
                                                topleft = (round(box_left), round(box_top))
                                                black_box = Image.new("RGB", (round(new_box_width), round(new_box_height)))
                                                im.paste(black_box, topleft)
                                                del data["shapes"][b]
                                                restart = TRUE
                                                break
                                            else:
                                                data["shapes"][b]["points"][0][0] = \
                                                    box_left - left
                                                data["shapes"][b]["points"][1][0] = \
                                                    box_right - left
                                                data["shapes"][b]["points"][0][1] = \
                                                    box_top - top
                                                data["shapes"][b]["points"][1][1] = \
                                                    box_bottom - top
                                                restart = FALSE
                                                boxes_kept += 1
                                if len(data["shapes"]) > 0:
                                    # load exif data
                                    exif_dict = piexif.load(im.info["exif"])
                                    exif_bytes = piexif.dump(exif_dict)
                                    img1 = im.crop((left, top, right, bottom))
                                    img1 = img1.save(save_dir + "img1.JPG", exif = exif_bytes)

                                    hash_md5 = hashlib.md5()
                                    with open(save_dir + "img1.JPG", "rb") as f:
                                        for chunk in iter(lambda: f.read(4096), b""):
                                            hash_md5.update(chunk)
                                    output = hash_md5.hexdigest()
                                    image_output = save_dir + output + ".JPG"
                                    os.rename(save_dir + "img1.JPG", image_output)
                                    annotation_output = save_dir + output + ".json"
                                    data["imagePath"] = output + ".JPG"
                                    with open(annotation_output, 'w') as new_annotation:
                                        json.dump(data, new_annotation, indent=2)