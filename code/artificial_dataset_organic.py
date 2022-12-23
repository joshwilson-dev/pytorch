################
#### Header ####
################

# Title: curate data within sub directory
# Author: Josh Wilson
# Date: 02-06-2022
# Description: 
# This script runs through sub-dirs of the selected directory
# looking for mask annotations and creates a csv with the gsd
# of each mask instance

###############
#### Setup ####
###############

import os
import math
import tkinter
from tkinter import filedialog
from tkinter import messagebox
import json
import piexif
import pandas as pd
import random
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import hashlib
import torchvision.transforms as T
import shutil
from ortools.sat.python import cp_model
import sys
import imagehash

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

def blackout_instance(image, box):
    box_width = box[2][0] - box[0][0]
    box_height = box[2][1] - box[0][1]
    topleft = (round(box[0][0]), round(box[0][1]))
    black_box = Image.new("RGB", (round(box_width), round(box_height)))
    image.paste(black_box, topleft)
    return image

file_path_variable = search_for_file_path()
# did the user select a dir or cancel?
if len(file_path_variable) > 0:
    # confirm dir with user
    check = messagebox.askquestion(
        "CONFIRM",
        "Are you sure you want to create a dataset from the files in:\n" + file_path_variable)
    if check =="yes":
        os.chdir(file_path_variable)
        # specify gsd categories
        gsd_cats = ["error", "fine"]
        gsd_bins = [0.0025, 0.0075]
        paths = ["boop"]
        for path in paths:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
        # iterate through images
        for root, dirs, files in os.walk(os.getcwd()):
            for file in files:
                if "boop" not in root:
                    if file.endswith(".JPG"):
                        # get image and annotation name and path
                        image_name = file
                        annotation_name = os.path.splitext(file)[0] + '.json'
                        image_path = os.path.join(root, image_name)
                        annotation_path = os.path.join(root, annotation_name)

                        # open the image
                        original_image = Image.open(image_path)

                        # read exif data
                        exif_dict = piexif.load(original_image.info['exif'])
                        comments = json.loads("".join(map(chr, [i for i in exif_dict["0th"][piexif.ImageIFD.XPComment] if i != 0])))
                        comments["original_image"] = file
                        exif_dict["0th"][piexif.ImageIFD.XPComment] = json.dumps(comments).encode('utf-16le')
                        exif_bytes = piexif.dump(exif_dict)

                        # get gsd
                        try:
                            gsd = float(comments["gsd"])
                            substrate = comments["ecosystem typology"]
                        except:
                            print("Incorrect comment data: ", os.path.join(root, file))
                            continue

                        try:
                            gsd_cat = gsd_cats[np.digitize(gsd,gsd_bins)]
                            if gsd_cat == "error":
                                raise ValueError('GSD too low: ', file)
                        except:
                            print('GSD out of range: ', file)
                            continue
                        # determine image patch locations
                        width, height = original_image.size
                        # rescale image to average gsd, this makes all
                        # objects and birds a consistant size.
                        target_gsd = (gsd_bins[0] + gsd_bins[-1])/2

                        scale = target_gsd/gsd
                        # scale original image to median gsd
                        width = int(width * scale)
                        height = int(height * scale)
                        original_image = original_image.resize((width, height))

                        patchsize = 800
                        n_crops_width = math.ceil(width / patchsize)
                        n_crops_height = math.ceil(height / patchsize)
                        padded_width = n_crops_width * patchsize
                        padded_height = n_crops_height * patchsize
                        pad_width = (padded_width - width) / 2
                        pad_height = (padded_height - height) / 2
                        print("Recording Patch Data From: ", file)
                        patch_id = 0
                        instance_paths = []
                        instance_masks = []
                        patch_masks = []
                        num = 0
                        for height_index in range(n_crops_height):
                            for width_index in range(n_crops_width):
                                left = width_index * patchsize - pad_width
                                right = left + patchsize
                                top = height_index * patchsize - pad_height
                                bottom = top + patchsize
                                patch_points = (left, top, right, bottom)
                                patch = original_image.crop(patch_points)
                                shapes = []
                                if os.path.exists(annotation_path):
                                    annotation = json.load(open(annotation_path))
                                    for shape in annotation["shapes"]:
                                        # get the box
                                        points_x, points_y = map(list, zip(*shape["points"]))
                                        # adjust box and points to patch coordinates
                                        points_x = [x * scale - left for x in points_x]
                                        points_y = [y * scale - top for y in points_y]
                                        box_xmin = min(points_x)
                                        box_xmax = max(points_x)
                                        box_ymin = min(points_y)
                                        box_ymax = max(points_y)
                                        # check if box is outside patch
                                        if box_xmin > patchsize or box_xmax < 0 or box_ymin > patchsize or box_ymax < 0:
                                            continue
                                        # calculate box area
                                        box_area = (box_xmax - box_xmin) * (box_ymax - box_ymin)
                                        if box_xmin < 0: box_xmin = 0
                                        if box_xmax > patchsize: box_xmax = patchsize
                                        if box_ymin < 0: box_ymin = 0
                                        if box_ymax > patchsize: box_ymax = patchsize
                                        # save box
                                        box = [
                                            [box_xmin, box_ymin],
                                            [box_xmax, box_ymin],
                                            [box_xmax, box_ymax],
                                            [box_xmin, box_ymax]]
                                        # calculate new box area
                                        if box_area - (box_xmax - box_xmin) * (box_ymax - box_ymin) != 0:
                                            blackout_instance(patch, box)
                                            continue

                                        # rectangle or unknown black out
                                        if "unknown" in shape["label"] or "rectangle" in shape["shape_type"]:
                                            blackout_instance(patch, box)
                                        # shadow drop, label the rest
                                        elif "shadow" not in shape["label"]:
                                            shapes.append({
                                                "label": shape["label"],
                                                "points": box,
                                                "group_id": 'null',
                                                "shape_type": 'polygon',
                                                "flags": {}})
                                md5hash = hashlib.md5(patch.tobytes()).hexdigest()
                                patch_id = md5hash + ".JPG"
                                # save patch
                                patch.save(os.path.join("boop", patch_id), exif = exif_bytes)
                                # save annotation
                                annotation_id = md5hash + '.json'
                                annotation = {
                                    "version": "5.0.1",
                                    "flags": {},
                                    "shapes": shapes,
                                    "imagePath": patch_id,
                                    "imageData": 'null',
                                    "imageHeight": 800,
                                    "imageWidth": 800}
                                annotation_str = json.dumps(annotation, indent = 2).replace('"null"', 'null')
                                with open(os.path.join("boop", annotation_id), 'w') as annotation_file:
                                    annotation_file.write(annotation_str)