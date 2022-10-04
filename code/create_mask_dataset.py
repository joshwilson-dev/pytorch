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
import csv
import math
import tkinter
from tkinter import filedialog
from tkinter import messagebox
import json
from requests import get
from pandas import json_normalize
import piexif
import pandas as pd
import random
from PIL import Image, ImageDraw, ImageEnhance
import numpy 
import hashlib
import torchvision.transforms as T
import shutil

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

def crop_mask(im, points, label, gsd_cat):
    polygon = [tuple(l) for l in points]
    pad = 100
    # find bounding box
    max_x = max([t[0] for t in polygon])
    min_x = min([t[0] for t in polygon])
    max_y = max([t[1] for t in polygon])
    min_y = min([t[1] for t in polygon])
    # subtract xmin andymin from each point
    origin = [(min_x - pad, min_y - pad)] * len(polygon)
    polygon = tuple(tuple(a - b for a, b in zip(tup1, tup2))\
        for tup1, tup2 in zip(polygon, origin))
    # crop to bounding box
    bird = im.crop((min_x - pad, min_y - pad, max_x + pad, max_y + pad))
    # convert to numpy (for convenience)
    imArray = numpy.asarray(bird)
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
    mask = numpy.array(maskIm)
    # assemble new image (uint8: 0-255)
    newImArray = numpy.empty(imArray.shape,dtype='uint8')
    # colors (three first columns, RGB)
    newImArray[:,:,:3] = imArray[:,:,:3]
    # transparency (4th column)
    newImArray[:,:,3] = mask*255
    # back to Image from numpy
    newIm = Image.fromarray(newImArray, "RGBA")
    # calculate md5#
    md5hash = hashlib.md5(newIm.tobytes()).hexdigest()
    # save image
    # if path doesn't exist, create it
    path = os.path.join("instances", label, gsd_cat)
    if not os.path.exists(path):
        os.makedirs(path)
    name = md5hash + ".png"
    newIm.save(os.path.join(path, name), format = "png")
    return [name, polygon]

def balance_resolutions(category, labels, df, min_instances):
    for label in labels:
        for gsd_cat in reversed(gsd_cats):
            count = df[df['label'] == label]
            count = count[count['gsd_cat'] == gsd_cat]
            count = len(count)
            # if gsd does not have at least min_instances instances per gsd_cat:
            # re-sample if very fine
            # otherwise downscale images from high res
            if count < min_instances:
                min_gsd = gsd_bins[gsd_cats.index(gsd_cat)]
                max_gsd = gsd_bins[gsd_cats.index(gsd_cat) + 1]
                high_res = df[df['label'] == label]
                high_res = high_res[high_res['gsd'] <= max_gsd]
                downscale = high_res.sample(n=min_instances - count, replace=True)
                # downscale['randnum'] = [x/10000 for x in random.sample(range(int(min_gsd * 10000), int(max_gsd * 10000)), downscale.shape[0])]
                # downscale = downscale.assign(downscale=lambda df: df.randnum/df.gsd)
                # downscale = downscale.drop("randnum", axis=1)
                # downscale = downscale.reset_index(drop=True)
                for index, row in downscale.iterrows():
                    old_path = os.path.join(category, row["label"], row["gsd_cat"], row["id"])
                    new_path = os.path.join(category, row["label"], gsd_cat)
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)
                    shutil.copy(old_path, os.path.join(new_path, row["id"]))
                    # path = os.path.join(category, row["label"], row["gsd_cat"])
                    # instance = Image.open(os.path.join(path, row["id"])).convert("RGBA")
                    # size = min(instance.size) / row["downscale"]
                    # instance = T.Resize(size=int(size))(instance)
                    # # calculate md5#
                    # md5hash = hashlib.md5(instance.tobytes()).hexdigest()
                    # # save image
                    # path = os.path.join(category, row["label"], gsd_cat)
                    # if not os.path.exists(path):
                    #     os.makedirs(path)
                    # name = md5hash + ".png"
                    # instance.save(os.path.join(path, name))
                    # # resize points
                    # if category == "instances":
                    #     points = row["points"]
                    #     points = tuple(tuple(item / row["downscale"] for item in point) for point in points)
                    #     downscale.at[index,'points'] = points
                    # downscale.at[index,'id'] = name
                downscale = downscale.assign(gsd_cat=gsd_cat)
                df = pd.concat([df, downscale])
            # if gsd has more than instances_per_species instances drop it down to instances_per_species
            elif count > min_instances:
                drop = df[df['label'] == label]
                drop = drop[drop['gsd_cat'] == gsd_cat]
                drop = drop.sample(n=count - min_instances)
                for index, row in drop:
                    os.remove(os.path.join(category, row["label"], gsd_cat, row["id"]))
                df.drop(drop.index)
    return df

def is_float(string):
    try:
        string = float(string)
        return string
    except: return string

def rotate_point(point, centre, deg):
    rotated_point = (
        centre[0] + (point[0]-centre[0])*math.cos(math.radians(deg)) - (point[1]-centre[1])*math.sin(math.radians(deg)),
        centre[1] + (point[0]-centre[0])*math.sin(math.radians(deg)) + (point[1]-centre[1])*math.cos(math.radians(deg)))
    return rotated_point

def transforms(instance, row, points=None):
    # scale
    min_gsd = gsd_bins[gsd_cats.index(row["instance_gsd_cat"])]
    max_gsd = gsd_bins[gsd_cats.index(row["instance_gsd_cat"]) + 1]
    scale = random.uniform(min_gsd, max_gsd)/row["instance_gsd"]
    size = min(instance.size) / scale
    instance = T.Resize(size=int(size))(instance)
    # colour
    colour = random.uniform(0.9, 1.1)
    instance = ImageEnhance.Color(instance)
    instance = instance.enhance(colour)
    contrast = random.uniform(0.9, 1.1)
    instance = ImageEnhance.Contrast(instance)
    instance = instance.enhance(contrast)
    brightness = random.uniform(0.9, 1.1)
    instance = ImageEnhance.Brightness(instance)
    instance = instance.enhance(brightness)
    # points
    if points != None:
        # rotate
        rotation = random.randint(0, 360)
        instance = T.RandomRotation((rotation, rotation))(instance)
        points = tuple(rotate_point(point, centre, -rotation) for point in points)
        points = tuple(tuple(item / scale for item in point) for point in points)
    return instance, points

file_path_variable = search_for_file_path()

# did the user select a dir or cancel?
if len(file_path_variable) > 0:
    # confirm dir with user
    check = messagebox.askquestion(
        "CONFIRM",
        "Are you sure you want to curate the files in:\n" + file_path_variable)
    if check =="yes":
        os.chdir(file_path_variable)
        # iterate through files in dir
        header = False
        instances = {"label": [], "gsd": [], "gsd_cat": [], "id": [], "points": [], "downscale": []}
        backgrounds = {"label": [], "gsd": [], "gsd_cat": [], "id": [], "downscale": []}
        gsd_cats = ["very fine", "fine", "coarse", "very coarse"]
        gsd_bins = [0.004, 0.007, 0.01, 0.013, 0.016]
        paths = ["instances", "substrates", "dataset"]
        for path in paths:
            if os.path.exists(path):
                shutil.rmtree(path)
        for root, dirs, files in os.walk(os.getcwd()):
            instance_id = 0
            for file in files:
                if "mask" in root:
                    if file.endswith(".json"):
                        # load annotation
                        annotation_path = os.path.join(root, file)
                        annotation = json.load(open(annotation_path))
                        # load image
                        image_name = annotation["imagePath"]
                        image_path = os.path.join(root, image_name)
                        image = Image.open(image_path).convert("RGBA")
                        image_width, image_height = image.size
                        # read exif data
                        exif_dict = piexif.load(image.info['exif'])
                        # get gsd
                        comments = json.loads("".join(map(chr, [i for i in exif_dict["0th"][piexif.ImageIFD.XPComment] if i != 0])))
                        gsd = is_float(comments["gsd"])
                        try: gsd_cat = gsd_cats[numpy.digitize(gsd,gsd_bins) - 1]
                        except: gsd_cat = numpy.nan
                        # add to dictionary
                        for instance in annotation["shapes"]:
                            instance_id, points = crop_mask(image.convert("RGBA"), instance["points"], instance["label"], gsd_cat)
                            instances["label"].append(instance["label"])
                            instances["gsd"].append(gsd)
                            instances["gsd_cat"].append(gsd_cat)
                            instances["id"].append(instance_id)
                            instances["points"].append(points)
                            instances["downscale"].append(1)

                if "backgrounds" in root:
                    image_name = file
                    image_path = os.path.join(root, image_name)
                    image = Image.open(image_path).convert("RGBA")
                    image_width, image_height = image.size
                    # read exif data
                    exif_dict = piexif.load(image.info['exif'])
                    # get gsd
                    comments = json.loads("".join(map(chr, [i for i in exif_dict["0th"][piexif.ImageIFD.XPComment] if i != 0])))
                    gsd = is_float(comments["gsd"])
                    try: gsd_cat = gsd_cats[numpy.digitize(gsd,gsd_bins) - 1]
                    except: gsd_cat = numpy.nan
                    # calculate md5#
                    md5hash = hashlib.md5(image.tobytes()).hexdigest()
                    # save image
                    path = os.path.join("substrates", os.path.basename(root), gsd_cat)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    name =  md5hash + ".png"
                    shutil.copy(image_path, os.path.join(path, name))
                    # add to dictionary
                    backgrounds["label"].append(os.path.basename(root))
                    backgrounds["gsd"].append(gsd)
                    backgrounds["gsd_cat"].append(gsd_cat)
                    backgrounds["id"].append(name)
                    backgrounds["downscale"].append(1)
        # convert dictionary to df
        instances = pd.DataFrame(data=instances)
        backgrounds = pd.DataFrame(data=backgrounds)
        # remove masks with unsuitable gsd values
        instances = instances.dropna()
        backgrounds = backgrounds.dropna()
        # check that each species has enough instances in each gsd category
        instances = balance_resolutions(category = "instances", labels = instances.label.unique(), df = instances, min_instances = 2)
        # check that each background has enough instances in each gsd category
        backgrounds = balance_resolutions(category = "substrates", labels = backgrounds.label.unique(), df = backgrounds, min_instances = 2)
        # place 1 instance of each species on each of the 2 background
        # images for each of 1 background types the until all 2
        # instances of each species have been placed 1 times
        # 2 * 1 * 2 = 4 images per gsd 16 total
        # add common column to merge by
        backgrounds = backgrounds.add_prefix('background_')
        instances = instances.add_prefix('instance_')
        instances["temp"] = 1
        backgrounds["temp"] = 1
        dataset = pd.DataFrame()
        # want 5 copies of each bird
        for repeats in range(5):
            instances = instances.sample(n=len(instances))
            backgrounds = backgrounds.sample(n=len(backgrounds))
            rep = instances.merge(backgrounds, on='temp').drop('temp', axis=1)
            rep = rep[rep["instance_gsd_cat"] == rep["background_gsd_cat"]]
            rep = rep.sort_values(by=['background_id'])
            dataset = pd.concat([dataset, rep])
        dataset.to_csv("dataset.csv")
        # actually create data
        prev_background_id = "none"
        path = "dataset"
        if not os.path.exists(path):
            os.makedirs(path)
        dataset = dataset.reset_index(drop=True)
        for index, row in dataset.iterrows():
            background_path = os.path.join("substrates", row["background_label"], row["background_gsd_cat"], row["background_id"])
            instance_path = os.path.join("instances", row["instance_label"], row["instance_gsd_cat"], row["instance_id"])
            if prev_background_id != row["background_id"]:
                if prev_background_id != "none":
                    md5hash = hashlib.md5(background.tobytes()).hexdigest()
                    image_name = md5hash + ".jpg"
                    label_name = md5hash + ".json"
                    background = background.convert("RGB")
                    print(image_name)
                    background.save(os.path.join(path, image_name))
                    annotation = {
                        "version": "5.0.1",
                        "flags": {},
                        "shapes": shapes,
                        "imagePath": image_name,
                        "imageData": 'null',
                        "imageHeight": height,
                        "imageWidth": width}
                    annotation_str = json.dumps(annotation, indent = 2).replace('"null"', 'null')
                    with open(os.path.join(path, label_name), 'w') as annotation_file:
                        annotation_file.write(annotation_str)
                background = Image.open(background_path).convert("RGBA")
                # transform background
                background, _ = transforms(background, row)
                width, height = background.size
                annotation = {""}
                prev_background_id = row["background_id"]
                background_width, background_height = background.size
                shapes = []
            instance = Image.open(instance_path).convert("RGBA")
            instance_width, instance_height = instance.size
            centre = (instance_width/2, instance_height/2)
            left = random.randint(0, background_width - instance_width)
            top = random.randint(0, background_height - instance_height)
            points = row["instance_points"]
            # transforms
            instance, points = transforms(instance, row, points)
            # paste instance on background
            background.paste(instance, (left, top), instance)
            # move points to pasted coordinates
            position = ((left, top))
            points = tuple(tuple(sum(x) for x in zip(a, position)) for a in points)
            # make annotation file
            shapes.append({
                "label": row["instance_label"],
                "points": points,
                "group_id": 'null',
                "shape_type": 'polygon',
                "flags": {}})

        #TODO make test set