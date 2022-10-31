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

def crop_mask(im, points):
    polygon = [tuple(l) for l in points]
    pad = 50
    # find bounding box
    max_x = max([t[0] for t in polygon])
    min_x = min([t[0] for t in polygon])
    max_y = max([t[1] for t in polygon])
    min_y = min([t[1] for t in polygon])
    centre_x = min_x + (max_x - min_x) / 2
    centre_y = min_y + (max_y - min_y) / 2
    # make image a square with bird at centre
    origin = [(centre_x - pad, centre_y - pad)] * len(polygon)
    polygon = tuple(tuple(a - b for a, b in zip(tup1, tup2))\
        for tup1, tup2 in zip(polygon, origin))
    # crop to bounding box
    bird = im.crop((centre_x - pad, centre_y - pad, centre_x + pad, centre_y + pad))
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
    name = md5hash + ".png"
    newIm.save(os.path.join("instances", name), format = "png")
    return [name, polygon]

def balance_resolutions(category, labels, df, min_instances, number):
    for label in labels:
        for gsd_cat in reversed(gsd_cats):
            count = df[df['label'] == label]
            count = count[count['gsd_cat'] == gsd_cat]
            count = len(count)
            # if gsd does not have at least min_instances instances per gsd_cat:
            # downscale images from higher or equal res
            if count < min_instances:
                print("Oversampling: ", label, gsd_cat)
                max_gsd = gsd_bins[gsd_cats.index(gsd_cat) + 1]
                high_res = df[df['label'] == label]
                high_res = high_res[high_res['gsd'] <= max_gsd]
                downscale = high_res.sample(n=min_instances - count, replace=True)
                downscale = downscale.reset_index(drop=True)
                for index, row in downscale.iterrows():
                    if category == "backgrounds":
                        downscale.loc[index, 'number'] = number
                        number = number + 1
                downscale = downscale.assign(gsd_cat=gsd_cat)
                df = pd.concat([df, downscale])
            # if gsd has more than instances_per_species instances drop it down to instances_per_species
            elif count > min_instances:
                print("Undersampling: ", label, gsd_cat)
                df = df.drop(df[df['label'].eq(label) & df['gsd_cat'].eq(gsd_cat)].sample(count - min_instances).index)
            df = df.reset_index(drop=True)
    return df, number

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

def blackout_instance(image, points):
    box_left = min(x[0] for x in points)
    box_right = max(x[0] for x in points)
    box_top = min(x[1] for x in points)
    box_bottom = max(x[1] for x in points)
    box_width = box_right - box_left
    box_height = box_bottom - box_top
    topleft = (round(box_left), round(box_top))
    black_box = Image.new("RGB", (round(box_width), round(box_height)))
    image.paste(black_box, topleft)
    return image

def transforms(instance, instance_gsd, random_gsd, points=None):
    # scale
    scale = random_gsd/instance_gsd
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
    # random flip
    # points
    if points != None:
        points = tuple(tuple(item / scale for item in point) for point in points)
        # rotate
        instance_width, instance_height = instance.size
        centre = (instance_width/2, instance_height/2)
        rotation = random.choice([0, 90, 180, 270])
        instance = T.RandomRotation((rotation, rotation))(instance)
        points = tuple(rotate_point(point, centre, -rotation) for point in points)
    return instance, points
number = 1
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
        instances = {"label": [], "gsd": [], "gsd_cat": [], "id": [], "points": []}
        backgrounds = {"label": [], "gsd": [], "gsd_cat": [], "id": [], "number": []}
        gsd_cats = ["fine"]
        gsd_bins = [0.004, 0.01]
        paths = ["instances", "dataset"]
        for path in paths:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
        for root, dirs, files in os.walk(os.getcwd()):
            for file in files:
                if file.endswith(".JPG"):
                    # get image and annotation name and path
                    image_name = file
                    annotation_name = os.path.splitext(file)[0] + '.json'
                    image_path = os.path.join(root, image_name)
                    annotation_path = os.path.join(root, annotation_name)

                    # open the image
                    image = Image.open(image_path)
                    image_width, image_height = image.size

                    # read exif data
                    exif_dict = piexif.load(image.info['exif'])
                    exif_bytes = piexif.dump(exif_dict)

                    # get gsd
                    comments = json.loads("".join(map(chr, [i for i in exif_dict["0th"][piexif.ImageIFD.XPComment] if i != 0])))
                    gsd = is_float(comments["gsd"])
                    substrate = is_float(comments["ecosystem typology"])
                    try:
                        gsd_cat = gsd_cats[numpy.digitize(gsd,gsd_bins) - 1]
                        print("Adding background: ", file)
                        # add to backgrounds dictionary
                        backgrounds["label"].append(substrate)
                        backgrounds["gsd"].append(gsd)
                        backgrounds["gsd_cat"].append(gsd_cat)
                        backgrounds["id"].append(file)
                        backgrounds["number"].append(number)
                        number = number + 1
                        # add to instanes dictionary if birds exist
                        if os.path.exists(annotation_path):
                            print("Adding instances from: ", file)
                            annotation = json.load(open(annotation_path))
                            for instance in annotation["shapes"]:
                                # crop out instances and save as image
                                instance_id, points = crop_mask(image.convert("RGBA"), instance["points"])
                                # blackout instance in original image
                                image = blackout_instance(image, instance["points"])
                                instances["label"].append(instance["label"])
                                instances["gsd"].append(gsd)
                                instances["gsd_cat"].append(gsd_cat)
                                instances["id"].append(instance_id)
                                instances["points"].append(points)
                            image.save(os.path.join("backgrounds", image_name), exif = exif_bytes)
                    except Exception as e: print(e)

        # convert dictionary to df
        instances = pd.DataFrame(data=instances)
        backgrounds = pd.DataFrame(data=backgrounds)
        # remove masks with unsuitable gsd values
        instances = instances.dropna()
        backgrounds = backgrounds.dropna()
        min_instances = 20
        min_backgrounds = 5
        # check that each species has enough instances in each gsd category
        instances, number = balance_resolutions(category = "instances", labels = instances.label.unique(), df = instances, min_instances = min_instances, number = number)
        # check that each background has enough instances in each gsd category
        # backgrounds, number = balance_resolutions(category = "backgrounds", labels = backgrounds.label.unique(), df = backgrounds, min_instances = min_backgrounds, number = number)
        # add common column to merge by
        backgrounds = backgrounds.add_prefix('background_')
        # seperate out shadows
        instances = instances.add_prefix('instance_')
        shadows = instances[instances["instance_label"] == "Shadow"]
        instances = instances[instances["instance_label"] != "Shadow"]
        instances["temp"] = 1
        backgrounds["temp"] = 1
        dataset = pd.DataFrame()
        # how many copies of each species per image
        instance_repeats = 5
        # how many copies of each background
        background_repeats = 1
        for repeat in range(background_repeats):
            # randomly select instance_repeats from each species
            sprites = instances.groupby('instance_label').apply(lambda x: x.sample(instance_repeats)).reset_index(drop=True)
            backgrounds = backgrounds.sample(n=len(backgrounds))
            rep = sprites.merge(backgrounds, on='temp').drop('temp', axis=1)
            rep = rep[rep["instance_gsd_cat"] == rep["background_gsd_cat"]]
            rep["background_repeat"] = repeat
            dataset = pd.concat([dataset, rep])
        # fix order
        dataset = dataset.sort_values(by=["background_repeat", "background_number", "instance_label"])
        # dataset = dataset.reset_index(drop=True)
        # dataset = dataset.assign(sorting = (dataset.index + 1) % min_instances)
        # dataset = dataset.sort_values(by=["background_repeat", "background_number", "sorting"])
        dataset = dataset.reset_index(drop=True)
        # save data to csv
        dataset.to_csv("dataset.csv")
        num_species = len(dataset.instance_label.unique())
        # actually create and save images
        for index, row in dataset.iterrows():
            if index % (num_species * instance_repeats) == 0:
                background_path = os.path.join("backgrounds", row["background_id"])
                background = Image.open(background_path).convert("RGBA")
                # transform background
                min_gsd = gsd_bins[gsd_cats.index(row["instance_gsd_cat"])]
                max_gsd = gsd_bins[gsd_cats.index(row["instance_gsd_cat"]) + 1]
                random_gsd = random.uniform(min_gsd, max_gsd)
                background, _ = transforms(background, row["background_gsd"], random_gsd)
                width, height = background.size
                annotation = {""}
                background_width, background_height = background.size
                shapes = []
            instance_path = os.path.join("instances", row["instance_id"])
            instance = Image.open(instance_path).convert("RGBA")
            shadow_row = shadows.sample(n=1)
            shadow_path = os.path.join("instances", shadow_row["instance_id"].item())
            shadow = Image.open(shadow_path).convert("RGBA")
            # transforms
            instance, points = transforms(instance, row["instance_gsd"], random_gsd, row["instance_points"])
            shadow, _ = transforms(shadow, shadow_row["instance_gsd"].item(), random_gsd, shadow_row["instance_points"].item())
            # paste instance on background
            instance_width, instance_height = instance.size
            left = random.randint(0, background_width - instance_width)
            top = random.randint(0, background_height - instance_height)
            shadow_offset = 10
            shadow_position_x = left + random.randint(-shadow_offset, shadow_offset)
            shadow_position_y = top + random.randint(-shadow_offset, shadow_offset)
            background.paste(shadow, (shadow_position_x, shadow_position_y), shadow)
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
            # save image if all instances have been paseted
            if (index + 1) % (num_species * instance_repeats) == 0:
                print("Saving final image: background {}, gsd category {}".format(row["background_label"], row["instance_gsd_cat"]))
                md5hash = hashlib.md5(background.tobytes()).hexdigest()
                image_name = md5hash + ".jpg"
                label_name = md5hash + ".json"
                background = background.convert("RGB")
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

        #TODO make test set