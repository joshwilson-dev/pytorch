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
    # convrt points to tuple
    polygon = [tuple(l) for l in points]
    # find bounding box
    max_x = max([t[0] for t in points])
    min_x = min([t[0] for t in points])
    max_y = max([t[1] for t in points])
    min_y = min([t[1] for t in points])
    # crop a square image with bird at centre
    width = max_x - min_x
    height = max_y - min_y
    size = max(width, height)/2
    centre_x = min_x + (max_x - min_x) / 2
    centre_y = min_y + (max_y - min_y) / 2
    origin = [(centre_x - size, centre_y - size)] * len(polygon)
    bird = im.crop((centre_x - size, centre_y - size, centre_x + size, centre_y + size))
    # adjust polygon to crop origin
    polygon = tuple(tuple(a - b for a, b in zip(tup1, tup2))\
        for tup1, tup2 in zip(polygon, origin))
    # convert to numpy (for convenience)
    imArray = np.asarray(bird)
    # crop out the mask
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
    mask = np.array(maskIm)
    # assemble new image (uint8: 0-255)
    newImArray = np.empty(imArray.shape,dtype='uint8')
    # colors (three first columns, RGB)
    newImArray[:,:,:3] = imArray[:,:,:3]
    # transparency (4th column)
    newImArray[:,:,3] = mask*255
    # back to Image from numpy
    mask = Image.fromarray(newImArray, "RGBA")
    return mask, polygon

def rotate_point(point, centre, deg):
    rotated_point = (
        centre[0] + (point[0]-centre[0])*math.cos(math.radians(deg)) - (point[1]-centre[1])*math.sin(math.radians(deg)),
        centre[1] + (point[0]-centre[0])*math.sin(math.radians(deg)) + (point[1]-centre[1])*math.cos(math.radians(deg)))
    return rotated_point

def blackout_instance(image, box):
    box_width = box[2][0] - box[0][0]
    box_height = box[2][1] - box[0][1]
    topleft = (round(box[0][0]), round(box[0][1]))
    black_box = Image.new("RGB", (round(box_width), round(box_height)))
    image.paste(black_box, topleft)
    return image

def transforms(instance, original_gsd, target_gsd, points=None):
    # scale
    scale = target_gsd/original_gsd
    size = min(instance.size) / scale
    instance = T.Resize(size=int(size))(instance)
    # colour
    lower = 0.9
    upper = 1.1
    colour = random.uniform(lower, upper)
    instance = ImageEnhance.Color(instance)
    instance = instance.enhance(colour)
    contrast = random.uniform(lower, upper)
    instance = ImageEnhance.Contrast(instance)
    instance = instance.enhance(contrast)
    brightness = random.uniform(lower, upper)
    instance = ImageEnhance.Brightness(instance)
    instance = instance.enhance(brightness)
    # random hflip
    hflip = random.randint(0, 1)
    if hflip == 1:
        instance = T.RandomHorizontalFlip(1)(instance)
    # random vflip
    vflip = random.randint(0, 1)
    if vflip == 1:
        instance = T.RandomVerticalFlip(1)(instance)
    # rotate
    instance_width, instance_height = instance.size
    centre = (instance_width/2, instance_height/2)
    rotation = random.choice([0, 90, 180, 270])
    instance = T.RandomRotation((rotation, rotation))(instance)
    # points
    if points != None:
        # scale
        points = tuple(tuple(item / scale for item in point) for point in points)
        # hflip
        if hflip == 1:
            points = tuple(tuple([2 * centre[0] - point[0], point[1]]) for point in points)
        # vflip
        if vflip == 1:
            points = tuple(tuple([point[0], 2 * centre[1] - point[1]]) for point in points)
        # rotate
        points = tuple(rotate_point(point, centre, -rotation) for point in points)
    return instance, points

def solve(df, threshold, data_type):
    '''
    Uses or-tools module to solve optimization

    '''
    counts = df.columns

    # Creates the model.
    model = cp_model.CpModel()

    # Step 1: Create the variables
    # array containing row selection flags i.e. True if row k is selected, False otherwise
    # Note: treated as 1/0 in arithmeetic expressions
    row_selection = [model.NewBoolVar(f'{i}') for i in range(df.shape[0])]

    # Step 2: Define the constraints
    # The sum of the weights for the selected rows should be >= threshold
    for count in counts:
        if count != "null" and count != "image_id" and count != "patch_id":
            model.Add(df[count].dot(row_selection) <= threshold)
        elif count == "null" and data_type == "test":
            model.Add(df["null"].dot(row_selection) <= 0)
   
    # Step 3: Define the objective function
    # Minimize the total cost (based upon rows selected)
    model.Maximize(pd.Series([1]*len(df)).dot(row_selection))

    # Step 4: Creates the solver and solve.
    solver = cp_model.CpSolver()
    solver.Solve(model)

    # Get the rows selected
    rows = [row for row in range(df.shape[0]) if solver.Value(row_selection[row])]

    return df.iloc[rows, :]

def add_to_classifier(instance, test_train, patch, image_id):
    # crop box with pad so we can random crop 
    # to make classifier robust to box classifier error
    pad = 2
    box_left = instance["box"][0][0] - pad
    box_right = instance["box"][2][0] + pad
    box_top = instance["box"][0][1] - pad
    box_bottom = instance["box"][2][1] + pad
    instance_crop = patch.crop((box_left, box_top, box_right, box_bottom))
    # read exif data
    exif_dict = piexif.load(original_image.info['exif'])
    comments = json.loads("".join(map(chr, [i for i in exif_dict["0th"][piexif.ImageIFD.XPComment] if i != 0])))
    comments["original_image"] = image_id
    exif_dict["0th"][piexif.ImageIFD.XPComment] = json.dumps(comments).encode('utf-16le')
    exif_bytes = piexif.dump(exif_dict)
    width, height = instance_crop.size
    width -= random.randint(0, 8)
    height -= random.randint(0, 8)
    # random crop
    instance_crop = T.RandomCrop(size = (height, width))(instance_crop)
    # pad crop to square and resize
    max_dim = max(width, height)
    if height > width:
        pad = [int((height - width)/2) + 20, 20]
    else:
        pad = [20, int((width - height)/2) + 20]
    instance_crop = T.Pad(padding=pad)(instance_crop)
    instance_crop = T.CenterCrop(size=max_dim)(instance_crop)
    instance_crop = T.transforms.Resize(224)(instance_crop)
    
    # determine image hash
    md5hash = hashlib.md5(instance_crop.tobytes()).hexdigest()
    instance_id = md5hash + ".JPG"
    
    # save image
    instance_crop.save(os.path.join("classifier_dataset", test_train, instance_id), exif = exif_bytes)
    
    # add image to corresponding regional dataset
    species = instance["species"].split("_")
    species = species[-2].split(" ")[0] + " " + species[-1]
    classifiers[test_train]["Global"]["images"].append(instance_id)
    classifiers[test_train]["Global"]["labels"].append(instance["species"])
    for region in regions:
        if species in birds_by_region[region]:
            # save label and image name into relveant label dict key
            classifiers[test_train][region]["images"].append(instance_id)
            classifiers[test_train][region]["labels"].append(instance["species"])
    return

def save_dataset(dataset, test_train, patchsize):
    dataset = dataset.groupby('image_id')
    for image_id, patches in dataset:
        # load the image
        image = Image.open(image_id)
        # read exif data
        exif_dict = piexif.load(original_image.info['exif'])
        comments = json.loads("".join(map(chr, [i for i in exif_dict["0th"][piexif.ImageIFD.XPComment] if i != 0])))
        comments["original_image"] = image_id
        exif_dict["0th"][piexif.ImageIFD.XPComment] = json.dumps(comments).encode('utf-16le')
        exif_bytes = piexif.dump(exif_dict)
        # group image data by patch
        patches = patches.groupby('patch_points')
        # iterate over patches
        for patch_points, instances in patches:
            # crop image
            patch = image.crop(patch_points)
            # iterate over instances
            shapes = []
            for index, instance in instances.iterrows():
                # crop and save instance if overlap greater than 0.5
                if instance["box"] != "null":
                    if instance["overlap"] == 1.0 and instance["species"] in included_species and instance["iou_type"] == "polygon":
                        add_to_classifier(instance, test_train, patch, image_id)
                        # create annotation file
                        shapes.append({
                            "label": instance["species"],
                            "points": instance["patch_mask"],
                            "group_id": 'null',
                            "shape_type": 'polygon',
                            "flags": {}})
                    # blackout instance in image if overlap less than 0.5
                    # or is a species not included
                    elif "shadow" not in instance["species"]:
                        patch = blackout_instance(patch, instance["box"])
            # calculate md5 for cropped image
            md5hash = hashlib.md5(patch.tobytes()).hexdigest()
            patch_id = md5hash + ".JPG"
            # save patch
            patch.save(os.path.join("detector_dataset", test_train, patch_id), exif = exif_bytes)
            # save annotation
            annotation_id = md5hash + '.json'
            annotation = {
                "version": "5.0.1",
                "flags": {},
                "shapes": shapes,
                "imagePath": patch_id,
                "imageData": 'null',
                "imageHeight": patchsize,
                "imageWidth": patchsize}
            annotation_str = json.dumps(annotation, indent = 2).replace('"null"', 'null')
            with open(os.path.join("detector_dataset", test_train, annotation_id), 'w') as annotation_file:
                annotation_file.write(annotation_str)
    return

file_path_variable = search_for_file_path()
# did the user select a dir or cancel?
if len(file_path_variable) > 0:
    # confirm dir with user
    check = messagebox.askquestion(
        "CONFIRM",
        "Are you sure you want to create a dataset from the files in:\n" + file_path_variable)
    if check =="yes":
        os.chdir(file_path_variable)
        # create dictionaries to store data
        patches = {"substrate": [], "gsd": [], "gsd_cat": [], "image_id": [], "patch_id": [], "patch_points": [], "species": [], "instance_id": [], "instance_path": [], "instance_mask": [], "patch_mask": [], "box": [], "iou_type": [], "overlap": [], "test": []}
        classifiers = {"test": {}, "train/artificial": {}, "train/organic": {}}
        # specify gsd categories
        gsd_cats = ["error", "fine"]
        gsd_bins = [0.003, 0.01]
        # specify regional categories
        # load json containing birds in each region
        birds_by_region = json.load(open("birds_by_region.json"))
        regions = birds_by_region.keys()
        for test_train in classifiers.keys():
            for region in regions:
                classifiers[test_train][region] = {"images": [], "labels": []}
            classifiers[test_train]["Global"] = {"images": [], "labels": []}
        paths = ["detector_dataset/train/artificial", "detector_dataset/train/organic", "detector_dataset/test", "classifier_dataset/train/artificial", "classifier_dataset/train/organic", "classifier_dataset/test", "instances"]
        for path in paths:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
        # iterate through images
        for root, dirs, files in os.walk(os.getcwd()):
            for file in files:
                if file.endswith(".JPG"):
                    # get image and annotation name and path
                    image_name = file
                    annotation_name = os.path.splitext(file)[0] + '.json'
                    image_path = os.path.join(root, image_name)
                    annotation_path = os.path.join(root, annotation_name)

                    # open the image
                    original_image = Image.open(image_path).convert("RGBA")

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
                    for height_index in range(n_crops_height):
                        for width_index in range(n_crops_width):
                            left = width_index * patchsize - pad_width
                            right = left + patchsize
                            top = height_index * patchsize - pad_height
                            bottom = top + patchsize
                            patch_points = (left, top, right, bottom)
                            # for each instance check if it belongs to this
                            # patch and record data if it does
                            if os.path.exists(annotation_path):
                                annotation = json.load(open(annotation_path))
                                instance_id = 0
                                for index in range(len(annotation["shapes"])):
                                    # if this is the first time through the annotation,
                                    # save each of the instances
                                    if annotation["shapes"][index]["shape_type"] == "polygon":
                                        if patch_id == 0:
                                            # crop out instance
                                            instance_crop, instance_mask = crop_mask(original_image, annotation["shapes"][index]["points"])
                                            # determine md5
                                            md5hash = hashlib.md5(instance_crop.tobytes()).hexdigest()
                                            instance_path = os.path.join("instances", md5hash + ".PNG")
                                            # save
                                            instance_crop.save(instance_path)
                                        else:
                                            instance_path = instance_paths[index] # load it from first iteration
                                            patch_mask = patch_masks[index]
                                            instance_mask = instance_masks[index]
                                    else:
                                        instance_path = "null"
                                        instance_mask = "null"
                                    points_x, points_y = map(list, zip(*annotation["shapes"][index]["points"]))
                                    # adjust box and points to patch coordinates
                                    points_x = [x - left for x in points_x]
                                    points_y = [y - top for y in points_y]
                                    patch_mask = tuple(zip(points_x, points_y))
                                    box_xmin = min(points_x)
                                    box_xmax = max(points_x)
                                    box_ymin = min(points_y)
                                    box_ymax = max(points_y)
                                    box_width = box_xmax - box_xmin
                                    box_height = box_ymax - box_ymin
                                    box_area = box_width * box_height
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
                                    # copy box to mask if necessary
                                    if len(points_x) == 2:
                                        patch_mask = box
                                    # save instance_path and mask to list
                                    instance_paths.append(instance_path)
                                    instance_masks.append(instance_mask)
                                    patch_masks.append(patch_mask)
                                    # check if box is outside patch
                                    if box_xmin > patchsize or box_xmax < 0 or \
                                        box_ymin > patchsize or box_ymax < 0:
                                        instance_id += 1
                                        continue
                                    # calculate new area
                                    new_box_width = box_xmax - box_xmin
                                    new_box_height = box_ymax - box_ymin
                                    new_box_area = new_box_width * new_box_height
                                    overlap = new_box_area/box_area
                                    # save patch data
                                    patches["substrate"].append(substrate)
                                    patches["gsd"].append(gsd)
                                    patches["gsd_cat"].append(gsd_cat)
                                    patches["image_id"].append(image_path)
                                    patches["patch_id"].append(patch_id)
                                    patches["patch_points"].append(patch_points)
                                    patches["species"].append(annotation["shapes"][index]["label"])
                                    patches["instance_id"].append(instance_id)
                                    patches["instance_path"].append(instance_path)
                                    patches["instance_mask"].append(instance_mask)
                                    patches["patch_mask"].append(patch_mask)
                                    patches["box"].append(box)
                                    patches["iou_type"].append(annotation["shapes"][index]["shape_type"])
                                    patches["overlap"].append(overlap)
                                    patches["test"].append(0)
                                    instance_id += 1
                            else:
                                patches["substrate"].append(substrate)
                                patches["gsd"].append(gsd)
                                patches["gsd_cat"].append(gsd_cat)
                                patches["image_id"].append(image_path)
                                patches["patch_id"].append(patch_id)
                                patches["patch_points"].append(patch_points)
                                patches["species"].append("null")
                                patches["instance_id"].append("null")
                                patches["instance_path"].append("null")
                                patches["instance_mask"].append("null")
                                patches["patch_mask"].append("null")
                                patches["box"].append("null")
                                patches["iou_type"].append("polygon")
                                patches["overlap"].append(1)
                                patches["test"].append(0)
                            patch_id += 1

        # convert dictionary to df
        patches = pd.DataFrame(data=patches)

        # check balance of dataset
        substrate_balance = (
            patches
            .groupby(['image_id'])
            .first()
            .groupby(['substrate'])
            .size())

        # keep only instances with enough masks
        min_instances = 15
        test_instances = 5
        train_instances = 1000

        species_count = patches.loc[patches['iou_type'] == "polygon"]
        species_count = (
            species_count
            .sort_values('overlap', ascending=False)
            .drop_duplicates(['image_id','instance_id'])
            .species
            .value_counts())
        included_species = list(species_count.index[species_count.gt(min_instances)])
        print(species_count)
        print(substrate_balance)
        # Drop species without enough instances
        instances_per_patch = patches[
            patches
            .species
            .isin(included_species)]
        # drop shadows, unknowns, small overlap
        # then calculate the number of each species in each patch
        instances_per_patch = instances_per_patch[~instances_per_patch['species'].str.contains('shadow')]
        instances_per_patch = instances_per_patch[instances_per_patch['iou_type'].str.contains('polygon')]
        instances_per_patch = instances_per_patch[~instances_per_patch['species'].str.contains('unknown')].reset_index()
        instances_per_patch = (
            instances_per_patch
            .query('overlap == 1.0')
            .groupby(['image_id', 'patch_id', 'species'])
            .size()
            .reset_index(name='counts')
            .pivot_table(
                index = ['image_id', 'patch_id'],
                columns="species",
                values="counts",
                fill_value=0)
            .reset_index())

        # get the maximum number of images to keep instances less than threshold
        test = solve(instances_per_patch, test_instances, "test")
        for column in test.columns:
            if column != "patch_id" and column != "image_id":
                print(column, ": ", test[column].sum())
        test = test.reset_index()[["image_id", "patch_id"]]

        # remove test images from instances per patch      
        instances_per_patch = pd.merge(instances_per_patch, test, how='outer', indicator=True)
        instances_per_patch = (
            instances_per_patch
            .loc[instances_per_patch._merge == 'left_only']
            .drop("_merge", axis=1))
        # make train set
        train = solve(instances_per_patch, train_instances, "train")
        for column in train.columns:
            if column != "patch_id" and column != "image_id":
                print(column, ": ", train[column].sum())
        train = train.reset_index()[["image_id", "patch_id"]]

        # seperate patches into train and test data
        test = pd.merge(patches, test, how='outer', indicator=True)
        test = (
            test
            .loc[test._merge == 'both']
            .reset_index())

        train = pd.merge(patches, train, how='outer', on = ["image_id", "patch_id"], indicator=True)
        train = (
            train
            .loc[train._merge == 'both']
            .reset_index())

        save_dataset(test, "test", patchsize)
        save_dataset(train, "train/organic", patchsize)

        # create artificial training data
        # seperate dataframe of masks and shadows
        masks = train[
            train
            .species
            .isin(included_species)]
        masks = masks[train['iou_type'] == "polygon"]
        masks = masks[~masks["species"].str.contains("null")]
        masks = masks[~masks["species"].str.contains("unknown")]
        masks = masks[~masks["species"].str.contains("shadow")]
        # split into instances and shadows
        shadows = patches[patches["species"].str.contains("shadow")]
        shadows = (
            shadows
            .sort_values('overlap', ascending=False)
            .drop_duplicates(['image_id','instance_id']))

        # loop over backgrounds and paste on instances
        total_images = len(train.groupby(['image_id', 'patch_id']).size())
        train = train.groupby('image_id')
        patch_number = 0
        for image_id, patches in train:
            # load the image
            image = Image.open(image_id)
            # read exif data
            exif_dict = piexif.load(original_image.info['exif'])
            comments = json.loads("".join(map(chr, [i for i in exif_dict["0th"][piexif.ImageIFD.XPComment] if i != 0])))
            comments["original_image"] = image_id
            exif_dict["0th"][piexif.ImageIFD.XPComment] = json.dumps(comments).encode('utf-16le')
            exif_bytes = piexif.dump(exif_dict)
            # group image data by patch
            patches = patches.groupby('patch_points')
            # iterate over patches
            for patch_points, instances in patches:
                # crop image
                patch = image.crop(patch_points)
                # iterate over instances and black out
                for index, instance in instances.iterrows():
                    if instance["box"] != "null" and "shadow" not in instance["species"]:
                        patch = blackout_instance(patch, instance["box"])
                # change background quality
                background_gsd = random.uniform(gsd_bins[0], gsd_bins[1])
                scale = background_gsd/instances["gsd"].iloc[0]
                size = min(patch.size) / scale
                patch = T.Resize(size=int(size))(patch)
                # transform the background
                patch, _ = transforms(patch, background_gsd, instances["gsd"].iloc[0])
                # randomly sample masks of species
                species = random.choice(pd.unique(masks['species']))
                common_name = species.split("_")[0]
                flock = masks[masks["species"] == species]
                instances_per_image = int(instances['gsd'].iloc[0] * 1000)
                flock = flock.sample(instances_per_image, replace = True).reset_index(drop=True)
                shapes = []
                pasted = 0
                # paste new instances onto background
                for index, bird_data in flock.iterrows():
                    # open the original image
                    bird = Image.open(bird_data["instance_path"]).convert("RGBA")
                    # crop the mask out
                    points = bird_data["instance_mask"]
                    # change bird quality
                    bird_gsd = random.uniform(gsd_bins[0], gsd_bins[1])
                    scale = bird_gsd/bird_data["gsd"]
                    size = min(bird.size) / scale
                    bird = T.Resize(size=int(size))(bird)
                    points = tuple(tuple(item / scale for item in point) for point in points)
                    # transform the mask
                    bird, points = transforms(bird, bird_gsd, instances["gsd"].iloc[0], points)
                    # determine locatoins to paste instances
                    if pasted == 0:
                        # determine random positions in background
                        # that don't overlap or exceed image border
                        bird_size = max(bird.size)
                        edge_pad = 5
                        patchsize = max(patch.size)
                        x = np.linspace(edge_pad, patchsize - bird_size * 1.5, int((patchsize - edge_pad - bird_size * 1.5)/bird_size * 0.9))
                        y = np.linspace(edge_pad, patchsize - bird_size * 1.5, int((patchsize - edge_pad - bird_size * 1.5)/bird_size * 0.9))
                        xv, yv = np.meshgrid(x, y)
                        positions = list(zip(xv.ravel(), yv.ravel()))
                        try:
                            positions = random.sample(positions, instances_per_image)
                        except:
                            instances_per_image -= 1
                            continue
                    # paste the mask
                    left, top = tuple(map(int, positions[pasted]))
                    shadow_prob = random.randint(0, 1)
                    if shadow_prob == 1:
                        # paste shadow under instance
                        shadow_row = shadows[shadows["species"].str.contains(common_name)]
                        try: shadow_row = shadow_row.sample(1, replace = True)
                        except: shadow_row = shadows.sample(1, replace = True)
                        # open the original image
                        shadow = Image.open(shadow_row["instance_path"].item()).convert("RGBA")
                        shadow_points = shadow_row["instance_mask"].item()
                        # change quality of shadow
                        scale = bird_gsd/shadow_row["gsd"]
                        size = min(shadow.size) / scale
                        shadow = T.Resize(size=int(size))(shadow)
                        shadow_points = tuple(tuple(item / scale for item in point) for point in shadow_points)
                        # transform the mask
                        shadow, _ = transforms(shadow, bird_gsd, instances["gsd"].iloc[0], shadow_points)
                        shadow_offset = int(min(bird.size)/2)
                        shadow_position_x = left + random.randint(-shadow_offset, shadow_offset)
                        shadow_position_y = top + random.randint(-shadow_offset, shadow_offset)
                        patch.paste(shadow, (shadow_position_x, shadow_position_y), shadow)
                    patch.paste(bird, (left, top), bird)
                    pasted += 1
                    # move points to pasted coordinates
                    points = tuple(tuple(sum(x) for x in zip(a, ((left, top)))) for a in points)
                    box_xmin = min(points, key = lambda t: t[0])[0]
                    box_xmax = max(points, key = lambda t: t[0])[0]
                    box_ymin = min(points, key = lambda t: t[1])[1]
                    box_ymax = max(points, key = lambda t: t[1])[1]
                    bird_data["box"] = [
                        [box_xmin, box_ymin],
                        [box_xmax, box_ymin],
                        [box_xmax, box_ymax],
                        [box_xmin, box_ymax]]
                    # save the classifier data
                    add_to_classifier(bird_data, "train/artificial", patch, image_id)
                    # make annotation file
                    shapes.append({
                        "label": bird_data["species"],
                        "points": points,
                        "group_id": 'null',
                        "shape_type": 'polygon',
                        "flags": {}})
                if instances_per_image > 0:
                    # save image if all, and at least one, instances have been pasted
                    print("Saving final image {} of {}".format(patch_number, total_images))
                    patch_number += 1
                    md5hash = hashlib.md5(patch.tobytes()).hexdigest()
                    image_id = md5hash + ".jpg"
                    annotation_id = md5hash + ".json"
                    patch = patch.convert("RGB")
                    patch.save(os.path.join("detector_dataset/train/artificial", image_id), exif = exif_bytes)
                    height, width = patch.size
                    annotation = {
                        "version": "5.0.1",
                        "flags": {},
                        "shapes": shapes,
                        "imagePath": image_id,
                        "imageData": 'null',
                        "imageHeight": height,
                        "imageWidth": width}
                    annotation_str = json.dumps(annotation, indent = 2).replace('"null"', 'null')
                    with open(os.path.join("detector_dataset/train/artificial", annotation_id), 'w') as annotation_file:
                        annotation_file.write(annotation_str)
        # create classifier annotation files
        for test_train in ["test", "train/artificial", "train/organic"]:
            for classifier in classifiers[test_train].keys():
                annotation = classifiers[test_train][classifier]
                dataset_path = os.path.join("classifier_dataset", test_train, classifier + ".json")
                annotation = json.dumps(annotation, indent=2)
                with open(dataset_path, "w") as annotation_file:
                    annotation_file.write(annotation)