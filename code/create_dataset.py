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
    centre_x = min_x + (max_x - min_x) / 2
    centre_y = min_y + (max_y - min_y) / 2
    origin = [(centre_x - width/2, centre_y - height/2)] * len(polygon)
    crop_corners = (centre_x - width/2, centre_y - height/2, centre_x + width/2, centre_y + height/2)
    bird = im.crop(crop_corners)
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
    return mask

def blackout_instance(image, box):
    box_width = box[2][0] - box[0][0]
    box_height = box[2][1] - box[0][1]
    topleft = (round(box[0][0]), round(box[0][1]))
    black_box = Image.new("RGB", (round(box_width), round(box_height)))
    image.paste(black_box, topleft)
    return image

def balance_instances(df, threshold):

    # Get names of columns containing class counts per image
    counts = df.columns

    # Create the model
    model = cp_model.CpModel()

    # create array containing row selection flags. 
    # True if row k is selected, False otherwise
    row_selection = [model.NewBoolVar(f'{i}') for i in range(df.shape[0])]

    # Number of instances for each class should be less than class_threshold
    for count in counts:
        if count != "null" and count != "image_id" and count != "patch_id":
            model.Add(df[count].dot(row_selection) <= threshold)
   
    # Maximise the number of imaes
    model.Maximize(pd.Series([1]*len(df)).dot(row_selection))

    # Solve
    solver = cp_model.CpSolver()
    solver.Solve(model)

    # Get the rows selected
    rows = [row for row in range(df.shape[0]) if solver.Value(row_selection[row])]
    return df.iloc[rows, :].reset_index()[["image_id", "patch_id"]]

def save_dataset(train, test):
    # to share coco categories accross train and test
    coco_categories = []
    for dataset in [train, test]:
        # use dataframe name as save directory
        dir = dataset.name
        # store coco annotaiton
        coco = {"images": [], "annotations": [], "categories": coco_categories}
        coco_path = "dataset/annotations/instances_{}.json".format(dir)
        coco_image_id = 0
        coco_category_id = 0
        coco_instance_id = 0
        # loop over dataset and save
        for patch_id, instances in dataset:
            # update patch data
            patch_path = os.path.join("dataset", dir, patch_id)
            labelme_path = os.path.splitext(patch_path)[0]+'.json'
            labelme_shapes = []
            # resize patch object to target gsd
            patch_object = instances["patch_object"].iloc[0]
            scale = instances["image_gsd"].iloc[0]/target_gsd
            size = tuple(int(dim * scale) for dim in patch_object.size)
            patch_object = patch_object.resize((size))
            # store coco info
            coco_image_id += 1
            coco_image_info = {
                "height": size[1],
                "width": size[0],
                "id": coco_image_id,
                "file_name": patch_id}
            coco["images"].append(coco_image_info)
            # loop over instances and save annotation
            for _, instance in instances.iterrows():
                instance_seg = []
                instance_box = []
                if instance["instance_id"] != "null":
                    instance_seg = tuple((int(point[0] * scale), int(point[1] * scale)) for point in instance["instance_box"])
                    instance_box = [
                        instance_seg[0][0],
                        instance_seg[0][1],
                        instance_seg[1][0] - instance_seg[0][0],
                        instance_seg[3][1] - instance_seg[0][1]]
                    area = instance_box[2] * instance_box[3]
                    # skip shadows
                    if "shadow" in instance["instance_class"]: continue
                    # blackout instances with low overlap
                    if (
                        instance["instance_overlap"] < overlap or\
                        "unknown" in instance["instance_class"] or\
                        instance["instance_class"] not in included_classes):
                        patch_object = blackout_instance(patch_object, instance_seg)
                        continue
                # save instance labelme annotation
                labelme_shapes.append({
                    "label": instance["instance_class"],
                    "points": instance_seg,
                    "group_id": 'null',
                    "shape_type": 'polygon',
                    "flags": {}})
                # save instance coco annotation
                if instance["instance_class"] != "null":
                    category_id = 0
                    for cat in coco["categories"]:
                        if instance["instance_class"] == cat["name"]:
                            category_id = cat["id"]
                            continue
                    if category_id == 0:
                        coco_category_id += 1
                        category_id = coco_category_id
                        coco_category = {
                            "id": category_id,
                            "name": instance["instance_class"],
                            "supercategory": "Bird"}
                        coco_categories.append(coco_category)
                    coco_instance_id += 1
                    coco_annotation = {
                        "iscrowd": 0,
                        "image_id": coco_image_id,
                        "bbox": instance_box,
                        "segmentation": [[item for sublist in instance_seg for item in sublist]],
                        "category_id": category_id,
                        "id": coco_instance_id,
                        "area": area}
                    coco["annotations"].append(coco_annotation)
            # save patch
            patch_object = patch_object.convert("RGB")
            patch_object.save(patch_path, exif = instance["image_exif"])
            # save labeleme
            labelme = {
                "version": "5.0.1",
                "flags": {},
                "shapes": labelme_shapes,
                "imagePath": patch_id,
                "imageData": 'null',
                "imageHeight": size[1],
                "imageWidth": size[0]}
            labelme = json.dumps(labelme, indent = 2).replace('"null"', 'null')
            with open(labelme_path, 'w') as labelme_file:
                labelme_file.write(labelme)
        # save coco
        coco = json.dumps(coco, indent = 2)
        with open(coco_path, 'w') as coco_file:
            coco_file.write(coco)

file_path_variable = search_for_file_path()
# did the user select a dir or cancel?
if len(file_path_variable) > 0:
    # confirm dir with user
    check = messagebox.askquestion(
        "CONFIRM",
        "Do you want to create a dataset from:\n" + file_path_variable)
    if check =="yes":
        os.chdir(file_path_variable)
        # create dictionaries to store data
        dataset_keys = [
            "image_id", "image_exif", "image_gsd", "patch_object", "patch_id",
            "ecosystem", "instance_object", "instance_id", "instance_class",
            "instance_box", "instance_shape_type", "instance_overlap"]
        data = {k:[] for k in dataset_keys}
        # data = {
        #     "image_id": [], "image_exif": [], "image_gsd": [],
        #     "patch_object": [], "patch_id": [], "ecosystem": [],
        #     "instance_object": [], "instance_id": [], "instance_class": [],
        #     "instance_box": [], "instance_shape_type": [], "instance_overlap": []}
        # variables
        target_gsd = 0.0025
        max_gsd = 0.0075
        overlap = 1.0
        natural_train_size = 150
        artificial_train_size = int(natural_train_size * 0.2)
        background_train_size = int((natural_train_size + artificial_train_size) * 0.2)
        natural_test_size = int((natural_train_size + artificial_train_size) * 0.2)
        background_test_size = int(natural_test_size * 0.2)
        min_instances_class = natural_train_size + artificial_train_size + natural_test_size
        min_polygons_class = artificial_train_size
        paths = ["dataset/train", "dataset/test", "dataset/annotations"]
        for path in paths:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
        # iterate through images
        for root, dirs, files in os.walk(os.getcwd()):
            for file in files:
                if file.endswith(".JPG"):
                    print("Recording Patch Data From: ", file)
                    # get image and annotation name and path
                    image_id = file
                    annotation_name = os.path.splitext(file)[0] + '.json'
                    image_path = os.path.join(root, image_id)
                    annotation_path = os.path.join(root, annotation_name)

                    # open the image
                    image = Image.open(image_path)

                    # read exif data
                    exif_dict = piexif.load(image.info['exif'])
                    comments = json.loads("".join(map(chr, [i for i in exif_dict["0th"][piexif.ImageIFD.XPComment] if i != 0])))

                    # get gsd
                    try:
                        image_gsd = float(comments["gsd"])
                        ecosystem = comments["ecosystem typology"]
                    except:
                        print("Incorrect comment data")
                        continue
                    if image_gsd > max_gsd:
                        print("GSD too coarse")
                        continue

                    # update exif data
                    comments["original_image"] = file
                    comments["gsd"] = target_gsd
                    exif_dict["0th"][piexif.ImageIFD.XPComment] = json.dumps(comments).encode('utf-16le')
                    image_exif = piexif.dump(exif_dict)

                    # rescale the image to target gsd
                    # This makes all objects and birds a consistant size.
                    scale = target_gsd/image_gsd
                    patchsize = 800 * scale
                    width, height = image.size

                    # determine how many crops within accross the image
                    n_crops_width = math.ceil(width / patchsize)
                    n_crops_height = math.ceil(height / patchsize)
                    # calculate the padded total size of the image
                    padded_width = n_crops_width * patchsize
                    padded_height = n_crops_height * patchsize
                    # calculate the width of the padding
                    pad_width = (padded_width - width) / 2
                    pad_height = (padded_height - height) / 2

                    # load the annotations
                    if os.path.exists(annotation_path):
                        annotation = json.load(open(annotation_path))
                    else: annotation = False
                    for height_index in range(n_crops_height):
                        for width_index in range(n_crops_width):
                            # calculate the edges of the patches
                            left = width_index * patchsize - pad_width
                            right = left + patchsize
                            top = height_index * patchsize - pad_height
                            bottom = top + patchsize
                            patch_points = (left, top, right, bottom)
                            # create the patch_object and patch_id
                            patch_object = image.crop(patch_points).convert("RGBA")
                            patch_id = (hashlib.md5(patch_object.tobytes()).hexdigest() + ".jpg")
                            # if there is an annotation,
                            # for each instance check if it overlaps
                            # with this patch and record data if it does
                            included_annotations = 0
                            if annotation != False:
                                for shape in annotation["shapes"]:
                                    if shape["label"] not in ["macrophyte", "water", "dirt", "tree", "artificial", "sand", "grass", "mud", "concrete", "artificial", "asphalt", "water", "rock", "boulder", "gravel", "shrub", "vehicle", "crop"]: 
                                        # move annotation to crop
                                        points = tuple((point[0] - left, point[1] - top) for point in shape["points"])
                                        # find bounding box corners
                                        xmin = min(points, key=lambda x: x[0])[0]
                                        xmax = max(points, key=lambda x: x[0])[0]
                                        ymin = min(points, key=lambda x: x[1])[1]
                                        ymax = max(points, key=lambda x: x[1])[1]

                                        # convert rectangle to mask if needed
                                        if shape["shape_type"] == "rectangle":
                                            points = [
                                                [xmin, ymin],
                                                [xmax, ymin],
                                                [xmax, ymax],
                                                [xmin, ymax]]
                                                                    
                                        # check if box is in patch at all
                                        if xmax < 0 or xmin > patchsize or ymax < 0 or ymin > patchsize:
                                            continue
                                        included_annotations += 1
                                        
                                        # check original area of bounding box in patch
                                        instance_area = (xmax - xmin) * (ymax - ymin)

                                        # crop bounding box to patch
                                        if xmin < 0: xmin = 0
                                        if xmax > patchsize: xmax = patchsize
                                        if ymin < 0: ymin = 0
                                        if ymax > patchsize: ymax = patchsize

                                        # calculate how much of the bounding box overlaps the patch
                                        instance_overlap = ((xmax - xmin) * (ymax - ymin)) / instance_area
                                        # get instance_object if it's a polygon
                                        instance_object = crop_mask(patch_object, points)
                                        instance_id = hashlib.md5(instance_object.tobytes()).hexdigest() + ".png"
                                        if shape["shape_type"] == "rectangle": instance_object = "box"
                                        instance_box = (
                                            (xmin,ymin),
                                            (xmax,ymin),
                                            (xmax,ymax),
                                            (xmin,ymax))

                                        instance_class = shape["label"]
                                        instance_shape_type = shape["shape_type"]

                                        # save patch data
                                        data["image_id"].append(image_id)
                                        data["image_exif"].append(image_exif)
                                        data["image_gsd"].append(image_gsd)
                                        data["patch_object"].append(patch_object)
                                        data["patch_id"].append(patch_id)
                                        data["ecosystem"].append(ecosystem)
                                        data["instance_object"].append(instance_object)
                                        data["instance_id"].append(instance_id)
                                        data["instance_class"].append(instance_class)
                                        data["instance_box"].append(instance_box)
                                        data["instance_shape_type"].append(instance_shape_type)
                                        data["instance_overlap"].append(instance_overlap)
                            # save patch as background if no birds
                            if included_annotations == 0:
                                data["image_id"].append(image_id)
                                data["image_exif"].append(image_exif)
                                data["image_gsd"].append(image_gsd)
                                data["patch_object"].append(patch_object)
                                data["patch_id"].append(patch_id)
                                data["ecosystem"].append(ecosystem)
                                data["instance_object"].append("null")
                                data["instance_id"].append("null")
                                data["instance_class"].append("null")
                                data["instance_box"].append("null")
                                data["instance_shape_type"].append("null")
                                data["instance_overlap"].append(1)
        # convert dictionary to dataframe
        data = pd.DataFrame(data=data)
        # class abundance
        total_class_count = (
            data
            .query("instance_overlap >= {0}".format(overlap))
            .instance_class
            .value_counts())
        # polygon abundance
        polygon_class_count = (
            data
            .query("instance_overlap >= {0}".format(overlap))
            .query("instance_shape_type == 'polygon'")
            .instance_class
            .value_counts())
        # print abundances
        print("\nMinimum overlap", overlap)
        print("\nMax GSD:", max_gsd)
        print("Target GSD:", target_gsd)
        print("\nNatural Train Size:", natural_train_size)
        print("Artificial Train Size:", artificial_train_size)
        print("Background Train Size:", background_train_size)
        print("Total Train Size:", natural_train_size + artificial_train_size + background_train_size)
        print("\nNatural Test_Size:", natural_test_size)
        print("Background Test Size:", background_test_size)
        print("Total Test Size:", natural_test_size + background_test_size)
        print("\nTotal Class Count:\n{}\n".format(total_class_count))
        print("Polygon Class Count:\n{}\n".format(polygon_class_count))
        # get a list of classes with enough instances and polygons
        total_count_threshold = list(total_class_count.index[total_class_count.gt(min_instances_class)])
        polygon_count_threshold = list(polygon_class_count.index[polygon_class_count.gt(min_polygons_class)])
        included_classes = list(set(total_count_threshold) & set(polygon_count_threshold))
        # grab masks of each included instance class
        artificial_masks = (
            data
            .query("instance_shape_type == 'polygon'")
            .query("instance_overlap >= {0}".format(overlap))
            .query("~instance_class.str.contains('shadow')", engine="python")
            .query("~instance_class.str.contains('unknown')", engine="python")
            .query("instance_class.isin({0})".format(included_classes), engine="python")
            .drop(['image_id', 'patch_object', 'patch_id', 'ecosystem'], axis=1)
            .groupby("instance_class")
            .sample(n=artificial_train_size, random_state=1))
        # grab backgrounds
        backgrounds = (
            data
            .query("instance_object == 'null'"))
        # grab backgrounds for artificial dataset
        background_classes = backgrounds['ecosystem'].nunique()
        artificial_backgrounds = (
            backgrounds
            .groupby("ecosystem")
            .sample(n=int((artificial_train_size * len(included_classes))/background_classes), random_state=1))
        # drop artificial backgrounds from backgrounds
        backgrounds = (
            pd.merge(backgrounds, artificial_backgrounds[["image_id", "patch_id"]], indicator=True, how='outer')
            .query('_merge=="left_only"')
            .drop('_merge', axis=1))
        # drop artificial backgrounds and sample train backgrounds
        train_backgrounds = (
            backgrounds
            .groupby("ecosystem")
            .sample(n= background_train_size, random_state=1))
        # drop artificial and train backgrounds and sample test backgrounds
        test_backgrounds = (
            pd.merge(backgrounds, train_backgrounds[["image_id", "patch_id"]], indicator=True, how='outer')
            .query('_merge=="left_only"')
            .drop('_merge', axis=1)
            .groupby("ecosystem")
            .sample(n= background_test_size, random_state=1))
        # paste masks on artificial backgrounds
        train_artificial = {k:[] for k in dataset_keys}
        for index in range(len(artificial_backgrounds)):
            patch_object = artificial_backgrounds.iloc[[index]]["patch_object"].item()
            instance_object = artificial_masks.iloc[[index]]["instance_object"].item()
            image_gsd = artificial_backgrounds.iloc[[index]]["image_gsd"].item()
            # resize instance to background gsd
            scale = artificial_masks.iloc[[index]]["image_gsd"].item()/image_gsd
            size = tuple(int(dim * scale) for dim in instance_object.size)
            instance_object = instance_object.resize((size))
            instance_width, instance_height = instance_object.size
            # paste instance randomly on background
            location = [int(dim/2) + random.randint(-int(dim/2), int(dim/2 - max(instance_object.size))) for dim in patch_object.size]
            instance_box = (
                (location[0], location[1]),
                (location[0] + instance_width, location[1]),
                (location[0] + instance_width, instance_height + location[1]),
                (location[0], instance_height + location[1]))
            patch_object.paste(instance_object, location, instance_object)
            # determine patch_id
            patch_id = hashlib.md5(patch_object.tobytes()).hexdigest() + ".jpg"
            # save new background object and id
            train_artificial["image_id"].append(artificial_backgrounds.iloc[[index]]["image_id"].item())
            train_artificial["image_exif"].append(artificial_backgrounds.iloc[[index]]["image_exif"].item())
            train_artificial["image_gsd"].append(image_gsd)
            train_artificial["patch_object"].append(patch_object)
            train_artificial["patch_id"].append(patch_id)
            train_artificial["ecosystem"].append(artificial_backgrounds.iloc[[index]]["ecosystem"].item())
            train_artificial["instance_object"].append(instance_object)
            train_artificial["instance_id"].append(artificial_masks.iloc[[index]]["instance_id"].item())
            train_artificial["instance_class"].append(artificial_masks.iloc[[index]]["instance_class"].item())
            train_artificial["instance_box"].append(instance_box)
            train_artificial["instance_shape_type"].append("polygon")
            train_artificial["instance_overlap"].append(1.0)
        # convert train_artificial to dataframe
        train_artificial = pd.DataFrame(data=train_artificial)
        # count class instances per patch
        # drop low overlap, shadows, unknowns, and artificial_masks 
        instances_per_patch = (
            data
            .query("instance_overlap >= {0}".format(overlap))
            .query("~instance_class.str.contains('shadow')", engine="python")
            .query("~instance_class.str.contains('unknown')", engine="python")
            .query("instance_class.isin({0})".format(included_classes), engine="python")
            .groupby(['image_id', 'patch_id', 'instance_class'])
            .size()
            .reset_index(name='counts')
            .pivot_table(
                index = ['image_id', 'patch_id'],
                columns="instance_class",
                values="counts",
                fill_value=0)
            .reset_index())
        train = balance_instances(instances_per_patch,natural_train_size)
        # drop train patches from possible test dataset and sample
        instances_per_patch = (
            pd.merge(instances_per_patch, train, indicator=True, how='outer')
            .query('_merge=="left_only"')
            .drop('_merge', axis=1))
        # subsample data for balanced test patches
        test = balance_instances(instances_per_patch,natural_test_size)
        # convert train and test back into patch data
        train = (
            pd.merge(data, train, indicator=True, how='outer')
            .query('_merge=="both"')
            .drop('_merge', axis=1))
        test = (
            pd.merge(data, test, indicator=True, how='outer')
            .query('_merge=="both"')
            .drop('_merge', axis=1))
        # join the natural, artificial, and background datasets
        # train
        train = (
            pd.concat([train, train_backgrounds, train_artificial], ignore_index=True)
            .groupby('patch_id'))
        train.name = 'train'
        # test
        test = (
            pd.concat([test, test_backgrounds], ignore_index=True)
            .groupby('patch_id'))
        test.name = 'test'
        # save datasets
        save_dataset(train, test)