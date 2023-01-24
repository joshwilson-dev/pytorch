################
#### Header ####
################

# Title: curate data within sub directory
# Author: Josh Wilson
# Date: 02-06-2022
# Description: 
# This script takes a set of images with annotation files and creates
# a dataset with a balance of each class and background

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
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import numpy as np
import hashlib
import shutil
from ortools.sat.python import cp_model
from shapely.geometry import Polygon
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
    size = max(width, height)
    centre_x = min_x + (max_x - min_x) / 2
    centre_y = min_y + (max_y - min_y) / 2
    origin = [(centre_x - size/2, centre_y - size/2)] * len(polygon)
    crop_corners = (centre_x - size/2, centre_y - size/2, centre_x + size/2, centre_y + size/2)
    bird = im.crop(crop_corners).convert('RGBA')
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

def blackout_instance(image, box):
    box_width = box[2]
    box_height = box[3]
    topleft = (round(box[0]), round(box[1]))
    black_box = Image.new("RGB", (round(box_width), round(box_height)))
    image.paste(black_box, topleft)
    return image

def balance_instances(df, ratio):
    # Create the model
    model = cp_model.CpModel()
    # create array containing row selection flags. 
    # True if row k is selected, False otherwise
    row_selection = [model.NewBoolVar(f'{i}') for i in range(df.shape[0])]
    # Number of instances for each class should be less than class_threshold
    for column in df.columns:
        if "aves" in column:
            model.Add(df[column].dot(row_selection) <= math.floor(ratio * df[column].sum()))
    # Maximise the number of imaes
    model.Maximize(pd.Series([1]*len(df)).dot(row_selection))
    # Solve
    solver = cp_model.CpSolver()
    solver.Solve(model)
    # Get the rows selected
    rows = [row for row in range(df.shape[0]) if solver.Value(row_selection[row])]
    return df.iloc[rows, :].reset_index()

def rotate_point(point, centre, deg):
    rotated_point = [
        centre[0] + (point[0]-centre[0])*math.cos(math.radians(deg)) - (point[1]-centre[1])*math.sin(math.radians(deg)),
        centre[1] + (point[0]-centre[0])*math.sin(math.radians(deg)) + (point[1]-centre[1])*math.cos(math.radians(deg))]
    return rotated_point

def transforms(instance, mask, gsd, min_gsd, max_gsd, random_state):
    # fake shadow
    shadow = instance.copy()
    width, height = shadow.size
    shadow_data = []
    for item in shadow.getdata():
        if item[3] == 255:
            shadow_data.append((0, 0, 0, int(255/2)))
        else:
            shadow_data.append(item)
    shadow.putdata(shadow_data)
    x_offset = random.randint(-10, 10)
    y_offset = random.randint(-10, 10)
    shadow = shadow.crop((min(x_offset, 0), min(y_offset, 0), max(width, width + x_offset), max(height, height + x_offset)))
    shadow.paste(instance, (max(x_offset, 0), max(y_offset, 0)), instance)
    instance = shadow.copy()
    mask = [[point[0] + max(x_offset, 0), point[1] + max(y_offset, 0)] for point in mask]

    # min_transform = 0.75
    # max_transform = 1.25
    # random.seed(random_state)
    # quality
    # scale_gsd = random.uniform(min_gsd, max_gsd)
    # scale = gsd / scale_gsd
    # scale_size = (int(dim * scale) for dim in instance.size)
    # size = (instance.size)
    # instance = instance.resize(scale_size)
    # instance = instance.resize(size)
    # # blur
    # blur = random.uniform(0, min_transform)
    # instance = instance.filter(ImageFilter.GaussianBlur(blur))
    # # colour
    # colour = random.uniform(min_transform, max_transform)
    # instance = ImageEnhance.Color(instance)
    # instance = instance.enhance(colour)
    # # contrast
    # contrast = random.uniform(min_transform, max_transform)
    # instance = ImageEnhance.Contrast(instance)
    # instance = instance.enhance(contrast)
    # # brightness
    # brightness = random.uniform(min_transform, max_transform)
    # instance = ImageEnhance.Brightness(instance)
    # instance = instance.enhance(brightness)
    # # rotate
    # centre = [max(instance.size)/2] * 2
    # rotation = random.sample([0, 90, 180, 270], 1)[0]
    # if rotation != 0:
    #     instance = instance.rotate(rotation)
    #     mask = [rotate_point(point, centre, -rotation) for point in mask]
    # # poly crop
    # poly_prob = random.uniform(0, 1)
    # if poly_prob > 0.7:
    #     instance = crop_polygon(instance)
    return instance, mask

def crop_polygon(instance):
    width, height = instance.size
    instance_area = width * height
    draw = ImageDraw.Draw(instance)
    for _ in range(3):
        overlap = 1
        while overlap > 0.05 or overlap < 0.01:
            points = tuple((random.randint(0, width) , random.randint(0, height)) for _ in range(3))
            overlap = Polygon(points).area/instance_area
        draw.polygon(points, fill=(0, 0, 0, 0))
    return instance

def save_dataset(train, test, shadows):
    # to share coco categories accross train and test
    coco_categories = []
    for dataset in [train, test]:
        # use dataframe name as save directory
        dir = dataset.name
        total_images = dataset.groupby(['image_path', 'patch_points']).ngroups
        dataset = dataset.groupby('image_path')
        # store coco annotaiton
        coco = {"images": [], "annotations": [], "categories": coco_categories}
        coco_path = "dataset/annotations/instances_{}.json".format(dir)
        coco_image_id = 0
        coco_category_id = 0
        coco_instance_id = 0
        # loop over dataset and save
        for image_path, patches in dataset:
            # open image
            image = Image.open(image_path)
            # read exif data
            exif_dict = piexif.load(image.info['exif'])
            comments = json.loads("".join(map(chr, [i for i in exif_dict["0th"][piexif.ImageIFD.XPComment] if i != 0])))
            # update exif data
            comments["original_image"] = image_path
            comments["gsd"] = target_gsd
            exif_dict["0th"][piexif.ImageIFD.XPComment] = json.dumps(comments).encode('utf-16le')
            image_exif = piexif.dump(exif_dict)
            patches = patches.groupby("patch_points")
            for patch_points, instances in patches:
                # crop the image to the patch
                patch_object = image.crop(patch_points)
                # resize patch object to target gsd
                scale = instances["image_gsd"].iloc[0]/target_gsd
                size = tuple(int(dim * scale) for dim in patch_object.size)
                patch_object = patch_object.resize((size))
                labelme_shapes = []
                coco_image_id += 1
                # print where you're up to n creating dataset
                if coco_image_id % int(total_images / 10) == 0:
                    print("Saving {} image {} of {}".format(dir, coco_image_id, total_images))
                # loop over instances and save annotation
                for index, instance in instances.iterrows():
                    # skip shadows
                    if "shadow" in instance["instance_class"]: continue
                    # if its not a background
                    if instance["instance_class"] != "null":
                        # if instance is not artificial we need to scale the mask
                        if instance["instance_id"] == "artificial": scale = 1
                        instance_mask = [[int(point[0] * scale), int(point[1] * scale)] for point in instance["instance_mask"]]
                        # generate the instance box
                        xmin = min(instance_mask, key=lambda x: x[0])[0]
                        xmax = max(instance_mask, key=lambda x: x[0])[0]
                        ymin = min(instance_mask, key=lambda x: x[1])[1]
                        ymax = max(instance_mask, key=lambda x: x[1])[1]
                        instance_box = [xmin, ymin, xmax - xmin, ymax - ymin]
                        area = instance_box[2] * instance_box[3]
                        # blackout instances with low overlap
                        if (
                            instance["instance_overlap"] < overlap or\
                            "unknown" in instance["instance_class"] or\
                            instance["instance_class"] not in included_classes):
                            patch_object = blackout_instance(patch_object, instance_box)
                            continue
                        # save instance labelme annotation
                        labelme_shapes.append({
                            "label": instance["instance_class"],
                            "points": instance_mask,
                            "group_id": 'null',
                            "shape_type": 'polygon',
                            "flags": {}})
                        # save instance coco annotation
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
                            "segmentation":  [[item for sublist in instance_mask for item in sublist]],
                            "category_id": category_id,
                            "id": coco_instance_id,
                            "area": area}
                        coco["annotations"].append(coco_annotation)
                    # if there is an instance object, paste it onto the image
                    if instance["instance_id"] == "artificial":
                        instance_object = instance["instance_object"]
                        instance_size = max(instance_object.size)
                        width_offset = instance_size/2 - instance_box[2]/2
                        height_offset = instance_size/2 - instance_box[3]/2
                        location = (int(instance_box[0] - width_offset), int(instance_box[1] - height_offset))
                        # paste a shadow under the instance if its not background
                        random.seed(index)
                        shadow_prob = random.randint(0, 100)
                        if shadow_prob > 30:
                            try:
                                species = instance["instance_class"].split('_')[0].replace(" ", "_").lower() + "_shadow"
                                shadow = (
                                    shadows
                                    .query("instance_class.isin({0})".format(species), engine="python")
                                    .sample(n = 2, random_state = index)
                                    .reset_index(drop = True))
                            except Exception as e:
                                # print("no shadows match species")
                                shadow = (
                                    shadows
                                    .sample(n = 2, random_state = index)
                                    .reset_index(drop = True))
                            shadow_object = shadow.iloc[0]["instance_object"]
                            patch_object.paste(shadow_object, (instance_box[0], instance_box[1]), shadow_object)
                            shadow_object = shadow.iloc[1]["instance_object"]
                            patch_object.paste(shadow_object, (random.randint(0, size[0] - max(shadow_object.size)), random.randint(0, size[1] - max(shadow_object.size))), shadow_object)
                        patch_object.paste(instance_object, location, instance_object)
                # determine name of patch
                patch_name = hashlib.md5(patch_object.tobytes()).hexdigest() + ".jpg"
                # update patch data
                patch_path = os.path.join("dataset", dir, patch_name)
                labelme_path = os.path.splitext(patch_path)[0]+'.json'
                # store coco info
                coco_image_info = {
                    "height": size[1],
                    "width": size[0],
                    "id": coco_image_id,
                    "file_name": patch_name}
                coco["images"].append(coco_image_info)
                # save patch
                patch_object.save(patch_path, exif = image_exif)
                # save labeleme
                labelme = {
                    "version": "5.0.1",
                    "flags": {},
                    "shapes": labelme_shapes,
                    "imagePath": patch_name,
                    "imageData": 'null',
                    "imageHeight": size[1],
                    "imageWidth": size[0]}
                labelme = json.dumps(labelme, indent = 2).replace('"null"', 'null')
                with open(labelme_path, 'w') as labelme_file:
                    labelme_file.write(labelme)
        # save index to class if it's the last loop
        if dir == "test":
            index_to_class = {}
            for category in coco["categories"]:
                index_to_class[category["id"]] = category["name"]
            with open('dataset/annotations/index_to_class.json', 'w') as file:
                file.write(json.dumps(index_to_class, indent = 2))
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
            "image_path", "image_gsd", "patch_points",
            "instance_object", "instance_class", "instance_crop",
            "instance_mask", "instance_overlap"]
        data = {k:[] for k in dataset_keys}
        # variables
        target_gsd = 0.005
        max_gsd = 0.0075
        min_gsd = 0.0025
        min_overlap = 0.9
        train_test_ratio = 0.25
        target_patchsize = 800
        instances_per_background = 6
        paths = ["dataset/train", "dataset/test", "dataset/annotations"]
        for path in paths:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
        # iterate through images
        for root, dirs, files in os.walk(os.getcwd()):
            for file in files:
                if file.endswith(".json"):
                    print("Recording Patch Data From: ", file)
                    # load the annotation
                    annotation_name = file
                    annotation_path = os.path.join(root, annotation_name)
                    annotation = json.load(open(annotation_path))

                    # get the image name and path
                    image_name = annotation["imagePath"]
                    image_path = os.path.join(root, image_name)
                    image = Image.open(image_path)

                    # get image dimensions
                    image_width = annotation["imageWidth"]
                    image_height = annotation["imageHeight"]

                    # get the gsd
                    image_gsd = annotation["gsd"]
                    if image_gsd > max_gsd:
                        print("GSD too coarse")
                        continue

                    # determine the scale factor to scale the image to
                    # the target gsd, which makes all object a consistent size
                    scale = target_gsd/image_gsd
                    patchsize = target_patchsize * scale

                    # determine how many crops within accross the image
                    n_crops_width = math.ceil(image_width / patchsize)
                    n_crops_height = math.ceil(image_height / patchsize)

                    # calculate the padded total size of the image
                    padded_width = n_crops_width * patchsize
                    padded_height = n_crops_height * patchsize
                    
                    # calculate the width of the padding
                    pad_width = (padded_width - image_width) / 2
                    pad_height = (padded_height - image_height) / 2
                    
                    for height_index in range(n_crops_height):
                        for width_index in range(n_crops_width):
                            # calculate the edges of the patches
                            left = width_index * patchsize - pad_width
                            right = left + patchsize
                            top = height_index * patchsize - pad_height
                            bottom = top + patchsize
                            patch_corners = [
                                [left, top],
                                [right, top],
                                [right, bottom],
                                [left, bottom]]
                            patch_points = (left, top, right, bottom)
                            patch_poly = Polygon(patch_corners)
                            # for each instance check if it overlaps
                            # with this patch and record data if it does
                            instance_in_patch = False
                            instance_id = 0
                            for shape in annotation["shapes"]:
                                instance_id += 1
                                # if the shape is a rectange alert and exit
                                if shape["shape_type"] == "rectangle":
                                    sys.exit("There is a rectange label in your data: {}".format(file))
                                # find bounding box corners
                                instance_mask = ((point[0], point[1]) for point in shape["points"])

                                # create instance polygon
                                instance_poly = Polygon(shape["points"])

                                # check if polygon is in patch
                                instance_intersection = patch_poly.intersection(instance_poly)
                        
                                # if polygon is not in patch skip
                                if instance_intersection.area == 0: continue

                                # calculate the overlap between the full mask and the patch mask
                                instance_overlap = instance_intersection.area/instance_poly.area

                                # get the instance object
                                instance_object, instance_crop = crop_mask(image, shape["points"])
                                
                                # resize instance object to target gsd
                                scale = image_gsd/target_gsd
                                size = tuple(int(dim * scale) for dim in instance_object.size)
                                instance_object = instance_object.resize((size))
                                instance_crop = [[point[0] * scale, point[1] * scale] for point in instance_crop]

                                # get the instance class and shape type
                                instance_class = shape["label"]

                                # save instance to temp data
                                data["image_path"].append(image_path)
                                data["image_gsd"].append(image_gsd)
                                data["patch_points"].append(patch_points)
                                data["instance_object"].append(instance_object)
                                data["instance_class"].append(instance_class)
                                data["instance_crop"].append(instance_crop)
                                data["instance_mask"].append(instance_mask)
                                data["instance_overlap"].append(instance_overlap)
                                instance_in_patch = True
                                
                            # if there was no instancs in the patch add it as background
                            if instance_in_patch == False:
                                data["image_path"].append(image_path)
                                data["image_gsd"].append(image_gsd)
                                data["patch_points"].append(patch_points)
                                data["instance_object"].append("null")
                                data["instance_class"].append("background")
                                data["instance_crop"].append("null")
                                data["instance_mask"].append("null")
                                data["instance_overlap"].append(1.0)
        # convert dictionary to dataframe
        data = (
            pd.DataFrame(data=data)
            .query("~instance_class.str.contains('shadow')", engine="python"))
        # polygon abundance
        class_count = (
            data
            .drop_duplicates(["image_path", "instance_mask"])
            .instance_class
            .value_counts())
        print("\tClass Count:\n{}\n".format(class_count))
        # count class instances per patch
        instances_per_patch = (
            data
            .query("~instance_class.str.contains('unknown')", engine="python")
            .query("instance_class != 'background'")
            .query("instance_overlap >= {0}".format(min_overlap))
            .groupby(['image_path', 'patch_points', 'instance_class'])
            .size()
            .reset_index(name='counts')
            .pivot_table(
                index = ['image_path', 'patch_points'],
                columns="instance_class",
                values="counts",
                fill_value=0)
            .reset_index())
        # generate test dataset
        test_instances = balance_instances(instances_per_patch, train_test_ratio)[["image_path", "patch_points"]]
        # convert test back into patch data
        test_instances = (
            pd.merge(data, test_instances, indicator=True, how='outer')
            .query('_merge=="both"')
            .drop('_merge', axis=1)
            .reset_index(drop = True))
        # create train_masks
        train_instances = (
            data
            .query("~instance_class.str.contains('unknown')", engine="python")
            .query("instance_class != 'background'")
            .query("instance_overlap >= {0}".format(min_overlap))
            .drop_duplicates(["image_path", "instance_mask"])
            .reset_index(drop = True))
        # determin the number of samples to take from each image
        patches_per_image = (
            data
            .query("instance_class == 'background'")
            .groupby("image_path")
            .size())
        # assign % of background patches to test set for each image
        test_backgrounds_per_image = {}
        for i in range(len(patches_per_image)):
            test_backgrounds_per_image[patches_per_image.index[i]] = math.floor(patches_per_image[i] * train_test_ratio)
        test_backgrounds = (
            data
            .query("instance_class == 'background'")
            .groupby("image_path")
            .apply(lambda group: group.sample(test_backgrounds_per_image[group.name], random_state=1)))
        # join test datasets
        test = (
            pd.concat([test_instances, test_backgrounds], ignore_index=True)
            .reset_index(drop = True))
        test.name = 'test'
        # drop test from data
        data = (
            pd.merge(data, test[["image_path", "patch_points"]], indicator=True, how='outer')
            .query('_merge=="left_only"')
            .drop('_merge', axis=1)
            .reset_index(drop = True))
        # generate training natural backgrounds
        artificial_train_backgrounds = (
            data
            .query("instance_class == 'null'")
            .reset_index(drop = True))


# here need to sample 25% of all backgrounds for train and fix masks and check shadows



        # step through backgrounds and paste instances on
        train = pd.DataFrame(columns=dataset_keys)
        for index1 in range(len(artificial_train_backgrounds.index)):
            random.seed(index1)
            species = random.sample(included_classes, 1)
            instances = (
                artificial_train_masks
                .query("instance_class.isin({0})".format(species), engine="python")
                .sample(n = instances_per_background, random_state = index1)
                .reset_index(drop = True))
            # create grid of potential bird positions
            bird_size = max(instances.loc[0]["instance_object"].size) 
            spacing = range(0, target_patchsize - bird_size, int(bird_size * 0.75))
            random.seed(index1)
            x = random.sample(spacing, instances_per_background)
            random.seed(index1 + 1)
            y = random.sample(spacing, instances_per_background)
            positions = [[x, y] for x, y in zip(x, y)]
            # save info to dataframe
            for index2 in range(instances_per_background):
                index = index1 * instances_per_background + index2
                instance_object, instance_crop = transforms(instances.loc[index2]["instance_object"], instances.loc[index2]["instance_crop"], target_gsd, min_gsd, max_gsd, index)
                instance_mask = [[point[0] + positions[index2][0], point[1] + positions[index2][1]] for point in instance_crop]
                train.loc[index] = artificial_train_backgrounds.loc[index1]
                train.loc[index]["instance_object"] = instance_object
                train.loc[index]["instance_class"] = instances.loc[index2]["instance_class"]
                train.loc[index]["instance_mask"] = instance_mask
                train.loc[index]["instance_id"] = "artificial"
        train.name = 'train'
        # save data
        save_dataset(train, test, shadows)