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
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import hashlib
import shutil
from ortools.sat.python import cp_model
from shapely.geometry import Polygon

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
    return mask, polygon

def blackout_instance(image, box):
    box_width = box[2][0] - box[0][0]
    box_height = box[2][1] - box[0][1]
    topleft = (round(box[0][0]), round(box[0][1]))
    black_box = Image.new("RGB", (round(box_width), round(box_height)))
    image.paste(black_box, topleft)
    return image

def balance_instances(df, threshold):
    # Create the model
    model = cp_model.CpModel()
    # create array containing row selection flags. 
    # True if row k is selected, False otherwise
    row_selection = [model.NewBoolVar(f'{i}') for i in range(df.shape[0])]
    # Number of instances for each class should be less than class_threshold
    for column in df.columns:
        if "aves" in column:
            model.Add(df[column].dot(row_selection) <= threshold)
    # Maximise the number of imaes
    model.Maximize(pd.Series([1]*len(df)).dot(row_selection))
    # Solve
    solver = cp_model.CpSolver()
    solver.Solve(model)
    # Get the rows selected
    rows = [row for row in range(df.shape[0]) if solver.Value(row_selection[row])]
    return df.iloc[rows, :].reset_index()

def rotate_point(point, centre, deg):
    rotated_point = (
        centre[0] + (point[0]-centre[0])*math.cos(math.radians(deg)) - (point[1]-centre[1])*math.sin(math.radians(deg)),
        centre[1] + (point[0]-centre[0])*math.sin(math.radians(deg)) + (point[1]-centre[1])*math.cos(math.radians(deg)))
    return rotated_point

def transforms(instance, box, gsd, min_gsd, max_gsd, random_state):
    min_transform = 0.5
    max_transform = 1.5
    random.seed(random_state)
    # quality
    scale_gsd = random.uniform(min_gsd, max_gsd)
    scale = gsd / scale_gsd
    scale_size = (int(dim * scale) for dim in instance.size)
    size = (instance.size)
    instance = instance.resize(scale_size)
    instance = instance.resize(size)
    # colour
    colour = random.uniform(min_transform, max_transform)
    instance = ImageEnhance.Color(instance)
    instance = instance.enhance(colour)
    # contrast
    contrast = random.uniform(min_transform, max_transform)
    instance = ImageEnhance.Contrast(instance)
    instance = instance.enhance(contrast)
    # brightness
    brightness = random.uniform(min_transform, max_transform)
    instance = ImageEnhance.Brightness(instance)
    instance = instance.enhance(brightness)
    # rotate
    centre = [max(instance.size)/2] * 2
    rotation = random.sample([0], 1)[0]
    instance = instance.rotate(rotation)
    box = tuple(rotate_point(point, centre, -rotation) for point in box)
    return instance, box

def save_dataset(train, test, shadows):
    # to share coco categories accross train and test
    coco_categories = []
    for dataset in [train, test]:
        # use dataframe name as save directory
        dir = dataset.name
        total_images = dataset.groupby(['image_path', 'patch_id']).ngroups
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
                    instance_seg = []
                    instance_box = []
                    # if instance is not background
                    if instance["instance_id"] != "null":
                        # if instance is artificial object 
                        if instance["instance_object"] == "null":
                            instance_seg = tuple((int(point[0] * scale), int(point[1] * scale)) for point in instance["instance_box"])
                        else:
                            instance_seg = tuple((int(point[0]), int(point[1])) for point in instance["instance_box"])
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
                    # if there is an instance object, paste it onto the image
                    if instance["instance_object"] != "null":
                        instance_object = instance["instance_object"]
                        instance_size = max(instance_object.size)
                        width_offset = instance_size/2 - instance_box[2]/2
                        height_offset = instance_size/2 - instance_box[3]/2
                        location = (int(instance_box[0] - width_offset), int(instance_box[1] - height_offset))
                        # paste a shadow under the instance if its not background
                        try:
                            species = instance["instance_class"].split('_')[0].replace(" ", "_").lower() + "_shadow"
                            shadow = (
                                shadows
                                .query("instance_class.isin({0})".format(species), engine="python")
                                .sample(n = 1, random_state = index))
                        except Exception as e:
                            # print("no shadows match species")
                            shadow = shadows.sample(n = 1, random_state = index)
                        shadow_object = shadow["instance_object"].item()
                        random.seed(index)
                        shadow_prob = random.randint(0, 100)
                        if shadow_prob > 70:
                            patch_object.paste(shadow_object, (instance_box[0], instance_box[1]), shadow_object)
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
            "image_path", "image_gsd", "patch_points", "patch_id",
            "instance_id", "instance_object", "instance_class", 
            "instance_mask", "instance_box", "instance_shape_type",
            "instance_overlap"]
        data = {k:[] for k in dataset_keys}
        # variables
        target_gsd = 0.005
        max_gsd = 0.0075
        min_gsd = 0.0025
        overlap = 1.0
        min_instances_class = 100
        train_test_ratio = 0.25
        test_instances_size = 25
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
                    
                    patch_id = 0
                    for height_index in range(n_crops_height):
                        for width_index in range(n_crops_width):
                            patch_id += 1
                            # calculate the edges of the patches
                            left = width_index * patchsize - pad_width
                            right = left + patchsize
                            top = height_index * patchsize - pad_height
                            bottom = top + patchsize
                            patch_points = (left, top, right, bottom)
                            # for each instance check if it overlaps
                            # with this patch and record data if it does
                            instance_id = 0
                            instance_in_patch = False
                            for shape in annotation["shapes"]:
                                instance_id += 1
                                # find bounding box corners
                                instance_points = shape["points"]
                                xmin = min(instance_points, key=lambda x: x[0])[0]
                                xmax = max(instance_points, key=lambda x: x[0])[0]
                                ymin = min(instance_points, key=lambda x: x[1])[1]
                                ymax = max(instance_points, key=lambda x: x[1])[1]

                                # convert rectangle to mask if needed
                                if shape["shape_type"] == "rectangle":
                                    print(file)
                                    instance_points = [
                                        [xmin, ymin],
                                        [xmax, ymin],
                                        [xmax, ymax],
                                        [xmin, ymax]]
                                
                                # save instance mask in image coordinates
                                instance_mask = instance_points

                                # move points to patch coordinates
                                xmin -= left
                                xmax -= left
                                ymin -= top
                                ymax -= top
                                                            
                                # check if box is in patch at all
                                if xmax < 0 or xmin > patchsize or ymax < 0 or ymin > patchsize:
                                    pass
                                
                                # if it's not substrate get the required parameters
                                elif "substrate" not in shape["label"] and "background" not in shape["label"]:
                                    # check original area of bounding box in patch
                                    instance_area = (xmax - xmin) * (ymax - ymin)

                                    # crop bounding box to patch
                                    if xmin < 0: xmin = 0
                                    if xmax > patchsize: xmax = patchsize
                                    if ymin < 0: ymin = 0
                                    if ymax > patchsize: ymax = patchsize

                                    # calculate how much of the bounding box overlaps the patch
                                    instance_overlap = ((xmax - xmin) * (ymax - ymin)) / instance_area

                                    # calculate the instance bounding box
                                    instance_box = (
                                        (xmin,ymin),
                                        (xmax,ymin),
                                        (xmax,ymax),
                                        (xmin,ymax))

                                    # get the instance class and shape type
                                    instance_class = shape["label"]
                                    instance_shape_type = shape["shape_type"]

                                    # save instance to temp data
                                    data["image_path"].append(image_path)
                                    data["image_gsd"].append(image_gsd)
                                    data["patch_points"].append(patch_points)
                                    data["patch_id"].append(patch_id)
                                    data["instance_id"].append(instance_id)
                                    data["instance_object"].append("null")
                                    data["instance_class"].append(instance_class)
                                    data["instance_mask"].append(instance_mask) # instance mask in image coordinates
                                    data["instance_box"].append(instance_box) # instance box in patch coordinates
                                    data["instance_shape_type"].append(instance_shape_type)
                                    data["instance_overlap"].append(instance_overlap)
                                    instance_in_patch = True
                                
                                # if it's the last shape, add temp data to data
                                if instance_id == len(annotation["shapes"]):
                                    # if there's no instances add patch as background
                                    if instance_in_patch == False:
                                        data["image_path"].append(image_path)
                                        data["image_gsd"].append(image_gsd)
                                        data["patch_points"].append(patch_points)
                                        data["patch_id"].append(patch_id)
                                        data["instance_id"].append("null")
                                        data["instance_object"].append("null")
                                        data["instance_class"].append("null")
                                        data["instance_mask"].append("null")
                                        data["instance_box"].append("null")
                                        data["instance_shape_type"].append("null")
                                        data["instance_overlap"].append(1.0)
        # convert dictionary to dataframe
        data = pd.DataFrame(data=data)
        # class abundance
        total_class_count = (
            data
            .drop_duplicates(["image_path", "instance_id"])
            .query("~instance_class.str.contains('unknown')", engine="python")
            .query("instance_class != 'null'")
            .instance_class
            .value_counts())
        # rectangle abundance
        rectangle_class_count = (
            data
            .drop_duplicates(["image_path", "instance_id"])
            .query("instance_shape_type == 'rectangle'")
            .query("~instance_class.str.contains('unknown')", engine="python")
            .query("instance_class != 'null'")
            .instance_class
            .value_counts())
        # polygon abundance
        polygon_class_count = (
            data
            .drop_duplicates(["image_path", "instance_id"])
            .query("instance_shape_type == 'polygon'")
            .query("~instance_class.str.contains('unknown')", engine="python")
            .query("instance_class != 'null'")
            .instance_class
            .value_counts())
        # print abundances
        print("\nMinimum overlap", overlap)
        print("\nMax GSD:", max_gsd)
        print("Target GSD:", target_gsd)
        print("\nRequired Counts")
        print("\tMinimum Instances Per Class Count:\n\t\t{}".format(min_instances_class))
        print("\nActual Counts")
        print("\tTotal Class Count:\n{}\n".format(total_class_count))
        print("\tRectangle Class Count:\n{}\n".format(rectangle_class_count))
        print("\tPolygon Class Count:\n{}\n".format(polygon_class_count))
        # grab shadows
        shadows = (
            data
            .query("instance_class.str.contains('shadow')", engine="python")
            .query("instance_shape_type == 'polygon'")
            .reset_index(drop = True))
        # actually get the shadow objects
        image_path = "null"
        for index, row in shadows.iterrows():
            if row["image_path"] != image_path:
                image_path = row["image_path"]
                image_gsd = row["image_gsd"]
                image = Image.open(image_path).convert('RGBA')
            instance_object, instance_mask = crop_mask(image, row["instance_mask"])
            # rescale instance to target gsd
            scale = image_gsd/target_gsd
            size = tuple(int(dim * scale) for dim in instance_object.size)
            instance_object = instance_object.resize((size))
            shadows.at[index, "instance_object"] = instance_object
        # drop classes without enough instances
        included_classes = list(polygon_class_count.index[polygon_class_count.gt(min_instances_class)])
        print("Included classes: ", included_classes)
        # count class instances per patch
        instances_per_patch = (
            data
            .query("~instance_class.str.contains('unknown')", engine="python")
            .query("~instance_class.str.contains('null')", engine="python")
            .query("instance_overlap >= {0}".format(overlap))
            .query("instance_class.isin({0})".format(included_classes), engine="python")
            .groupby(['image_path', 'patch_id', 'instance_class'])
            .size()
            .reset_index(name='counts')
            .pivot_table(
                index = ['image_path', 'patch_id'],
                columns="instance_class",
                values="counts",
                fill_value=0)
            .reset_index())
        # generate test dataset
        natural_test_instances = balance_instances(instances_per_patch, test_instances_size)[["image_path", "patch_id"]]
        # convert test back into patch data
        natural_test_instances = (
            pd.merge(data, natural_test_instances, indicator=True, how='outer')
            .query('_merge=="both"')
            .drop('_merge', axis=1)
            .reset_index(drop = True))
        # determin the number of samples to take from each image
        patches_per_image = (
            data
            .query("instance_class == 'null'")
            .groupby("image_path")
            .size())
        # assign % of background patches to test set for each image
        test_backgrounds_per_image = {}
        for i in range(len(patches_per_image)):
            test_backgrounds_per_image[patches_per_image.index[i]] = int(patches_per_image[i] * train_test_ratio)
        natural_test_backgrounds = (
            data
            .query("instance_class == 'null'")
            .groupby("image_path")
            .apply(lambda group: group.sample(test_backgrounds_per_image[group.name], random_state=1)))
        # join test datasets
        test = (
            pd.concat([natural_test_instances, natural_test_backgrounds], ignore_index=True)
            .reset_index(drop = True))
        test.name = 'test'
        # drop test from data
        data = (
            pd.merge(data, test[["image_path", "patch_id"]], indicator=True, how='outer')
            .query('_merge=="left_only"')
            .drop('_merge', axis=1)
            .reset_index(drop = True))
        # generate training natural backgrounds
        artificial_train_backgrounds = (
            data
            .query("instance_class == 'null'")
            .reset_index(drop = True))
        # create train_masks
        artificial_train_masks = (
            data
            .query("instance_class != 'null'")
            .query("~instance_class.str.contains('unknown')", engine="python")
            .query("instance_class.isin({0})".format(included_classes), engine="python")
            .query("instance_shape_type == 'polygon'")
            .drop_duplicates(["image_path", "instance_id"])
            .reset_index(drop = True))
        # actually get the instance objects
        image_path = "null"
        for index, row in artificial_train_masks.iterrows():
            if row["image_path"] != image_path:
                image_path = row["image_path"]
                image_gsd = row["image_gsd"]
                image = Image.open(image_path).convert('RGBA')
            instance_object, instance_mask = crop_mask(image, row["instance_mask"])
            # rescale instance to target gsd
            scale = image_gsd/target_gsd
            size = tuple(int(dim * scale) for dim in instance_object.size)
            instance_object = instance_object.resize((size))
            instance_mask = tuple((point[0] * scale, point[1] * scale) for point in instance_mask)
            xmin = min(instance_mask, key=lambda x: x[0])[0]
            xmax = max(instance_mask, key=lambda x: x[0])[0]
            ymin = min(instance_mask, key=lambda x: x[1])[1]
            ymax = max(instance_mask, key=lambda x: x[1])[1]
            instance_box = (
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax))
            artificial_train_masks.at[index, "instance_object"] = instance_object
            artificial_train_masks.at[index, "instance_box"] = instance_box
        # step through backgrounds and paste instances on
        train = pd.DataFrame(columns=dataset_keys)
        for index1 in range(len(artificial_train_backgrounds.index)):
            random.seed(index1)
            species = random.sample(included_classes, 1)
            instances = (
                artificial_train_masks
                .query("instance_class.isin({0})".format(species), engine="python"))
            n = min(instances_per_background, len(instances))
            instances = (
                instances
                .sample(n = n, random_state = index1)
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
            for index2 in range(n):
                index = index1 * instances_per_background + index2
                instance_box = instances.loc[index2]["instance_box"]
                instance_object = instances.loc[index2]["instance_object"]
                instance_object, instance_box = transforms(instance_object, instance_box, target_gsd, min_gsd, max_gsd, index)
                instance_box = tuple((point[0] + positions[index2][0], point[1] + positions[index2][1]) for point in instance_box)
                train.loc[index] = artificial_train_backgrounds.loc[index1]
                train.loc[index]["instance_object"] = instance_object
                train.loc[index]["instance_class"] = instances.loc[index2]["instance_class"]
                train.loc[index]["instance_id"] = instances.loc[index2]["instance_id"]
                train.loc[index]["instance_box"] = instance_box
        train.name = 'train'
        # save data
        save_dataset(train, test, shadows)