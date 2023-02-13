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

def balance_instances(df, max_instances_class, ratio):
    # Create the model
    model = cp_model.CpModel()
    # create array containing row selection flags. 
    # True if row k is selected, False otherwise
    row_selection = [model.NewBoolVar(f'{i}') for i in range(df.shape[0])]
    # Number of instances for each class should be less than class_threshold
    for column in df.columns:
        if "aves" in column:
            model.Add(df[column].dot(row_selection) <= math.floor(min(ratio * df[column].sum(), max_instances_class)))
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

def transforms(instance, mask, gsd, min_gsd, max_gsd):
    min_transform = 0.75
    max_transform = 1.25
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
    rotation = random.sample([0, 90, 180, 270], 1)[0]
    if rotation != 0:
        instance = instance.rotate(rotation)
        mask = [rotate_point(point, centre, -rotation) for point in mask]
    # poly crop
    poly_prob = random.uniform(0, 1)
    if poly_prob > 0.95:
        instance = crop_polygon(instance)
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

def create_shadow(bird):
    shadow = bird.copy()
    width, height = shadow.size
    max_offset = 0.75
    shadow_data = []
    for item in shadow.getdata():
        if item[3] > 0:
            shadow_data.append((0, 0, 0, random.randint(125, 200)))
        else:
            shadow_data.append(item)
    shadow.putdata(shadow_data)
    x_offset = random.randint(int(-width * max_offset), int(width * max_offset))
    y_offset = random.randint(int(-height * max_offset), int(height * max_offset))
    shadow = shadow.crop((min(x_offset, 0), min(y_offset, 0), max(width, width + x_offset), max(height, height + y_offset)))
    return shadow

def save_dataset(train, test):
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
            # # update exif data
            # exif_dict = piexif.load(image.info['exif'])
            # comments = json.loads("".join(map(chr, [i for i in exif_dict["0th"][piexif.ImageIFD.XPComment] if i != 0])))
            # comments["original_image"] = image_path
            # comments["gsd"] = target_gsd
            # exif_dict["0th"][piexif.ImageIFD.XPComment] = json.dumps(comments).encode('utf-16le')
            # image_exif = piexif.dump(exif_dict)
            gsd = patches.iloc[0]["image_gsd"]
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
                if instances.iloc[0]["instance_type"] == "artificial":
                    # create grid of potential bird positions
                    bird_size = max(max(bird.size) for bird in instances["instance_object"])
                    spacing = range(0, target_patchsize - bird_size, int(bird_size * 0.75))
                    x = random.sample(spacing, instances_per_background)
                    y = random.sample(spacing, instances_per_background)
                    positions = [[x, y] for x, y in zip(x, y)]
                instances = instances.reset_index(drop = True)
                for index, instance in instances.iterrows():
                    # if its not a background
                    if instance["instance_class"] != "background":
                        # if instance is not artificial we need to scale the mask
                        if instance["instance_type"] == "artificial":
                            instance_mask = [[point[0] + positions[index][0], point[1] + positions[index][1]] for point in instance["instance_mask"]]
                        else:
                            instance_mask = [[int((point[0] - patch_points[0]) * scale), int((point[1] - patch_points[1]) * scale)] for point in instance["instance_mask"]]
                        # generate the instance box
                        xmin = min(instance_mask, key=lambda x: x[0])[0]
                        xmax = max(instance_mask, key=lambda x: x[0])[0]
                        ymin = min(instance_mask, key=lambda x: x[1])[1]
                        ymax = max(instance_mask, key=lambda x: x[1])[1]
                        coco_instance_box = [xmin, ymin, xmax - xmin, ymax - ymin]
                        labelme_instance_box = [[xmin, ymin], [xmax, ymax]]
                        area = coco_instance_box[2] * coco_instance_box[3]
                        # blackout instances that aren't included
                        if instance["instance_class"] not in included_classes or\
                            instance["instance_overlap"] < min_overlap:
                            patch_object = blackout_instance(patch_object, coco_instance_box)
                            continue
                        # save instance labelme annotation
                        labelme_annotation = {
                            "label": instance["instance_class"],
                            "group_id": 'null',
                            "flags": {}}
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
                            "category_id": category_id,
                            "segmentation": [[item for sublist in instance_mask for item in sublist]],
                            "bbox": coco_instance_box,
                            "id": coco_instance_id,
                            "area": area}
                        if len(instance_mask) == 4:
                            labelme_annotation["points"] = labelme_instance_box
                            labelme_annotation["shape_type"] = "rectangle"
                        else:
                            labelme_annotation["points"] = instance_mask
                            labelme_annotation["shape_type"] = "polygon"
                        labelme_shapes.append(labelme_annotation)
                        coco["annotations"].append(coco_annotation)
                    # if there is an instance object, paste it onto the image
                    if instance["instance_type"] == "artificial":
                        instance_object = instance["instance_object"]
                        location = (positions[index][0], positions[index][1])
                        # paste shadow under bird
                        if random.uniform(0, 1) < 0.3:
                            shadow = create_shadow(instance_object)
                            patch_object.paste(shadow, location, shadow)
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
                    "file_name": patch_name,
                    "gsd": gsd}
                coco["images"].append(coco_image_info)
                # save patch
                patch_object.save(patch_path)#, exif = image_exif)
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
            "instance_type", "instance_object", "instance_class",
            "instance_crop", "instance_mask", "instance_shape_type",
            "instance_overlap"]
        data = {k:[] for k in dataset_keys}
        # variables
        target_gsd = 0.005
        max_gsd = 0.0075
        min_gsd = 0.0025
        min_instances_class = 100
        min_polygons_class = 50
        max_instances_class = 10000
        train_test_ratio = 0.25
        max_instances_class_test = max_instances_class * train_test_ratio
        max_instances_class_train = max_instances_class - max_instances_class_test
        min_overlap = 0.9
        target_patchsize = 800
        instances_per_background = 4
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
                    try:
                        image_gsd = annotation["gsd"]
                    except:
                        print("NO GSD")
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
                                # if the shape is a rectange convert to mask
                                if shape["shape_type"] == "rectangle":
                                    xmin = shape["points"][0][0]
                                    xmax = shape["points"][1][0]
                                    ymin = shape["points"][0][1]
                                    ymax = shape["points"][1][1]
                                    instance_mask = tuple(((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)))
                                # convert mask to tuple
                                else: instance_mask = tuple((point[0], point[1]) for point in shape["points"])

                                # create instance polygon
                                instance_poly = Polygon(instance_mask)

                                # check if polygon is in patch
                                instance_intersection = patch_poly.intersection(instance_poly)
                        
                                # if polygon is not in patch skip
                                if instance_intersection.area == 0: continue

                                # calculate the overlap between the full mask and the patch mask
                                instance_overlap = instance_intersection.area/instance_poly.area
                                
                                if shape["shape_type"] == "polygon":
                                    # get the instance object
                                    instance_object, instance_crop = crop_mask(image, shape["points"])
                                    # resize instance object to target gsd
                                    scale = image_gsd/target_gsd
                                    size = tuple(int(dim * scale) for dim in instance_object.size)
                                    instance_object = instance_object.resize((size))
                                    instance_crop = [[point[0] * scale, point[1] * scale] for point in instance_crop]
                                else:
                                    instance_object = "null"
                                    instance_crop = "null"

                                # get the instance class and shape type
                                instance_class = shape["label"]

                                # save instance to temp data
                                data["image_path"].append(image_path)
                                data["image_gsd"].append(image_gsd)
                                data["patch_points"].append(patch_points)
                                data["instance_type"].append("natural")
                                data["instance_object"].append(instance_object)
                                data["instance_class"].append(instance_class)
                                data["instance_crop"].append(instance_crop)
                                data["instance_mask"].append(instance_mask)
                                data["instance_shape_type"].append(shape["shape_type"])
                                data["instance_overlap"].append(instance_overlap)
                                instance_in_patch = True
                                
                            # if there was no instancs in the patch add it as background
                            if instance_in_patch == False:
                                data["image_path"].append(image_path)
                                data["image_gsd"].append(image_gsd)
                                data["patch_points"].append(patch_points)
                                data["instance_type"].append("natural")
                                data["instance_object"].append("null")
                                data["instance_class"].append("background")
                                data["instance_crop"].append("null")
                                data["instance_mask"].append("null")
                                data["instance_shape_type"].append("null")
                                data["instance_overlap"].append(1.0)
        # convert dictionary to dataframe
        data = pd.DataFrame(data=data)
        # class abundance
        total_class_count = (
            data
            .drop_duplicates(["image_path", "instance_mask"])
            .instance_class
            .value_counts())
        print("\tClass Count:\n{}\n".format(total_class_count))
        # polygon abundance
        polygon_class_count = (
            data
            .query("instance_shape_type == 'polygon'")
            .drop_duplicates(["image_path", "instance_mask"])
            .instance_class
            .value_counts())
        print("\tPolygon Class Count:\n{}\n".format(polygon_class_count))
        # only include species with enough data
        total_count_threshold = list(total_class_count.index[total_class_count.gt(min_instances_class)])
        polygon_count_threshold = list(polygon_class_count.index[polygon_class_count.gt(min_polygons_class)])
        included_classes = list(set(total_count_threshold) & set(polygon_count_threshold))
        print("Included Classes: ", included_classes)
        # count class instances per patch
        instances_per_patch = (
            data
            .query("instance_class.isin({0})".format(included_classes), engine="python")
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
        test_sample = balance_instances(instances_per_patch, max_instances_class_test, train_test_ratio)
        # convert test back into patch data
        test_merge = pd.merge(data, test_sample[["image_path", "patch_points"]], indicator=True, how='outer')
        test_instances = (
            test_merge
            .query('_merge=="both"')
            .drop('_merge', axis=1)
            .reset_index(drop = True))
        print("Test Instances:\n{}\n".format(test_instances.query("instance_overlap >= {0}".format(min_overlap)).query("instance_class.isin({0})".format(included_classes), engine="python").instance_class.value_counts()))
        print("Test patches:\n{}\n".format(test_instances.groupby(["image_path", "patch_points"]).ngroups))
        # drop test_instances from data
        data = (
            test_merge
            .query('_merge=="left_only"')
            .drop('_merge', axis=1)
            .reset_index(drop = True))
        # drop test_instances from instances_per_patch
        instances_per_patch = (
            pd.merge(instances_per_patch, test_instances, indicator=True, how='outer')
            .query('_merge=="left_only"')
            .drop('_merge', axis=1)
            .reset_index(drop = True))
        # get train_masks
        train_masks = (
            data
            .query("instance_class.isin({0})".format(included_classes), engine="python")
            .query("instance_shape_type == 'polygon'")
            .query("~instance_class.str.contains('unknown')", engine="python")
            .query("instance_overlap >= {0}".format(min_overlap))
            .drop_duplicates(["image_path", "instance_mask"])
            .reset_index(drop = True))
        # generate train dataset
        train_sample = balance_instances(instances_per_patch, max_instances_class_train, 1)
        # convert test back into patch data
        train_instances = (
            pd.merge(data, train_sample[["image_path", "patch_points"]], indicator=True, how='outer')
            .query('_merge=="both"')
            .drop('_merge', axis=1)
            .reset_index(drop = True))
        print("Train Instances:\n{}\n".format(train_instances.query("instance_overlap >= {0}".format(min_overlap)).query("instance_class.isin({0})".format(included_classes), engine="python").instance_class.value_counts()))
        print("Train patches:\n{}\n".format(train_instances.groupby(["image_path", "patch_points"]).ngroups))
        # drop train instances from data
        data = data.query("instance_class == 'background'")
        # determine the number of background samples to take from each image
        patches_per_image = (
            data
            .groupby("image_path")
            .size())
        # assign % of background patches to test set for each image
        test_backgrounds_per_image = {}
        for i in range(len(patches_per_image)):
            test_backgrounds_per_image[patches_per_image.index[i]] = int(patches_per_image[i] * train_test_ratio)
        test_backgrounds = (
            data
            .groupby("image_path")
            .apply(lambda group: group.sample(test_backgrounds_per_image[group.name], random_state=1))
            .reset_index(drop = True))
        print("Test Backgrounds: ", len(test_backgrounds))
        # drop test_backgrounds from data
        train_backgrounds = (
            pd.merge(data, test_backgrounds[["image_path", "patch_points"]], indicator=True, how='outer')
            .query('_merge=="left_only"')
            .drop('_merge', axis=1)
            .reset_index(drop = True))
        print("Train Backgrounds: ", len(train_backgrounds))
        # step through backgrounds and paste instances on
        random.seed(42)
        random_state = 42
        train_data = [train_instances]
        for image_path, patches in train_backgrounds.groupby("image_path"):
            image_gsd = patches.iloc[0]["image_gsd"]
            for patch_points, _ in patches.groupby("patch_points"):
                species = random.sample(list(train_masks["instance_class"].unique()), 1)
                instances = (
                    train_masks
                    .query("instance_class.isin({0})".format(species), engine="python")
                    .sample(n = instances_per_background, replace = True, random_state = random_state)
                    .reset_index(drop = True))
                random_state += 1
                # save info to dataframe
                for index, row in instances.iterrows():
                    instance_object, instance_mask = transforms(row["instance_object"], row["instance_crop"], target_gsd, min_gsd, max_gsd)
                    train_row = pd.DataFrame([[image_path, image_gsd, patch_points, instance_object, row["instance_class"], instance_mask, "artificial", 1.0]], columns = ["image_path", "image_gsd", "patch_points", "instance_object", "instance_class", "instance_mask", "instance_type", "instance_overlap"])
                    train_data.append(train_row)
        train = (
            pd.concat(train_data, ignore_index=True)
            .reset_index(drop = True))
        train.name = 'train'
        # join test datasets
        test = (
            pd.concat([test_instances, test_backgrounds], ignore_index=True)
            .reset_index(drop = True))
        test.name = 'test'
        # save data
        save_dataset(train, test)