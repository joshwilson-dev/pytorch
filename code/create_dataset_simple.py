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
import pandas as pd
from PIL import Image
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
    # Maximise the number of patches
    model.Maximize(pd.Series([1]*len(df)).dot(row_selection))
    # Solve
    solver = cp_model.CpSolver()
    solver.Solve(model)
    # Get the rows selected
    rows = [row for row in range(df.shape[0]) if solver.Value(row_selection[row])]
    return df.iloc[rows, :].reset_index()

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
                instances = instances.reset_index(drop = True)
                for index, instance in instances.iterrows():
                    # if its not a background
                    if instance["instance_class"] != "background":
                        # scale points to target gsd
                        instance_mask = [[int((point[0] - patch_points[0]) * scale), int((point[1] - patch_points[1]) * scale)] for point in instance["instance_mask"]]
                        # generate the instance box
                        xmin = min(instance_mask, key=lambda x: x[0])[0]
                        xmax = max(instance_mask, key=lambda x: x[0])[0]
                        ymin = min(instance_mask, key=lambda x: x[1])[1]
                        ymax = max(instance_mask, key=lambda x: x[1])[1]
                        coco_instance_box = [xmin, ymin, xmax - xmin, ymax - ymin]
                        labelme_instance_box = [[xmin, ymin], [xmax, ymax]]
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
                            "area": instance["original_area"],
                            "resized_area": instance["resized_area"]}
                        if len(instance_mask) == 4:
                            labelme_annotation["points"] = labelme_instance_box
                            labelme_annotation["shape_type"] = "rectangle"
                        else:
                            labelme_annotation["points"] = instance_mask
                            labelme_annotation["shape_type"] = "polygon"
                        labelme_shapes.append(labelme_annotation)
                        coco["annotations"].append(coco_annotation)
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
                label = json.loads(category["name"].replace("'", '"'))
                category_info = {
                    "common_name": label["name"],
                    "class": label["class"],
                    "order": label["order"],
                    "family": label["family"],
                    "genus": label["genus"],
                    "species": label["species"],
                    "age": label["age"]}
                index_to_class[category["id"]] = category_info
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
            "image_path", "image_gsd", "patch_points", "instance_class",
            "instance_mask", "instance_overlap", "instance_area", "instance_pose"]
        data = {k:[] for k in dataset_keys}
        # variables
        target_gsd = 0.005
        train_test_ratio = 0.25
        min_overlap = 0.9
        target_patchsize = 800
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
                                instance_mask = shape["points"]
                                # convert box to mask
                                if shape["shape_type"] == "rectangle":
                                    xmin = min(shape["points"], key=lambda x: x[0])[0]
                                    xmax = max(shape["points"], key=lambda x: x[0])[0]
                                    ymin = min(shape["points"], key=lambda x: x[1])[1]
                                    ymax = max(shape["points"], key=lambda x: x[1])[1]
                                    instance_mask = [
                                        [xmin, ymin],
                                        [xmax, ymin],
                                        [xmax, ymax],
                                        [xmin, ymin]]
                                # create instance polygon
                                instance_poly = Polygon(instance_mask)

                                # determine area
                                instance_area = instance_poly.area

                                # check if polygon is in patch
                                instance_intersection = patch_poly.intersection(instance_poly)
                        
                                # if polygon is not in patch skip
                                if instance_intersection.area == 0: continue

                                # calculate the overlap between the full mask and the patch mask
                                instance_overlap = instance_intersection.area/instance_area

                                # get the instance class and shape type
                                label = json.loads(shape["label"].replace("'", '"'))
                                instance_pose = json.dumps(label).replace('"', "'")
                                del label["pose"]
                                instance_class = json.dumps(label).replace('"', "'")

                                # save instance to temp data
                                data["image_path"].append(image_path)
                                data["image_gsd"].append(image_gsd)
                                data["patch_points"].append(patch_points)
                                data["instance_mask"].append(instance_mask)
                                data["instance_pose"].append(instance_pose)
                                data["instance_class"].append(instance_class)
                                data["instance_overlap"].append(instance_overlap)
                                data["instance_area"].append(instance_area)
                                instance_in_patch = True
                                
                            # if there was no instances in the patch add it as background
                            if instance_in_patch == False:
                                data["image_path"].append(image_path)
                                data["image_gsd"].append(image_gsd)
                                data["patch_points"].append(patch_points)
                                data["instance_mask"].append("background")
                                data["instance_pose"].append("background")
                                data["instance_class"].append("background")
                                data["instance_overlap"].append("background")
                                data["instance_area"].append("background")

        # convert dictionary to dataframe
        data = pd.DataFrame(data=data)
        # drop data not included
        included_data = (
            data
            .query("instance_overlap >= {0}".format(min_overlap))
            .query("~instance_class.str.contains('unknown')", engine="python"))
        # class abundance
        instance_count = (
            included_data
            .instance_class
            .value_counts())
        print("\tClass Count:\n{}\n".format(instance_count))       
        # list of inclued species
        included_classes = included_data["istance_class"].unique()
        # count class instances per patch
        instances_per_patch = (
            included_data
            .query("instance_class != 'background'")
            .groupby(['image_path', 'patch_points', 'instance_pose'])
            .size()
            .reset_index(name='counts')
            .pivot_table(
                index = ['image_path', 'patch_points'],
                columns="instance_pose",
                values="counts",
                fill_value=0)
            .reset_index())
        # generate balanced test dataset
        test_sample = balance_instances(instances_per_patch, train_test_ratio)
        # convert test back into patch data
        test_instances = (
            pd.merge(data, test_sample[["image_path", "patch_points"]], indicator=True, how='outer')
            .query('_merge=="both"')
            .drop('_merge', axis=1)
            .reset_index(drop = True))
        print("Test Instances:\n{}\n".format(test_instances.query("instance_shape_type == 'polygon'").query("instance_overlap >= {0}".format(min_overlap)).query("original_area >= {0}".format(min_pixel_area)).query("instance_class.isin({0})".format(included_classes), engine="python").instance_class.value_counts()))
        print("Test patches:\n{}\n".format(test_instances.groupby(["image_path", "patch_points"]).ngroups))
        # drop test_instances from included_data
        included_data = (
            pd.merge(included_data, test_sample[["image_path", "patch_points"]], indicator=True, how='outer')
            .query('_merge=="left_only"')
            .drop('_merge', axis=1)
            .reset_index(drop = True))
        # drop test_instances from instances_per_patch
        instances_per_patch = (
            pd.merge(instances_per_patch, test_instances, indicator=True, how='outer')
            .query('_merge=="left_only"')
            .drop('_merge', axis=1)
            .reset_index(drop = True))
        # generate balanced train dataset
        train_sample = balance_instances(instances_per_patch, 1)
        # convert test back into patch data
        train_instances = (
            pd.merge(data, train_sample[["image_path", "patch_points"]], indicator=True, how='outer')
            .query('_merge=="both"')
            .drop('_merge', axis=1)
            .reset_index(drop = True))
        print("Train Instances:\n{}\n".format(train_instances.query("instance_shape_type == 'polygon'").query("instance_overlap >= {0}".format(min_overlap)).query("original_area >= {0}".format(min_pixel_area)).query("instance_class.isin({0})".format(included_classes), engine="python").instance_class.value_counts()))
        print("Train patches:\n{}\n".format(train_instances.groupby(["image_path", "patch_points"]).ngroups))
        # drop train instances from data
        included_data = included_data.query("instance_class == 'background'")
        # determine the number of background samples to take from each image
        patches_per_image = (
            included_data
            .groupby("image_path")
            .size())
        # assign % of background patches to test set for each image
        test_backgrounds_per_image = {}
        for i in range(len(patches_per_image)):
            test_backgrounds_per_image[patches_per_image.index[i]] = int(patches_per_image[i] * train_test_ratio)
        test_backgrounds = (
            included_data
            .groupby("image_path")
            .apply(lambda group: group.sample(test_backgrounds_per_image[group.name], random_state=1))
            .reset_index(drop = True))
        print("Test Backgrounds: ", len(test_backgrounds))
        # drop test_backgrounds from data
        train_backgrounds = (
            pd.merge(included_data, test_backgrounds[["image_path", "patch_points"]], indicator=True, how='outer')
            .query('_merge=="left_only"')
            .drop('_merge', axis=1)
            .reset_index(drop = True))
        print("Train Backgrounds: ", len(train_backgrounds))
        train = (
            pd.concat([train_instances, train_backgrounds], ignore_index=True)
            .reset_index(drop = True))
        train.name = 'train'
        # join test datasets
        test = (
            pd.concat([test_instances, test_backgrounds], ignore_index=True)
            .reset_index(drop = True))
        test.name = 'test'
        # save data
        save_dataset(train, test)