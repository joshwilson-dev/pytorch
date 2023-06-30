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
import json
import pandas as pd
import random
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import shutil
import hashlib
from ortools.sat.python import cp_model
from shapely.geometry import Polygon

#################
#### Content ####
#################

def crop_mask(image, points):

    # Find the bounding box
    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_x = max(p[0] for p in points)
    max_y = max(p[1] for p in points)

    # Calculate crop size and center coordinates
    width = max_x - min_x
    height = max_y - min_y
    size = max(width, height)
    center_x = min_x + width / 2
    center_y = min_y + height / 2

    # Crop the image
    crop_left = int(center_x - size / 2)
    crop_upper = int(center_y - size / 2)
    crop_right = int(center_x + size / 2)
    crop_lower = int(center_y + size / 2)
    bird = image.crop((crop_left, crop_upper, crop_right, crop_lower)).convert('RGBA')

    # Adjust polygon to crop origin
    origin = [(center_x - size / 2, center_y - size / 2)] * len(points)
    points = tuple(tuple(a - b for a, b in zip(t1, t2)) for t1, t2 in zip(points, origin))

    # Create mask image
    mask_im = Image.new('L', bird.size, 0)
    ImageDraw.Draw(mask_im).polygon(points, outline=1, fill=1)
    mask = np.array(mask_im)

    # Combine image and mask
    new_im_array = np.array(bird)
    new_im_array[:, :, 3] = mask * 255

    # Create final image
    mask = Image.fromarray(new_im_array, 'RGBA')
    return mask

def balance_instances(df, ratio):
    # Create the model
    model = cp_model.CpModel()
    num_rows = df.shape[0]
    
    # Create array containing row selection flags. True if row k is selected, False otherwise
    row_selection = [model.NewBoolVar(f'row_{i}') for i in range(num_rows)]
    
    # Number of instances for each class should be less than class_threshold
    for column in df.columns:
        if "aves" in column:  # Assuming "aves" indicates the class of interest
            model.Add(df[column].dot(row_selection) <= math.floor(ratio * df[column].sum()))
    
    # Maximize the number of selected rows
    model.Maximize(sum(row_selection))
    
    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    # Check if an optimal solution is found
    if status == cp_model.OPTIMAL:
        selected_rows = [i for i in range(num_rows) if solver.Value(row_selection[i])]
        return df.iloc[selected_rows].reset_index(drop=True)
    else:
        raise ValueError("Optimal solution not found.")

def rotate_point(point, centre, deg):
    rotated_point = [
        centre[0] + (point[0]-centre[0])*math.cos(math.radians(deg)) - (point[1]-centre[1])*math.sin(math.radians(deg)),
        centre[1] + (point[0]-centre[0])*math.sin(math.radians(deg)) + (point[1]-centre[1])*math.cos(math.radians(deg))]
    return rotated_point

def transforms(instance, points, gsd, target_gsd):

    # Translate points to instance coordinates
    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_x = max(p[0] for p in points)
    max_y = max(p[1] for p in points)
    width = max_x - min_x
    height = max_y - min_y
    size = max(width, height)
    width_pad = (size - width)/2
    height_pad = (size - height)/2
    points = [(point[0] - min_x + width_pad, point[1] - min_y + height_pad) for point in points]

    min_transform = 0.75
    max_transform = 1.25

    # Quality
    target_scale = gsd/target_gsd
    target_size = tuple(int(dim * target_scale) for dim in instance.size)
    instance = instance.resize(target_size)
    points = [[point[0] * target_scale, point[1] * target_scale] for point in points]

    # Colour
    colour = random.uniform(min_transform, max_transform)
    instance = ImageEnhance.Color(instance)
    instance = instance.enhance(colour)

    # Contrast
    contrast = random.uniform(min_transform, max_transform)
    instance = ImageEnhance.Contrast(instance)
    instance = instance.enhance(contrast)

    # Brightness
    brightness = random.uniform(min_transform, max_transform)
    instance = ImageEnhance.Brightness(instance)
    instance = instance.enhance(brightness)

    # Rotate
    centre = [max(instance.size)/2] * 2
    rotation = random.sample([0, 90, 180, 270], 1)[0]
    if rotation != 0:
        instance = instance.rotate(rotation)
        points = [rotate_point(point, centre, -rotation) for point in points]
    return instance, points

def blackout_instance(image, box):
    box_width = box[2]
    box_height = box[3]
    topleft = (round(box[0]), round(box[1]))
    black_box = Image.new("RGB", (round(box_width), round(box_height)))
    image.paste(black_box, topleft)
    return image

def create_shadow(bird):
    shadow = bird.copy()
    width, height = shadow.size
    max_offset = 0.75
    shadow_data = []
    transparency = random.randint(50, 200)
    for item in shadow.getdata():
        if item[3] > 0:
            shadow_data.append((0, 0, 0, transparency))
        else:
            shadow_data.append(item)
    shadow.putdata(shadow_data)
    x_offset = random.randint(int(-width * max_offset), int(width * max_offset))
    y_offset = random.randint(int(-height * max_offset), int(height * max_offset))
    shadow = shadow.crop((min(x_offset, 0), min(y_offset, 0), max(width, width + x_offset), max(height, height + y_offset)))
    return shadow

def get_pastepoints(instances):
    # Determine average bird size
    bird_sizes = [sum(bird.size)/2 for bird in instances["instance"]]
    bird_size = sum(bird_sizes)/len(bird_sizes)

    # Generate grid
    spacing = range(0, min(int(target_patchsize - bird_size), target_patchsize - 200), min(int(bird_size * 0.75), 200))

    # Randomly sample two points on grid
    try:
        x = random.sample(spacing, instances_per_background)
        y = random.sample(spacing, instances_per_background)
    except:
        print(bird_size, spacing)
    pastepoints = [[x, y] for x, y in zip(x, y)]
    return pastepoints

def paste_instances(instance, patch, pastepoints, index):
    location = (pastepoints[index][0], pastepoints[index][1])

    # Paste shadow under bird
    patch = patch.convert("RGBA")
    if random.uniform(0, 1) < 0.3:
        shadow = create_shadow(instance)
        patch.paste(shadow, location, shadow)

    # Paste bird
    patch.paste(instance, location, instance)
    patch = patch.convert("RGB")
    return patch

def save_dataset(train, test):
    coco_categories = []
    for dataset in [train, test]:

        dir = dataset.name

        # Determine total images to print
        total_images = dataset.groupby(['imagepath', 'patchpoints']).ngroups

        # Coco annotation setup
        coco = {"images": [], "annotations": [], "categories": coco_categories}
        coco_path = "dataset/annotations/instances_{}.json".format(dir)
        coco_image_id = 0
        coco_category_id = 0
        coco_instance_id = 0

        dataset = dataset.groupby('imagepath')
        # Iterate images
        for imagepath, patches in dataset:

            # Load image
            image = Image.open(imagepath)

            # Resize image to target gsd
            gsd = patches["gsd"].iloc[0]
            scale = gsd/target_gsd
            size = tuple(int(dim * scale) for dim in image.size)
            image = image.resize((size))

            # Iterate over patches
            patches = patches.groupby("patchpoints")
            for patchpoints, annotations in patches:
                labelme_shapes = []
                complete = annotations.iloc[0]["complete"]

                # Print progress
                coco_image_id += 1
                if coco_image_id % int(total_images / 10) == 0:
                    print("Saving {} image {} of {}".format(dir, coco_image_id, total_images))

                # Crop the image to the patch
                patchpoints = [(x * scale, y * scale) for x, y in patchpoints]
                croppoints = (
                    patchpoints[0][0],
                    patchpoints[0][1],
                    patchpoints[1][0],
                    patchpoints[2][1])
                patch = image.crop(croppoints)

                # If the patch is artificial, generate paste points
                if complete == "artificial":
                    pastepoints = get_pastepoints(annotations)
                
                # Iterate over instances
                for index, annotation in annotations.reset_index(drop = True).iterrows():

                    label = annotation["label"]

                    if label != "background":
                        # Get annotation data
                        points = annotation["points"]
                        instance = annotation["instance"]
                        overlap = annotation["overlap"]
                        shapetype = annotation["shapetype"]
                        area = annotation["area"]

                        # Paste instance onto background
                        if complete == "artificial":
                            patch = paste_instances(instance, patch, pastepoints, index)
                            points = [[point[0] + pastepoints[index][0], point[1] + pastepoints[index][1]] for point in points]
                        
                        # Rescale the mask
                        else:
                            points = [[int((point[0] * scale - patchpoints[0][0])), int((point[1] * scale - patchpoints[0][1]))] for point in points]
                        
                        # Generate the box
                        xmin = min(points, key=lambda x: x[0])[0]
                        xmax = max(points, key=lambda x: x[0])[0]
                        ymin = min(points, key=lambda x: x[1])[1]
                        ymax = max(points, key=lambda x: x[1])[1]
                        coco_instance_box = [xmin, ymin, xmax - xmin, ymax - ymin]
                        labelme_instance_box = [[xmin, ymin], [xmax, ymax]]

                        # Blackout instances that aren't included
                        if label not in included_classes or\
                            overlap < min_overlap: #or shapetype != "polygon":
                            patch = blackout_instance(patch, coco_instance_box)
                            continue

                        # Save instance labelme annotation
                        labelme_annotation = {
                            "label": label,
                            "shape_type": shapetype,
                            "group_id": 'null',
                            "flags": {}}
                        if shapetype == "rectangle":
                            labelme_annotation["points"] = labelme_instance_box
                        else:
                            labelme_annotation["points"] = points
                        labelme_shapes.append(labelme_annotation)

                        # Save instance coco annotation
                        category_id = 0
                        for cat in coco["categories"]:
                            if label == cat["name"]:
                                category_id = cat["id"]
                                continue
                        if category_id == 0:
                            coco_category_id += 1
                            category_id = coco_category_id
                            coco_category = {
                                "id": category_id,
                                "name": label}
                            coco_categories.append(coco_category)
                        coco_instance_id += 1
                        coco_annotation = {
                            "iscrowd": 0,
                            "image_id": coco_image_id,
                            "category_id": category_id,
                            "segmentation": [[item for sublist in points for item in sublist]],
                            "bbox": coco_instance_box,
                            "id": coco_instance_id,
                            "area": area}
                        coco["annotations"].append(coco_annotation)

                # Save patch
                patch_name = hashlib.md5(patch.tobytes()).hexdigest() + ".jpg"
                patch_path = os.path.join("dataset", dir, patch_name)
                patchsize = patch.size
                patch.save(patch_path)

                # Store coco info
                coco_image_info = {
                    "height": patchsize[1],
                    "width": patchsize[0],
                    "id": coco_image_id,
                    "file_name": patch_name}
                coco["images"].append(coco_image_info)
                
                # Save labeleme
                labelme_path = os.path.splitext(patch_path)[0]+'.json'
                labelme = {
                    "version": "5.0.1",
                    "flags": {},
                    "shapes": labelme_shapes,
                    "imagePath": patch_name,
                    "imageData": 'null',
                    "imageHeight": patchsize[1],
                    "imageWidth": patchsize[0]}
                labelme = json.dumps(labelme, indent = 2).replace('"null"', 'null')
                with open(labelme_path, 'w') as labelme_file:
                    labelme_file.write(labelme)

        # Save index to label if it's the last loop
        if dir == "test":
            index_to_class = {}
            for category in coco["categories"]:
                label = json.loads(category["name"].replace("'", '"'))
                category_info = {}
                for key in label.keys():
                    category_info[key] = label[key]
                index_to_class[category["id"]] = category_info
            with open('dataset/annotations/index_to_class.json', 'w') as file:
                file.write(json.dumps(index_to_class, indent = 2))

        # Save coco
        coco = json.dumps(coco, indent = 2)
        with open(coco_path, 'w') as coco_file:
            coco_file.write(coco)

# Define variables
target_gsd = 0.005
min_polygons = 50
train_test_ratio = 0.25
min_overlap = 0.9
target_patchsize = 1300
instances_per_background = 2

# Change to root location
root = "./datasets/bird-mask"
os.chdir(root)

# Create dictionaries to store data
dataset_keys = [
    "imagepath", "gsd", "camera", "latitude", "longitude", "complete", "date", "patchpoints",
    "instance", "label", "points", "shapetype", "overlap", "area", "pose"]
data = {k:[] for k in dataset_keys}

# Crate output directories
paths = ["dataset/train", "dataset/test", "dataset/annotations"]
for path in paths:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

# Iterate through annotation files
for entry in os.scandir("input"):
    if entry.path.endswith(".json"):
        print("Recording Patch Data From: ", entry.path)

        # Load the annotation
        annotation = json.load(open(entry.path))

        # Load the image
        imagepath = os.path.join("input", annotation["imagePath"])
        image = Image.open(imagepath)
        width, height = image.size

        # Get the image metrics
        gsd = annotation["gsd"]
        camera = annotation["camera"]
        latitude = str(round(annotation["latitude"], 1))
        longitude = str(round(annotation["longitude"], 1))
        datetime = annotation["datetime"]
        date = datetime.split(" ")[0]
        complete = annotation["complete"]

        # Determine the scale factor
        scale = target_gsd/gsd
        patchsize = target_patchsize * scale

        # Determine how many crops within accross the image
        n_crops_width = math.ceil(width / patchsize)
        n_crops_height = math.ceil(height / patchsize)

        # Calculate the padded total size of the image
        padded_width = n_crops_width * patchsize
        padded_height = n_crops_height * patchsize

        # Calculate the width of the padding
        pad_width = (padded_width - width) / 2
        pad_height = (padded_height - height) / 2

        # Iterate over the patches and save information
        for height_index in range(n_crops_height):
            for width_index in range(n_crops_width):

                # Determine the corners of the patch
                left = width_index * patchsize - pad_width
                right = left + patchsize
                top = height_index * patchsize - pad_height
                bottom = top + patchsize
                patchpoints = (
                    (left, top),
                    (right, top),
                    (right, bottom),
                    (left, bottom))
                patch_polygon = Polygon(patchpoints)

                # Record data for instances overlapping patch
                background = True
                for shape in annotation["shapes"]:

                    # Get the instance shape_type
                    shapetype = shape["shape_type"]

                    # Get the instance points
                    points = tuple((point[0], point[1]) for point in shape["points"])

                    # Convert rectangle to points format
                    if shapetype == "rectangle":
                        points = tuple((
                            (points[0][0], points[0][1]),
                            (points[1][0], points[0][1]),
                            (points[1][0], points[1][1]),
                            (points[0][0], points[1][1])))
                    
                    # Create rectangle
                    min_x = min(p[0] for p in points)
                    min_y = min(p[1] for p in points)
                    max_x = max(p[0] for p in points)
                    max_y = max(p[1] for p in points)
                    box = tuple((
                        (min_x, min_y),
                        (max_x, min_y),
                        (max_x, max_y),
                        (min_x, max_y)))
                    
                    # Create the instance polygon and determine area
                    polygon = Polygon(box)
                    polygon_area = polygon.area

                    # Check if polygon is in patch, skip if it isn't
                    overlap_area = patch_polygon.intersection(polygon).area
                    if overlap_area == 0: continue

                    # calculate the overlap between the full mask and the patch mask
                    overlap = overlap_area/polygon_area

                    # Get the label and pose
                    label = json.loads(shape["label"].replace("'", '"'))

                    del label["sex"]

                    if label["obscured"] == 'yes':
                        overlap = 0
                    del label["obscured"]

                    pose = label["pose"]
                    del label["pose"]

                    label = json.dumps(label).replace('"', "'")

                    # Get the instance object
                    if shapetype == "polygon":
                        instance = crop_mask(image, points)
                    else:
                        instance = "null"

                    # Save instance information
                    data["imagepath"].append(imagepath)
                    data["gsd"].append(gsd)
                    data["camera"].append(camera)
                    data["latitude"].append(latitude)
                    data["longitude"].append(longitude)
                    data["complete"].append(complete)
                    data["date"].append(date)
                    data["patchpoints"].append(patchpoints)
                    data["instance"].append(instance)
                    data["pose"].append(pose)
                    data["label"].append(label)
                    data["points"].append(points)
                    data["shapetype"].append(shapetype)
                    data["overlap"].append(overlap)
                    data["area"].append(polygon_area)
                    background = False
  
                # Save backgrounds
                if background == True and complete == "true":
                    data["imagepath"].append(imagepath)
                    data["gsd"].append(gsd)
                    data["camera"].append(camera)
                    data["latitude"].append(latitude)
                    data["longitude"].append(longitude)
                    data["complete"].append(complete)
                    data["date"].append(date)
                    data["patchpoints"].append(patchpoints)
                    data["instance"].append("background")
                    data["pose"].append("background")
                    data["label"].append("background")
                    data["points"].append("background")
                    data["shapetype"].append("background")
                    data["overlap"].append(1)
                    data["area"].append("background")

# Convert dictionary to dataframe
data = pd.DataFrame(data=data)

# Determine the total abundance of each label
label_count = (
    data
    .query("~label.str.contains('unknown')", engine="python")
    .drop_duplicates(subset=['imagepath', 'points'])
    .label
    .value_counts())

print("\tLabel Class Count:\n{}\n".format(label_count.to_string()))

# Determine the abundance of each polygon label
polygon_count = (
    data
    .query("shapetype == 'polygon'")
    .query("~label.str.contains('unknown')", engine="python")
    .drop_duplicates(subset=['imagepath', 'points'])
    .label
    .value_counts())

print("\tPolygon Class Count:\n{}\n".format(polygon_count.to_string()))

# Only include labels with enough polygons
included_classes = polygon_count[polygon_count > min_polygons].index.tolist() + ["background"]

print("Included Classes: ", included_classes)

# Count the label instances per patch
instances_per_patch = (
    data
    .query("complete == 'true'")
    # .query("shapetype == 'polygon'")
    .query("overlap >= @min_overlap")
    .query("label != 'background'")
    .query("label.isin({0})".format(included_classes), engine="python")
    .assign(label=lambda df: df["label"] + df["pose"])
    .groupby(['imagepath', 'patchpoints', 'label'])
    .size()
    .reset_index(name='counts')
    .pivot_table(
        index=['imagepath', 'patchpoints'],
        columns=["label"],
        values="counts",
        fill_value=0)
    .reset_index())

# Generate the positive test dataset
test_instances = balance_instances(instances_per_patch, train_test_ratio)

# Drop test instances from instances_per_patch
instances_per_patch = (
    instances_per_patch
    .merge(test_instances, indicator=True, how= "left")
    .query('_merge=="left_only"')
    .drop('_merge', axis=1)
    .reset_index(drop = True))

# Convert test back into patch data
test_instances = (
    pd.merge(data, test_instances[['imagepath', 'patchpoints']], indicator=True, how='left')
    .query('_merge=="both"')
    .drop('_merge', axis=1)
    .reset_index(drop = True))
n_test_foreground = len(test_instances.drop_duplicates(['imagepath', 'patchpoints']))
print("Foreground test patches: ", n_test_foreground)

# Get train instances
train_instances = balance_instances(instances_per_patch, 1)

# Convert train back into patch data
train_instances = (
    pd.merge(data, train_instances[['imagepath', 'patchpoints']], indicator=True, how='left')
    .query('_merge=="both"')
    .drop('_merge', axis=1)
    .reset_index(drop = True))
n_train_natural = len(train_instances.drop_duplicates(['imagepath', 'patchpoints']))
print("Natural train patches: ", n_train_natural)

# Get training masks
train_masks = (
    pd.merge(data, test_instances[['imagepath', 'patchpoints']], indicator=True, how='left')
    .query('_merge=="left_only"')
    .drop('_merge', axis=1)
    .query("label.isin({0})".format(included_classes), engine="python")
    .query("shapetype == 'polygon'")
    .drop_duplicates(["imagepath", "points"])
    .assign(gsd_cat=lambda df: str(round(df["gsd"], 3)))
    .assign(survey=lambda df: df["date"] + df["camera"] + df["latitude"] + df["longitude"])
    .assign(filter = lambda df: df["label"] + df["pose"] + df["survey"] + df["gsd_cat"] + df["imagepath"])
    .reset_index(drop = True))

# Get number of locations
locations = (
    data
    .query("complete == 'true'")
    .query("label == 'background'")
    .groupby(["latitude", "longitude"]))
print("Locations:", locations.ngroups)

# Get number of background samples per location
backgrounds_per_loc = (n_train_natural + n_test_foreground) / locations.ngroups

# Sample backgrounds
backgrounds = (
    data
    .query("complete == 'true'")
    .query("label == 'background'")
    .sample(frac=1)
    .groupby(["latitude", "longitude"])
    .head(5 * int(backgrounds_per_loc))
    .reset_index(drop = True))
n_total_backgrounds = len(backgrounds)
print("Total background patches: ", n_total_backgrounds)

# Get test backgrounds
test_backgrounds = backgrounds.sample(int(n_total_backgrounds * train_test_ratio))
n_test_background = len(test_backgrounds)
print("Background test patches: ", n_test_background)

# Drop the test backgrounds from backgrounds
train_backgrounds = backgrounds.drop(index = test_backgrounds.index).reset_index(drop=True)
n_train_artificial = len(train_backgrounds)
print("Artificial train patches: ", n_train_artificial)

# Create nested dictionaries for train masks using groupby
grouped_data = train_masks.groupby(['label', 'pose', 'survey', 'gsd_cat', 'imagepath'])
filters = {}
for (label, pose, survey, gsd_cat, imagepath), _ in grouped_data:
    filters.setdefault(label, {}).setdefault(pose, {}).setdefault(survey, {}).setdefault(gsd_cat, {})[imagepath] = {}

# Loop over artificial backgrounds and add instances
random_state = 0
random.seed(random_state)
train_rows = []
for i in range(n_train_artificial):
    imagepath = train_backgrounds.iloc[i]["imagepath"]
    gsd = train_backgrounds.iloc[i]["gsd"]
    patchpoints = train_backgrounds.iloc[i]["patchpoints"]

    # Print progress
    if i % int(n_train_artificial / 10) == 0:
        print("Pasting instances onto background {} of {}".format(i, n_train_artificial))

    # Equally represent each label
    label_filter = random.sample(filters.keys(), 1)[0]

    # Equally represent each pose, within each label
    pose_filter = random.sample(filters[label_filter].keys(), 1)[0]

    for _ in range(instances_per_background):
        # Equally represent each survey within each pose
        survey_filter = random.sample(filters[label_filter][pose_filter].keys(), 1)[0]

        # Equally represent each gsd within each survey
        gsd_filter = random.sample(filters[label_filter][pose_filter][survey_filter].keys(), 1)[0]

        # Equally sample from images
        image_filter = random.sample(filters[label_filter][pose_filter][survey_filter][gsd_filter].keys(), 1)[0]

        # Randomly pick one of the remaining instances 
        filter_value = label_filter + pose_filter + survey_filter + gsd_filter + image_filter
        annotations = (
            train_masks
            .query("filter == @filter_value", engine="python")
            .sample(n = 1, random_state = random_state)
            .reset_index(drop = True))

        # Update random state
        random_state += 1
        random.seed(random_state)

        # Save info to dataframe
        for index, annotation in annotations.iterrows():
            camera = annotation["camera"]
            latitude = annotation["latitude"]
            longitude = annotation["longitude"]
            date = annotation["date"]
            label = annotation["label"]
            area = annotation["area"]
            pose = annotation["pose"]
            instance, points = transforms(annotation["instance"], annotation["points"], annotation["gsd"], target_gsd)
            train_rows.append([imagepath, patchpoints, gsd, camera, latitude, longitude, "artificial", date, instance, label, points, "polygon", 1, area, pose])

# Join train datasets
train_artificial = pd.DataFrame(train_rows, columns=["imagepath", "patchpoints", "gsd", "camera", "latitude", "longitude", "complete", "date", "instance", "label", "points", "shapetype", "overlap", "area", "pose"])

train = (
    pd.concat([train_instances, train_artificial], ignore_index=True)
    .reset_index(drop = True))
train.name = 'train'

# Join test datasets
test = (
    pd.concat([test_instances, test_backgrounds], ignore_index=True)
    .reset_index(drop = True))
test.name = 'test'

# Save data
save_dataset(train, test)