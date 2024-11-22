# Title:        CS1 - Balanced Sampling
# Description:  Pre-process images of birds captured from a drone to prepare
#               them for training an object detection network.
#               Step 1: Loop over annotation files and get metrics for each bird
#               Step 2: Use contraint optimisation to select images for the
#                       validation, train, and test datasets.
#               Step 3: Save selected images and annotation files.
# Author:       Anonymous
# Date:         05/06/2024

#### Import packages
# In base Python
import os
import shutil
import json
import math
# Not in base python
import pandas as pd
from PIL import Image
from ortools.sat.python import cp_model
from shapely.geometry import Polygon

#### Define variables
# Set root (directory containing the original images)
root = "Detecting_and_identifying_birds_in_images_captured_using_a_drone"

# Create directories
directories = ["train", "test", "validation", "annotations"]
for directory in directories:
    path = os.path.join(root, "balanced", directory)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

# Define parameters
validation_split = 0.2
train_test_split = 0.8
min_instances = 0
max_fg = 1000
max_bg = 100
max_train_fg = max_fg * train_test_split
max_test_fg = max_fg * (1 - train_test_split)
max_train_bg = max_bg * train_test_split
max_test_bg = max_bg * (1 - train_test_split)
min_overlap = 0.7
target_gsd = 0.005
target_panelsize = 800

# Create dataset within which we will save annotation data
dataset_keys = [
    "imagepath", "originalimagePath", "uav", "droneheight", "birdheight",
    "camera", "sensorwidth", "sensorheight", "focallength", "imagewidth",
    "imageheight", "gsd", "latitude", "longitude", "location", "datetime",
    "panelcorners", "label", "points", "shapetype", "overlap", "commonname",
    "class", "order", "family", "genus", "species", "pose", "age", "obscured",
    "area"]
data = {k:[] for k in dataset_keys}

#### Define functions
# Define function to create validation set
def validate(df, ratio):
    # Samples locations to maximise instances without sampleing more than ratio 
    # Create the model
    model = cp_model.CpModel()
    num_rows = df.shape[0]
    
    # Create array containing row selection flags.
    row_choice = [model.NewBoolVar(f'row_{i}') for i in range(num_rows)]

    # Get a list of labels
    labels = [k for k in list(df.columns.levels[0].unique()) if 'name' in k]
    labels = labels

    # Iterate over labels and calculate maximum sample
    for label in labels:

        # Count instances per class
        label_df = df[label]
        label_count = label_df.values.sum()

        # Sample less than count * ratio
        label_sample = round(label_count * ratio)
        model.Add(df[label].sum(axis = 1).dot(row_choice) <= label_sample)

    # Maximise the number of instances selected
    model.Maximize(df.sum(axis=1).dot(row_choice))

    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    # Check if an optimal solution is found
    if status == cp_model.OPTIMAL:
        selected_rows = [
            i for i in range(num_rows) if solver.Value(row_choice[i])]
        return df.iloc[selected_rows]
    else:
        raise ValueError("Optimal solution not found.")

# Define function to subsample dataset in an equal way
def balanced_subsample(df, max_fg, max_bg, ratio):
    # 1.    Calculate the number of instances to sample from each label
    # 2.    Calculate the number of instances to sample from each pose within
    #       each label to get the required number of instances for that label
    #       prioritising equal number of instances among poses within labels.
    # 3.    Calculate the number of instances to sample from each area within
    #       each pose to get the required number of instances for that pose
    #       prioritising equal number of instances among areas within poses.
    
    # Create the model
    model = cp_model.CpModel()
    num_rows = df.shape[0]
    
    # Create array containing row choice flags.
    row_choice = [model.NewBoolVar(f'row_{i}') for i in range(num_rows)]

    # Get a list of labels
    labels = [k for k in list(df.columns.levels[0].unique()) if 'name' in k]
    labels = labels

    # Iterate over labels and calculate maximum sample
    for label in labels:
        if "background" in label:
            max_sample = max_bg
        else:
            max_sample = max_fg

        # Sample min of max_instances * ratio or total_instances * ratio
        label_df = df[label]
        label_count = label_df.values.sum()

        # If there's only one panel with instances make sure it's included
        label_min = label_df.sum(axis=1)[label_df.sum(axis=1) > 0].min()
        if pd.isna(label_min): label_min = 0
        if label_min == label_count:
            model.Add(df[label].sum(axis = 1).dot(row_choice) >= label_min)
            continue
        label_sample = round(min(label_count * ratio, max_sample))
        print("{}: {}/{}".format(label, label_sample, label_count))

        # Sample poses as equally as possible within each label
        unique_poses = list(label_df.columns.unique(level = 'pose'))
        pose_counts = []
        for pose in unique_poses:
            pose_counts.append(label_df[pose].values.sum())
        # Zip the two lists together
        combined = list(zip(unique_poses, pose_counts))
        # Sort the combined list based on the counts (second element)
        sorted_combined = sorted(combined, key=lambda x: x[1])
        # Unzip the sorted list
        unique_poses, pose_counts = zip(*sorted_combined)

        pose_required = label_sample
        for pose in unique_poses:
            pose_df = label_df[pose]
            pose_count = pose_df.values.sum()
            pose_sample = 0
            # The number of instances to sample from a pose is the number of
            # instances required for this label, divided by the number of poses
            # left to sample from. Poses are ordered from least to most common.
            index = unique_poses.index(pose)
            req_avg = pose_required/(len(unique_poses) - index)
            pose_sample = min(pose_count * ratio, req_avg)
            pose_required -= pose_sample
            print("\t{}: {}/{}".format(pose, pose_sample, pose_count))
            
            # Sample areas as equally as possible within each pose
            unique_areas = list(pose_df.columns.unique(level = 'area_bins'))
            area_counts = []
            for area in unique_areas:
                area_counts.append(pose_df[area].values.sum())
            # Zip the two lists together
            combined = list(zip(unique_areas, area_counts))
            # Sort the combined list based on the counts (second element)
            sorted_combined = sorted(combined, key=lambda x: x[1])
            # Unzip the sorted list
            unique_areas, area_counts = zip(*sorted_combined)
            area_required = pose_sample
            for area in unique_areas:
                area_df = pose_df[area]
                area_count = area_df.values.sum()
                area_sample = 0
                # See comment on line 94 for description of how area is sampled.
                index = unique_areas.index(area)
                req_avg = area_required/(len(unique_areas) - index)
                area_sample = math.ceil(min(area_count * ratio, req_avg))
                area_required -= area_sample
                print("\t\t\t{}: {}/{}".format(area, area_sample, area_count))
                model.Add(df[label][pose][area].dot(row_choice) <= area_sample)
    
    # Maximise the number of instances selected
    model.Maximize(df.sum(axis=1).dot(row_choice))

    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    # Check if an optimal solution is found
    if status == cp_model.OPTIMAL:
        selected_rows = [
            i for i in range(num_rows) if solver.Value(row_choice[i])]
        return df.iloc[selected_rows].index
    else:
        raise ValueError("Optimal solution not found.")

def blackout_instance(image, box):
    # Pastes a black box onto an image 
    box_width = box[2]
    box_height = box[3]
    topleft = (round(box[0]), round(box[1]))
    black_box = Image.new("RGB", (round(box_width), round(box_height)))
    image.paste(black_box, topleft)
    return image

def save_dataset(train, test, validation):

    # Specifying index to class
    index_to_class = json.load(open("AS1_index_to_class.json"))
    index_to_class_path = os.path.join(
        root, "balanced/annotations/AS1_index_to_class.json")

    # Creating class to index list
    class_to_index = [value for value in index_to_class.values()]

    # Creating coco categories
    coco_categories = []
    for key, value in index_to_class.items():
        coco_categories.append(
            {"id": int(key), "name": json.dumps(value).replace('"', "'")})
    
    # Set image and instance id so that they continuously increase
    # then you can easily join datasets later 
    coco_image_id = 1
    coco_instance_id = 1

    for dataset in [train, test, validation]:
        image_number = 0
        dir = dataset.name

        # Determine total images to print
        total_images = dataset.groupby(['imagepath', 'panelcorners']).ngroups

        # Coco annotation setup
        coco = {"images": [], "annotations": [], "categories": coco_categories}
        coco_path = root + "/balanced/annotations/{}.json".format(dir)

        # Step through images, crop, save, and create annotation
        dataset = dataset.groupby('imagepath')
        for imagepath, panels in dataset:
            # Get original image path
            originalimagePath = panels["originalimagePath"].iloc[0]
            intermediateImagePath = os.path.basename(imagepath)

            # Load image
            image = Image.open(imagepath)

            # Resize image to target gsd
            gsd = panels["gsd"].iloc[0]
            scale = gsd/target_gsd
            size = tuple(int(dim * scale) for dim in image.size)
            image = image.resize((size))

            # Iterate over panels
            panels = panels.groupby("panelcorners")
            for panelcorners, annotations in panels:
                labelme_shapes = []

                # Print progress
                if image_number % int(total_images / 5) == 0:
                    print(
                        "Saving {} panel {}/{}"
                        .format(dir, image_number, total_images))
                image_number += 1

                # Crop the image to the panel
                panelcorners = [(x * scale, y * scale) for x, y in panelcorners]
                croppoints = (
                    panelcorners[0][0],
                    panelcorners[0][1],
                    panelcorners[1][0],
                    panelcorners[2][1])
                panel = image.crop(croppoints)
                
                # Iterate over instances
                for _, annotation in (
                    annotations.reset_index(drop = True).iterrows()):
                    label = annotation["label"]

                    if "background" not in label:
                        # Get annotation data
                        points = annotation["points"]
                        overlap = annotation["overlap"]
                        obscured = annotation["obscured"]
                        count = annotation["count"]
                        shapetype = annotation["shapetype"]

                        # Always make it a rectange because we don't need masks
                        shapetype = "rectangle"

                        # Rescale the mask
                        points = [
                            [int((point[0] * scale - panelcorners[0][0])),
                            int((point[1] * scale - panelcorners[0][1]))]
                            for point in points]
                        
                        # Generate the box and crop to panel
                        xmin = max(min(points, key=lambda x: x[0])[0], 0)
                        xmax = min(
                            max(points, key=lambda x: x[0])[0],
                            target_panelsize)
                        ymin = max(min(points, key=lambda x: x[1])[1], 0)
                        ymax = min(
                            max(points, key=lambda x: x[1])[1],
                            target_panelsize)
                        coco_instance_box = [
                            xmin,
                            ymin,
                            xmax - xmin,
                            ymax - ymin]
                        labelme_instance_box = [[xmin, ymin], [xmax, ymax]]

                        # Blackout instances that have low instances or overlap
                        if count < min_instances or \
                            overlap < min_overlap or \
                            obscured == 'yes' or \
                            'unknown' in label:
                            panel = blackout_instance(panel, coco_instance_box)
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
                        coco_annotation = {
                            "iscrowd": 0,
                            "image_id": coco_image_id,
                            "category_id": class_to_index.index(
                                json.loads(label.replace("'", '"'))) + 1,
                            "segmentation": [
                                [item for lst in points for item in lst]],
                            "bbox": coco_instance_box,
                            "id": coco_instance_id,
                            "area": annotation["area"]}
                        coco["annotations"].append(coco_annotation)
                        coco_instance_id += 1

                # Save panel
                panel_name = str(coco_image_id) + ".jpg"
                panel_path = os.path.join(root, "balanced", dir, panel_name)
                panelsize = panel.size
                panel.save(panel_path)

                # Store coco info
                coco_image_info = {
                    "height": panelsize[1],
                    "width": panelsize[0],
                    "id": coco_image_id,
                    "file_name": panel_name}
                coco["images"].append(coco_image_info)
                coco_image_id += 1
                
                # Save labeleme
                labelme_path = os.path.splitext(panel_path)[0]+'.json'
                labelme = {
                    "version": "5.0.1",
                    "flags": {},
                    "shapes": labelme_shapes,
                    "imagePath": panel_name,
                    "imageData": 'null',
                    "imageHeight": panelsize[1],
                    "imageWidth": panelsize[0],
                    "originalImagePath": originalimagePath,
                    "intermediateImagePath": intermediateImagePath}
                labelme = json.dumps(labelme, indent = 2).replace(
                    '"null"',
                    'null')
                with open(labelme_path, 'w') as labelme_file:
                    labelme_file.write(labelme)

        # Save coco
        coco = json.dumps(coco, indent = 2)
        with open(coco_path, 'w') as coco_file:
            coco_file.write(coco)

    # Save index_to_class
    index_to_class = json.dumps(index_to_class, indent = 2)
    with open(index_to_class_path, 'w') as index_to_class__file:
        index_to_class__file.write(index_to_class)
    return

# Extract information from each image
for entry in os.scandir(root):
    if entry.path.endswith(".json"):
        print("Recording Panel Data From: ", entry.path)

        # Load the annotation
        annotation = json.load(open(entry.path))

        # Load the image
        imagepath = os.path.join(root, annotation["imagePath"])
        image = Image.open(imagepath)
        imagewidth, imageheight = image.size

        # Get the image metrics
        gsd = annotation["gsd"]
        latitude = annotation["latitude"]
        longitude = annotation["longitude"]
        uav = annotation["uav"]
        camera = annotation["camera"]
        droneheight = annotation["drone_height"]
        birdheight = annotation["bird_height"]
        originalimagePath = annotation["originalimagePath"]
        sensorwidth = annotation["sensorwidth"]
        sensorheight = annotation["sensorheight"]
        focallength = annotation["focallength"]
        datetime = annotation["datetime"]

        # Create location feature
        location = str(round(latitude, 3)) + ", " + str(round(longitude, 3))

        # Determine the scale factor
        scale = target_gsd/gsd
        panelsize = target_panelsize * scale

        # Determine how many crops within accross the image
        n_crops_width = math.ceil(imagewidth / panelsize)
        n_crops_height = math.ceil(imageheight / panelsize)

        # Calculate the padded total size of the image
        padded_width = n_crops_width * panelsize
        padded_height = n_crops_height * panelsize

        # Calculate the width of the padding
        pad_width = (padded_width - imagewidth) / 2
        pad_height = (padded_height - imageheight) / 2

        # Iterate over the panels and save information
        for height_index in range(n_crops_height):
            for width_index in range(n_crops_width):

                # Determine the corners of the panel
                left = width_index * panelsize - pad_width
                right = left + panelsize
                top = height_index * panelsize - pad_height
                bottom = top + panelsize
                panelcorners = (
                    (left, top),
                    (right, top),
                    (right, bottom),
                    (left, bottom))
                panel_polygon = Polygon(panelcorners)

                # Record data for instances overlapping panel
                background = True
                for shape in annotation["shapes"]:

                    # Get the instance shape_type
                    shapetype = shape["shape_type"]

                    # Get the instance points
                    points = tuple(
                        (point[0], point[1]) for point in shape["points"])

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

                    # Check if polygon is in panel, skip if it isn't
                    overlap_area = panel_polygon.intersection(polygon).area
                    if overlap_area == 0: continue

                    # calculate the overlap btwn the full mask and the crop mask
                    overlap = overlap_area/polygon.area

                    # Get the label and pose
                    label = json.loads(shape["label"].replace("'", '"'))

                    order = label["order"]
                    family = label["family"]
                    genus = label["genus"]
                    species = label["species"]
                    age = label["age"]
                    commonname = label["name"]

                    obscured = label["obscured"]
                    del label["obscured"]

                    pose = label["pose"]
                    del label["pose"]

                    label = json.dumps(label).replace('"', "'")

                    # Save instance information
                    data["imagepath"].append(imagepath)
                    data["originalimagePath"].append(originalimagePath)
                    data["uav"].append(uav)
                    data["droneheight"].append(droneheight)
                    data["birdheight"].append(birdheight)
                    data["camera"].append(camera)
                    data["sensorwidth"].append(sensorwidth)
                    data["sensorheight"].append(sensorheight)
                    data["focallength"].append(focallength)
                    data["imagewidth"].append(imagewidth)
                    data["imageheight"].append(imageheight)
                    data["gsd"].append(gsd)
                    data["latitude"].append(latitude)
                    data["longitude"].append(longitude)
                    data["location"].append(location)
                    data["datetime"].append(datetime)
                    data["panelcorners"].append(panelcorners)
                    data["label"].append(label)
                    data["commonname"].append(commonname)
                    data["class"].append('aves')
                    data["order"].append(order)
                    data["family"].append(family)
                    data["genus"].append(genus)
                    data["species"].append(species)
                    data["pose"].append(pose)
                    data["age"].append(age)
                    data["obscured"].append(obscured)
                    data["points"].append(points)
                    data["shapetype"].append(shapetype)
                    data["area"].append(polygon.area)
                    data["overlap"].append(overlap)
                    background = False
  
                # Save backgrounds
                if background == True:
                    data["imagepath"].append(imagepath)
                    data["originalimagePath"].append(originalimagePath)
                    data["uav"].append(uav)
                    data["droneheight"].append(droneheight)
                    data["birdheight"].append(birdheight)
                    data["camera"].append(camera)
                    data["sensorwidth"].append(sensorwidth)
                    data["sensorheight"].append(sensorheight)
                    data["focallength"].append(focallength)
                    data["imagewidth"].append(imagewidth)
                    data["imageheight"].append(imageheight)
                    data["gsd"].append(gsd)
                    data["latitude"].append(latitude)
                    data["longitude"].append(longitude)
                    data["location"].append(location)
                    data["datetime"].append(datetime)
                    data["panelcorners"].append(panelcorners)
                    data["label"].append("name: background - " + location)
                    data["commonname"].append("background")
                    data["class"].append("background")
                    data["order"].append("background")
                    data["family"].append("background")
                    data["genus"].append("background")
                    data["species"].append("background")
                    data["pose"].append("background")
                    data["age"].append("background")
                    data["obscured"].append("no")
                    data["points"].append("background")
                    data["shapetype"].append("background")
                    data["area"].append(panelsize*panelsize)
                    data["overlap"].append(1)

# Convert dictionary to dataframe
data = pd.DataFrame(data=data)
data["dataset"] = "total"

# Create area_bins
bins = list(range(0, 4501, 1500)) + [1e10]
data["area_bins"] = pd.cut(data['area'], bins).astype(str)

# Determine the total abundance of each class
class_count = (
    data
    .query("overlap >= @min_overlap")
    .query("obscured == 'no'")
    .drop_duplicates(subset=['imagepath', 'points'])
    .groupby('label')
    .size())
print("\tClass Count:\n{}\n".format(class_count.to_string()))

# Map the counts back to the original DataFrame based on 'label'
data['count'] = data['label'].map(class_count)

# count the instances per location
instances_per_location = (
    data
    .query("overlap >= @min_overlap")
    .query("obscured == 'no'")
    .query('~label.str.contains("unknown")')
    .query('~label.str.contains("background")')
    .query("count >= @min_instances")
    .groupby(['location', 'label', 'imagepath'])
    .size()
    .reset_index(name='counts')
    .pivot_table(
        index=['location'],
        columns=['label', 'imagepath'],
        values="counts",
        fill_value=0))

validation_images = validate(instances_per_location, validation_split)
validation_images = validation_images.droplevel(0, axis=1)
validation_images = validation_images.loc[
    :,
    (validation_images != 0).any(axis=0)]
validation_images = pd.DataFrame(data = validation_images.columns.unique())

# Convert validation back into panel data
validation = (
    pd.merge(data, validation_images, indicator=True, how='left')
    .query('_merge=="both"')
    .drop('_merge', axis=1)
    .reset_index(drop = True))
validation["dataset"] = 'validation'
validation.name = 'validation'
n_validation = len(validation.drop_duplicates(['imagepath', 'panelcorners']))

# Drop validation from data
train_data = (
    pd.merge(data, validation_images, indicator=True, how='left')
    .query('_merge=="left_only"')
    .drop('_merge', axis=1)
    .reset_index(drop = True))

# Count the label instances per panel
instances_per_panel = (
    train_data
    .query("overlap >= @min_overlap")
    .query("obscured == 'no'")
    .query('~label.str.contains("unknown")')
    .query("count >= @min_instances")
    .groupby(['imagepath', 'panelcorners', 'label', 'pose', 'area_bins'])
    .size()
    .reset_index(name='counts')
    .pivot_table(
        index=['imagepath', 'panelcorners'],
        columns=['label', 'pose', 'area_bins'],
        values="counts",
        fill_value=0)
    .sample(frac = 1))

# Generate the train dataset
train = balanced_subsample(
    instances_per_panel.copy(),
    max_train_fg,
    max_train_bg,
    train_test_split)

# Drop the train data from instances_per_panel
instances_per_panel = instances_per_panel.drop(index = train)

train = pd.DataFrame(list(train), columns=['imagepath', 'panelcorners'])
# Convert train back into panel data
train = (
    pd.merge(data, train, indicator=True, how='left')
    .query('_merge=="both"')
    .drop('_merge', axis=1)
    .reset_index(drop = True))
train["dataset"] = 'train'
train.name = 'train'
n_train = len(train.drop_duplicates(['imagepath', 'panelcorners']))

# Generate the test dataset
test = balanced_subsample(instances_per_panel, max_test_fg, max_test_bg, 1)
test = pd.DataFrame(list(test), columns=['imagepath', 'panelcorners'])
# Convert test back into panel data
test = (
    pd.merge(data, test, indicator=True, how='left')
    .query('_merge=="both"')
    .drop('_merge', axis=1)
    .reset_index(drop = True))
test["dataset"] = 'test'
test.name = 'test'
n_test = len(test.drop_duplicates(['imagepath', 'panelcorners']))

# Print lengths of datasets
print("Train panels: ", n_train)
print("Test panels: ", n_test)
print("Validation panels: ", n_validation)

# Join full dataset
dataset = (
    pd.concat([train, test, validation, data], ignore_index=True)
    .reset_index(drop = True))
dataset.loc[dataset.dataset == "total", 'panelcorners'] = "NA"
dataset.loc[dataset.dataset == "total", 'overlap'] = 1
dataset = dataset.drop_duplicates(
    subset=['imagepath', 'panelcorners', 'points'])
dataset.to_csv(os.path.join(root, "balanced/annotations/dataset.csv"))

# Save data
save_dataset(train, test, validation)