# Author:       Joshua P. Wilson
# Description   Code to prepare images for training
# Date:         14.04.2024

# Install packages
# pip install pandas==2.0.3
# pip install shapely==2.0.3
# pip install piexif==1.1.3
# pip install pillow==10.2.0
# pip install ortools==9.9.3963

# Import packages
import os
import shutil
import json
import math
import hashlib

import pandas as pd
from PIL import Image
from ortools.sat.python import cp_model
from shapely.geometry import Polygon
import piexif

# Set root and setup directories
root = "data/"
directories = ["train", "test", "annotations"]
for directory in directories:
    path = os.path.join(root, "balanced", directory)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

# Define parameters
train_test_split = 0.7
max_instances = 10000000
max_train_instances = round(max_instances * train_test_split)
max_test_instances = round(max_instances * (1 - train_test_split))
min_overlap = 0.75
target_gsd = 0.005
target_patchsize = 800
min_instances = 0

# Create dataset to save annotation data to
dataset_keys = [
    "imagepath", "uav", "height", "camera", "sensorwidth", "sensorheight",
    "focallength", "imagewidth", "imageheight", "gsd", "latitude", "longitude",
    "datetime", "patchcorners", "label", "points", "shapetype", "overlap",
    "commonname", "class", "order", "family", "genus", "species", "pose", "age",
    "obscured", "area"]
data = {k:[] for k in dataset_keys}


def balanced_subsample(df, train_test_split, max_sample):
    # Create the model
    model = cp_model.CpModel()
    num_rows = df.shape[0]
    
    # Create array containing row selection flags.
    row_selection = [model.NewBoolVar(f'row_{i}') for i in range(num_rows)]

    # Get a list of labels
    labels = [k for k in list(df.columns.unique()) if 'name' in k]

    for label in labels:
        count = df[label].sum()
        min_sample = df[label][df[label] > 0].min()
        if pd.isna(min_sample): min_sample = 0
        max_split = math.ceil(train_test_split * count)
        model.Add(df[label].dot(row_selection) <= max(min(max_sample, max_split), min_sample))
        model.Add(df[label].dot(row_selection) >= min_sample)
        print(label)
        # print(count)
        # print(min_sample)
        # print(max_split)
        # print(max_sample)
        # print("<=", max(min(max_sample, max_split), min_sample))
        # print(">=", min_sample)

    # Maximize the number of patches selected
    model.Maximize(sum(row_selection))

    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    # Check if an optimal solution is found
    if status == cp_model.OPTIMAL:
        selected_rows = [i for i in range(num_rows) if solver.Value(row_selection[i])]
        return df.iloc[selected_rows].index
    else:
        raise ValueError("Optimal solution not found.")

def blackout_instance(image, box):
    box_width = box[2]
    box_height = box[3]
    topleft = (round(box[0]), round(box[1]))
    black_box = Image.new("RGB", (round(box_width), round(box_height)))
    image.paste(black_box, topleft)
    return image

def save_dataset(train, test):
    
    # Specifying index to class
    index_to_class = json.load(open("resources/index_to_class.json"))
    index_to_class_path = os.path.join(root, "balanced/annotations/index_to_class.json")

    # Creating class to index list
    class_to_index = [value for value in index_to_class.values()]

    # Creating coco categories
    coco_categories = []
    for key, value in index_to_class.items():
        coco_categories.append({"id": int(key), "name": json.dumps(value).replace('"', "'")})

    for dataset in [train, test]:
        dir = dataset.name

        # Determine total images to print
        total_images = dataset.groupby(['imagepath', 'patchcorners']).ngroups

        # Coco annotation setup
        coco = {"images": [], "annotations": [], "categories": coco_categories}
        coco_path = root + "/balanced/annotations/{}.json".format(dir)
        coco_image_id = 1
        coco_instance_id = 1

        # Step through images, crop, save, and create annotation
        dataset = dataset.groupby('imagepath')
        for imagepath, patches in dataset:

            # Load image
            image = Image.open(imagepath)

            # Resize image to target gsd
            gsd = patches["gsd"].iloc[0]
            scale = gsd/target_gsd
            size = tuple(int(dim * scale) for dim in image.size)
            image = image.resize((size))

            # Iterate over patches
            patches = patches.groupby("patchcorners")
            for patchcorners, annotations in patches:
                labelme_shapes = []

                # Print progress
                if coco_image_id % int(total_images / 5) == 0:
                    print("Saving {} image {} of {}".format(dir, coco_image_id, total_images))

                # Crop the image to the patch
                patchcorners = [(x * scale, y * scale) for x, y in patchcorners]
                croppoints = (
                    patchcorners[0][0],
                    patchcorners[0][1],
                    patchcorners[1][0],
                    patchcorners[2][1])
                patch = image.crop(croppoints)
                
                # Iterate over instances
                for index, annotation in annotations.reset_index(drop = True).iterrows():
                    label = annotation["label"]

                    if "background" not in label:
                        # Get annotation data
                        points = annotation["points"]
                        overlap = annotation["overlap"]
                        obscured = annotation["obscured"]
                        count = annotation["count"]
                        shapetype = annotation["shapetype"]

                        # Rescale the mask
                        points = [[int((point[0] * scale - patchcorners[0][0])), int((point[1] * scale - patchcorners[0][1]))] for point in points]
                        
                        # Generate the box
                        xmin = min(points, key=lambda x: x[0])[0]
                        xmax = max(points, key=lambda x: x[0])[0]
                        ymin = min(points, key=lambda x: x[1])[1]
                        ymax = max(points, key=lambda x: x[1])[1]
                        coco_instance_box = [xmin, ymin, xmax - xmin, ymax - ymin]
                        labelme_instance_box = [[xmin, ymin], [xmax, ymax]]

                        # Blackout instances that have low instances or overlap
                        if count < min_instances or overlap < min_overlap or obscured == 'yes' or 'unknown' in label:
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
                        coco_annotation = {
                            "iscrowd": 0,
                            "image_id": coco_image_id,
                            "category_id": class_to_index.index(json.loads(label.replace("'", '"'))) + 1,
                            "segmentation": [[item for sublist in points for item in sublist]],
                            "bbox": coco_instance_box,
                            "id": coco_instance_id,
                            "area": annotation["area"]}
                        coco["annotations"].append(coco_annotation)
                        coco_instance_id += 1

                # Save patch
                patch_name = hashlib.md5(patch.tobytes()).hexdigest() + ".jpg"
                patch_path = os.path.join(root, "balanced", dir, patch_name)
                patchsize = patch.size
                exif = piexif.load(image.info['exif'])
                exif = piexif.dump(exif)
                patch.save(patch_path, exif = exif)

                # Store coco info
                coco_image_info = {
                    "height": patchsize[1],
                    "width": patchsize[0],
                    "id": coco_image_id,
                    "file_name": patch_name}
                coco["images"].append(coco_image_info)
                coco_image_id += 1
                
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
for entry in os.scandir(os.path.join(root,"consolidated")):
    if entry.path.endswith(".json"):
        print("Recording Patch Data From: ", entry.path)

        # Load the annotation
        annotation = json.load(open(entry.path))

        # Load the image
        imagepath = os.path.join(root, "consolidated", annotation["imagePath"])
        image = Image.open(imagepath)
        imagewidth, imageheight = image.size

        # Get the image metrics
        gsd = annotation["gsd"]
        latitude = annotation["latitude"]
        longitude = annotation["longitude"]
        uav = annotation["uav"]
        camera = annotation["camera"]
        height = annotation["drone_height"]
        sensorwidth = annotation["sensorwidth"]
        sensorheight = annotation["sensorheight"]
        focallength = annotation["focallength"]
        datetime = annotation["datetime"]

        # Create location feature
        location = str(round(latitude, 1)) + ", " + str(round(longitude, 1))
        
        # Determine the scale factor
        scale = target_gsd/gsd
        patchsize = target_patchsize * scale

        # Determine how many crops within accross the image
        n_crops_width = math.ceil(imagewidth / patchsize)
        n_crops_height = math.ceil(imageheight / patchsize)

        # Calculate the padded total size of the image
        padded_width = n_crops_width * patchsize
        padded_height = n_crops_height * patchsize

        # Calculate the width of the padding
        pad_width = (padded_width - imagewidth) / 2
        pad_height = (padded_height - imageheight) / 2

        # Iterate over the patches and save information
        for height_index in range(n_crops_height):
            for width_index in range(n_crops_width):

                # Determine the corners of the patch
                left = width_index * patchsize - pad_width
                right = left + patchsize
                top = height_index * patchsize - pad_height
                bottom = top + patchsize
                patchcorners = (
                    (left, top),
                    (right, top),
                    (right, bottom),
                    (left, bottom))
                patch_polygon = Polygon(patchcorners)

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

                    # Check if polygon is in patch, skip if it isn't
                    overlap_area = patch_polygon.intersection(polygon).area
                    if overlap_area == 0: continue

                    # calculate the overlap between the full mask and the patch mask
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
                    data["uav"].append(uav)
                    data["height"].append(height)
                    data["camera"].append(camera)
                    data["sensorwidth"].append(sensorwidth)
                    data["sensorheight"].append(sensorheight)
                    data["focallength"].append(focallength)
                    data["imagewidth"].append(imagewidth)
                    data["imageheight"].append(imageheight)
                    data["gsd"].append(gsd)
                    data["latitude"].append(latitude)
                    data["longitude"].append(longitude)
                    data["datetime"].append(datetime)
                    data["patchcorners"].append(patchcorners)
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
                    data["uav"].append(uav)
                    data["height"].append(height)
                    data["camera"].append(camera)
                    data["sensorwidth"].append(sensorwidth)
                    data["sensorheight"].append(sensorheight)
                    data["focallength"].append(focallength)
                    data["imagewidth"].append(imagewidth)
                    data["imageheight"].append(imageheight)
                    data["gsd"].append(gsd)
                    data["latitude"].append(latitude)
                    data["longitude"].append(longitude)
                    data["datetime"].append(datetime)
                    data["patchcorners"].append(patchcorners)
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
                    data["area"].append(patchsize*patchsize)
                    data["overlap"].append(1)

# Convert dictionary to dataframe
data = pd.DataFrame(data=data)
data["dataset"] = "total"

# Determine the total abundance of each class
class_count = (
    data
    # .query("label != 'background'")
    .query("overlap >= @min_overlap")
    .query("obscured == 'no'")
    .drop_duplicates(subset=['imagepath', 'points'])
    .groupby('label')
    .size())
print("\tClass Count:\n{}\n".format(class_count.to_string()))

# Map the counts back to the original DataFrame based on 'label'
data['count'] = data['label'].map(class_count)

# Count the label instances per patch
instances_per_patch = (
    data
    # .query("label != 'background'")
    .query("overlap >= @min_overlap")
    .query("obscured == 'no'")
    .query("'unknown' not in label")
    .query("count >= @min_instances")
    .groupby(['imagepath', 'patchcorners', 'label'])
    .size()
    .reset_index(name='counts')
    .pivot_table(
        index=['imagepath', 'patchcorners'],
        columns=['label'],
        values="counts",
        fill_value=0))

# Generate the train dataset
train = balanced_subsample(instances_per_patch.copy(), train_test_split, max_train_instances)

# Drop the train data from instances_per_patch
instances_per_patch = instances_per_patch.drop(index = train)

# Convert train back into patch data
train = pd.DataFrame(list(train), columns=['imagepath', 'patchcorners'])
train = (
    pd.merge(data, train, indicator=True, how='left')
    .query('_merge=="both"')
    .drop('_merge', axis=1)
    .reset_index(drop = True))
train["dataset"] = 'train'
train.name = 'train'
n_train = len(train.drop_duplicates(['imagepath', 'patchcorners']))

# Convert test back into patch data
# Generate the train dataset
test = balanced_subsample(instances_per_patch.copy(), 1, max_test_instances)
test = pd.DataFrame(list(test), columns=['imagepath', 'patchcorners'])
test = (
    pd.merge(data, test, indicator=True, how='left')
    .query('_merge=="both"')
    .drop('_merge', axis=1)
    .reset_index(drop = True))
test["dataset"] = 'test'
test.name = 'test'
n_test = len(test.drop_duplicates(['imagepath', 'patchcorners']))

# Print lengths of datasets
print("Train patches: ", n_train)
print("Test patches: ", n_test)

# Join full dataset
dataset = (
    pd.concat([train, test, data], ignore_index=True)
    .reset_index(drop = True))
dataset.loc[dataset.dataset == "total", 'patchcorners'] = "NA"
dataset.loc[dataset.dataset == "total", overlap] = 1
dataset = dataset.drop_duplicates(subset=['imagepath', 'patchcorners', 'points'])
dataset.to_csv(os.path.join(root, "balanced/annotations/dataset.csv"))

# Save data
save_dataset(train, test)