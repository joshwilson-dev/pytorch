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

#################
#### Content ####
#################

def balance_instances(df, fg_max, fg_bg_ratio, ratio, buffer):
    # Create the model
    model = cp_model.CpModel()
    num_rows = df.shape[0]
    
    # Create array containing row selection flags. True if row k is selected, False otherwise
    row_selection = [model.NewBoolVar(f'row_{i}') for i in range(num_rows)]

    # Get a list of species and poses
    labels = [k for k in list(df.columns.levels[0].unique()) if 'name' in k] + ["background"]
    
    # Sort df so that columns with smallest sum appear first
    df.loc['Total'] = df.sum()
    df = df.sort_values(by = 'Total', axis = 1)
    df = df.drop(labels = 'Total')

    for label in labels:
        label_df = df[label]
        n_label_total = label_df.values.sum()
        if label == "background":
            # Sample the backgrounds to get ratio of foreground to background images, assuming 3 instances per image
            n_label_upper = min((len(labels) - 1) * fg_max * ratio * fg_bg_ratio * 0.33, n_label_total)
            n_label_lower = n_label_upper
        else:
            # Sample the minimum of fg_max, or the number of instances, per label
            n_label_upper = round(min(n_label_total, fg_max) * ratio)
            if n_label_total <= fg_max * buffer: n_label_lower = n_label_upper
            else: n_label_lower = round(fg_max * ratio * buffer)
        print("Label Limit: {}: {} of {}".format(label, n_label_upper, n_label_total))
        print("Lower Limit: {}: {} of {}".format(label, n_label_lower, n_label_total))

        # Sample poses as equally as possible within each label
        unique_poses = list(label_df.columns.unique(level = 'pose'))
        pose_remainder = n_label_lower
        for pose in unique_poses:
            pose_df = label_df[pose]
            n_pose_total = pose_df.values.sum()
            n_pose_sample = 0
            index = unique_poses.index(pose)
            req_avg = pose_remainder/(len(unique_poses) - index)
            if n_pose_total * ratio <= req_avg:
                n_pose_sample = n_pose_total * ratio
            else:
                n_pose_sample = req_avg
            pose_remainder -= n_pose_sample
            print("\t{}: {} of {}".format(pose, math.ceil(n_pose_sample), n_pose_total))
            
            # Sample location as equally as possible within each pose 
            unique_locations = list(pose_df.columns.unique(level = 'location'))
            loc_remainder = n_pose_sample
            for location in unique_locations:
                loc_df = pose_df[location]
                n_loc_total = loc_df.values.sum()
                n_loc_sample = 0
                index = unique_locations.index(location)
                req_avg = loc_remainder/(len(unique_locations) - index)
                if n_loc_total * ratio <= req_avg:
                    n_loc_sample = n_loc_total * ratio
                else:
                    n_loc_sample = req_avg
                loc_remainder -= n_loc_sample
                print("\t\t{}: {} of {}".format(location, math.ceil(n_loc_sample), n_loc_total))

                # Sample areas as equally as possible within each location
                unique_areas = list(loc_df.columns.unique(level = 'area_bins'))
                area_remainder = n_loc_sample
                for area in unique_areas:
                    area_df = loc_df[area]
                    n_area_total = area_df.values.sum()
                    n_area_sample = 0
                    index = unique_areas.index(area)
                    req_avg = area_remainder/(len(unique_areas) - index)
                    if n_area_total * ratio <= req_avg:
                        n_area_sample = math.ceil(n_area_total * ratio)
                    else:
                        n_area_sample = math.ceil(req_avg)
                    area_remainder -= n_area_sample
                    print("\t\t\t{}: {} of {}".format(area, n_area_sample, n_area_total))
                    model.Add(df[label][pose][location][area].dot(row_selection) <= n_area_sample)

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
        total_images = dataset.groupby(['imagepath', 'patchpoints']).ngroups

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
            patches = patches.groupby("patchpoints")
            for patchpoints, annotations in patches:
                labelme_shapes = []

                # Print progress
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
                
                # Iterate over instances
                for index, annotation in annotations.reset_index(drop = True).iterrows():
                    label = annotation["label"]

                    if label != "background":
                        # Get annotation data
                        points = annotation["points"]
                        overlap = annotation["overlap"]
                        shapetype = annotation["shapetype"]

                        # Rescale the mask
                        points = [[int((point[0] * scale - patchpoints[0][0])), int((point[1] * scale - patchpoints[0][1]))] for point in points]
                        
                        # Generate the box
                        xmin = min(points, key=lambda x: x[0])[0]
                        xmax = max(points, key=lambda x: x[0])[0]
                        ymin = min(points, key=lambda x: x[1])[1]
                        ymax = max(points, key=lambda x: x[1])[1]
                        coco_instance_box = [xmin, ymin, xmax - xmin, ymax - ymin]
                        labelme_instance_box = [[xmin, ymin], [xmax, ymax]]

                        # Blackout instances that aren't included
                        if label not in included_classes or overlap < min_overlap:
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

# Set root and setup directories
root = "data/bird_2024_02_19/"
# root = "datasets/trial/"
directories = ["train", "test", "annotations"]
for directory in directories:
    path = os.path.join(root, "balanced", directory)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

# Define parameters
min_instances = 0
max_train_instances = 200
max_test_instances = 100
min_overlap = 0.75
target_gsd = 0.005
fg_bg_ratio = 1.0
target_patchsize = 800

# Create dataset to save annotation data to
dataset_keys = [
    "imagepath", "patchpoints", "label", "points", "shapetype", "overlap",
    "gsd", "latitude", "longitude", "pose", "area"]
data = {k:[] for k in dataset_keys}


# Extract information from each image
for entry in os.scandir(os.path.join(root,"raw")):
    if entry.path.endswith(".json"):
        print("Recording Patch Data From: ", entry.path)

        # Load the annotation
        annotation = json.load(open(entry.path))

        # Load the image
        imagepath = os.path.join(root, "raw", annotation["imagePath"])
        image = Image.open(imagepath)
        width, height = image.size

        # Get the image metrics
        gsd = annotation["gsd"]
        latitude = annotation["latitude"]
        longitude = annotation["longitude"]

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

                    # Check if polygon is in patch, skip if it isn't
                    overlap_area = patch_polygon.intersection(polygon).area
                    if overlap_area == 0: continue

                    # calculate the overlap between the full mask and the patch mask
                    overlap = overlap_area/polygon.area

                    # Get the label and pose
                    label = json.loads(shape["label"].replace("'", '"'))

                    if label["obscured"] == 'yes':
                        overlap = 0
                    del label["obscured"]

                    pose = label["pose"]
                    del label["pose"]

                    label = json.dumps(label).replace('"', "'")

                    # Save instance information
                    data["imagepath"].append(imagepath)
                    data["gsd"].append(gsd)
                    data["latitude"].append(latitude)
                    data["longitude"].append(longitude)
                    data["patchpoints"].append(patchpoints)
                    data["pose"].append(pose)
                    data["label"].append(label)
                    data["points"].append(points)
                    data["shapetype"].append(shapetype)
                    data["area"].append(polygon.area)
                    data["overlap"].append(overlap)
                    background = False
  
                # Save backgrounds
                if background == True:
                    data["imagepath"].append(imagepath)
                    data["gsd"].append(gsd)
                    data["latitude"].append(latitude)
                    data["longitude"].append(longitude)
                    data["patchpoints"].append(patchpoints)
                    data["pose"].append("background")
                    data["label"].append("background")
                    data["points"].append("background")
                    data["shapetype"].append("background")
                    data["area"].append(patchsize*patchsize)
                    data["overlap"].append(1)

# Convert dictionary to dataframe
data = pd.DataFrame(data=data)
data["dataset"] = "original"

# Calculate location and area_cat
bins = [0, 1000, 2000, 4000, 1e10]
data["area_bins"] = pd.cut(data['area'], bins).astype(str)
data["location"] = data["latitude"].round(0).astype(str) + ", " + data["longitude"].round(0).astype(str)

# Determine the total abundance of each label
label_count = (
    data
    .query("overlap >= @min_overlap")
    .query("label != 'background'")
    .query("~label.str.contains('bird')", engine="python")
    .drop_duplicates(subset=['imagepath', 'points'])
    .label
    .value_counts())

print("\tLabel Class Count:\n{}\n".format(label_count.to_string()))

# Only include labels with enough polygons
included_classes = label_count[label_count >= min_instances].index.tolist() + ["background"]
print("Included Classes: ", included_classes)
data = data.assign(included = lambda x: x.label.isin(included_classes))

# Count the label instances per patch
instances_per_patch = (
    data
    .query("overlap >= @min_overlap")
    .query("included")
    .groupby(['imagepath', 'patchpoints', 'label', 'pose', 'location', 'area_bins'])
    .size()
    .reset_index(name='counts')
    .pivot_table(
        index=['imagepath', 'patchpoints'],
        columns=['label', 'pose', 'location', 'area_bins'],
        values="counts",
        fill_value=0))

# Generate the train dataset
train = balance_instances(instances_per_patch.copy(), max_train_instances, fg_bg_ratio, 0.75, 1)

# Drop the train data from instances_per_patch
instances_per_patch = instances_per_patch.drop(index = train)

train = pd.DataFrame(list(train), columns=['imagepath', 'patchpoints'])
# Convert train back into patch data
train = (
    pd.merge(data, train, indicator=True, how='left')
    .query('_merge=="both"')
    .drop('_merge', axis=1)
    .reset_index(drop = True))
train["dataset"] = 'train'
train.name = 'train'
n_train = len(train.drop_duplicates(['imagepath', 'patchpoints']))

# Generate the test dataset
test = balance_instances(instances_per_patch, max_test_instances, fg_bg_ratio, 1, 1)
test = pd.DataFrame(list(test), columns=['imagepath', 'patchpoints'])
# Convert test back into patch data
test = (
    pd.merge(data, test, indicator=True, how='left')
    .query('_merge=="both"')
    .drop('_merge', axis=1)
    .reset_index(drop = True))
test["dataset"] = 'test'
test.name = 'test'
n_test = len(test.drop_duplicates(['imagepath', 'patchpoints']))

# Print lengths of datasets
print("Train patches: ", n_train)
print("Test patches: ", n_test)

# Join full dataset
dataset = (
    pd.concat([train, test], ignore_index=True)
    .reset_index(drop = True))
dataset.to_csv(os.path.join(root, "balanced/annotations/dataset.csv"))

# Save data
save_dataset(train, test)