import json
import os
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
import numpy as np
import pandas as pd
import shutil

root = "data/balanced-2024_04_14"

img_output = os.path.join(root, "test_artificial")
directories = ["test_artificial"]
for directory in directories:
    path = os.path.join(root, directory)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def crop_mask(imagepath, points):

    image = Image.open(imagepath)

    # Find the bounding box
    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_x = max(p[0] for p in points)
    max_y = max(p[1] for p in points)

    # Calculate crop size and center coordinates
    width = max_x - min_x
    height = max_y - min_y

    center_x = min_x + width / 2
    center_y = min_y + height / 2

    # Crop the image
    crop_left = int(center_x - width / 2)
    crop_upper = int(center_y - height / 2)
    crop_right = int(center_x + width / 2)
    crop_lower = int(center_y + height / 2)
    bird = image.crop((crop_left, crop_upper, crop_right, crop_lower)).convert('RGBA')

    # Adjust polygon to crop origin
    origin = [(center_x - width / 2, center_y - height / 2)] * len(points)
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

def save_dataset(foregrounds, backgrounds):
    index_to_class = json.load(open("resources/index_to_class.json"))
    class_to_index = [value for value in index_to_class.values()]

    # Creating coco categories
    coco_categories = []
    for key, value in index_to_class.items():
        coco_categories.append({"id": int(key), "name": json.dumps(value).replace('"', "'")})

    # define intermediate areas
    areas = range(25, 525, 25)
    n_areas = len(areas)
    gsd = 0.005
    crop_size = 800
    coco = {"images": [], "annotations": [], "categories": coco_categories}
    coco_path = root + "/balanced/annotations/test_artificial.json"
    coco_image_id = 1
    coco_instance_id = 1
    n_bg = len(backgrounds)
    n_fg = len(foregrounds)
    # Step through backgrounds
    print("Total images:", n_bg * n_fg * n_areas)
    for i, bg_data in backgrounds.iterrows():
        print("BG {} of {}".format(i+1, n_bg))
        # Get the gsd
        bg_gsd = bg_data["gsd"]
        # Load the image
        bg = Image.open(bg_data['imagepath'])
        # Resize the to the final gsd
        bg_scale = bg_gsd/gsd
        bg_size = tuple(int(dim * bg_scale) for dim in bg.size)
        bg = bg.resize((bg_size))
        # Crop the image to 800 x 800
        left = (bg.width - crop_size) / 2
        top = (bg.height - crop_size) / 2
        right = (bg.width + crop_size) / 2
        bottom = (bg.height + crop_size) / 2
        # Crop the resized image
        bg = bg.crop((left, top, right, bottom))
        # for each area
        for i, fg_data in foregrounds.iterrows():
            # print("\tFG {} of {}".format(i+1, n_fg))
            bg2 = bg.copy()
            fg = fg_data['instance']
            fg_gsd = fg_data["gsd"]
            fg_area = fg_data["area"]
            fg_scale = fg_gsd/gsd
            for i, area in enumerate(areas):
                # print("\t\tA {} of {}".format(i+1, n_areas))
                # Rescale fg to intermediate resolution
                fg_scale1 = (fg_area/area)**0.5
                fg_size1 = tuple(int(dim / fg_scale1) for dim in fg.size)
                fg1 = fg.resize((fg_size1))
                fg1_gsd = fg_gsd * fg_scale1
                # Rescale foreground to final resolution
                fg_scale2 = fg1_gsd/gsd
                fg_size2 = tuple(int(dim * fg_scale2) for dim in fg1.size)
                fg2 = fg1.resize((fg_size2))
                # Paste fg at centre of bg
                paste_position = ((bg.width - fg2.width) // 2, (bg.height - fg2.height) // 2)
                bg2.paste(fg2, paste_position, fg2)
                # Save the image
                img_name = str(coco_image_id) + ".jpg"
                img_path = os.path.join(img_output, img_name)
                bg2.save(img_path)
                # Rescale points
                points = fg_data["points"]
                xmin = min(points, key=lambda x: x[0])[0]
                ymin = min(points, key=lambda x: x[1])[1]
                points = [[int(((point[0] - xmin) * fg_scale) + paste_position[0]), int(((point[1] - ymin) * fg_scale) + paste_position[1])] for point in points]
                xmin = min(points, key=lambda x: x[0])[0]
                xmax = max(points, key=lambda x: x[0])[0]
                ymin = min(points, key=lambda x: x[1])[1]
                ymax = max(points, key=lambda x: x[1])[1]
                coco_instance_box = [xmin, ymin, xmax - xmin, ymax - ymin]
                # Create labelme annotation
                labelme_shape = {
                    "label": fg_data["label"],
                    "shape_type": "polygon",
                    "points": points,
                    "group_id": 'null',
                    "flags": {}}
                labelme_path = os.path.splitext(img_path)[0]+'.json'
                labelme = {
                    "version": "5.0.1",
                    "flags": {},
                    "shapes": [labelme_shape],
                    "imagePath": img_name,
                    "imageData": 'null',
                    "imageHeight": crop_size,
                    "imageWidth": crop_size}
                labelme = json.dumps(labelme, indent = 2).replace('"null"', 'null')
                with open(labelme_path, 'w') as labelme_file:
                    labelme_file.write(labelme)
                # Create coco annotation
                coco_annotation = {
                    "iscrowd": 0,
                    "image_id": coco_image_id,
                    "category_id": class_to_index.index(json.loads(fg_data["label"].replace("'", '"'))) + 1,
                    "segmentation": [[item for sublist in points for item in sublist]],
                    "bbox": coco_instance_box,
                    "id": coco_instance_id,
                    "area": area}
                coco["annotations"].append(coco_annotation)
                coco_image_info = {
                    "height": crop_size,
                    "width": crop_size,
                    "id": coco_image_id,
                    "file_name": img_name}
                coco["images"].append(coco_image_info)
                coco_image_id += 1
                coco_instance_id += 1
    # Save coco
    coco = json.dumps(coco, indent = 2)
    with open(coco_path, 'w') as coco_file:
        coco_file.write(coco)

    return

dataset_keys = ["imagepath", "label", "points", "gsd", "area", "location"]
data = {k:[] for k in dataset_keys}

for entry in os.scandir(os.path.join(root,"test")):
    if entry.path.endswith(".json"):
        print("Recording data from: ", entry.path)
        # Load the annotation
        annotation = json.load(open(entry.path))
        # Load the image
        imagepath = os.path.join(root, "test", annotation["imagePath"])
        # Get image metrics
        gsd = annotation["gsd"]
        latitude = annotation["latitude"]
        longitude = annotation["longitude"]
        location = str(round(latitude, 0)) + ", " + str(round(longitude, 0))
        # Extract polygons
        if len(annotation["shapes"]) != 0:
            for shape in annotation["shapes"]:
                shapetype = shape["shape_type"]
                if shapetype == "polygon":
                    points = tuple((point[0], point[1]) for point in shape["points"])
                    polygon = Polygon(points)
                    area = polygon.area
                    label = json.loads(shape["label"].replace("'", '"'))
                    if label["obscured"] == 'yes':
                        continue
                    del label["obscured"]
                    pose = label["pose"]
                    if pose != "resting":
                        continue
                    del label["pose"]
                    name = label["name"]
                    if name == 'bird':
                        continue
                    label = json.dumps(label).replace('"', "'")
                    
                    # Save instance
                    data["imagepath"].append(imagepath)
                    data["gsd"].append(gsd)
                    data["label"].append(label)
                    data["points"].append(points)
                    data["area"].append(area)
                    data["location"].append(location)
        else:
            data["imagepath"].append(imagepath)
            data["gsd"].append(gsd)
            data["label"].append("background")
            data["points"].append("background")
            data["area"].append(0)
            data["location"].append(location)

# Convert dictionary to dataframe
data = pd.DataFrame(data=data)

# Select top 10 area foregrounds
print("Creating foregrounds")
foregrounds = (
    data
    .query("label != 'background'")
    .query("area > 500")
    .groupby('label')
    .filter(lambda x: len(x) > 10)
    .groupby('label')
    .apply(lambda x: x.nlargest(10, 'area'))
    .reset_index(drop=True)
)

# Crop out instances and replace imagepath
instances = []
for _, row in foregrounds.iterrows():
    instances.append(crop_mask(row['imagepath'], row['points']))
instances = pd.DataFrame(data=instances)
foregrounds["instance"] = instances[0]

print("Creating backgrounds")
# Select 1 background from each group
backgrounds = (
    data
    .query("label == 'background'")
    .groupby('location', group_keys=False)
    .apply(lambda x: x.nsmallest(1, 'gsd'))
    .reset_index(drop=True)
)

print("Saving dataset")
save_dataset(foregrounds, backgrounds)