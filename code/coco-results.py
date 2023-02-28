import json
import csv
# load annotation
annotation_path = "datasets/seed-candice/dataset/annotations/instances_test.json"
coco_annotation = json.load(open(annotation_path))

# create index to class
index_to_class = {}
for category in coco_annotation["categories"]:
    id = category["id"]
    name = category["name"]
    index_to_class[id] = name

# create id to filename
id_to_filename = {}
for image in coco_annotation["images"]:
    id = image["id"]
    file_name = image["file_name"]
    id_to_filename[id] = file_name

# Create the results file
csv_path = "datasets/seed-candice/dataset/annotations/results.csv"
with open(csv_path, 'a+', newline='') as csvfile:
    fieldnames = ["image_name", "class"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for annotation in coco_annotation["annotations"]:
        image_name = id_to_filename[annotation["image_id"]]
        label = index_to_class[annotation["category_id"]]
        row = {"image_name": image_name, "class": label}
        writer.writerow(row)