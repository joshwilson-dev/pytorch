import json
import csv

annotations = json.load(open("datasets/bird_2024_01_20/balanced/annotations/test.json"))

instances = []
for annotation in annotations["annotations"]:
    category_id = annotation["category_id"]
    label = json.loads(annotations["categories"][category_id - 1]["name"].replace("'", '"'))
    label = label["name"] + " - " + label["age"]
    image_id = annotation["image_id"]
    image_name = annotations["images"][image_id - 1]["file_name"]
    area = annotation["area"]
    box = annotation["bbox"]
    xcentre = box[0] + box[2]/2
    ycentre = box[1] + box[3]/2
    instance = {"filename": image_name, "xcentre": xcentre, "ycentre": ycentre, "class": label, "area": area}
    instances.append(instance)

with open('test.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = ["filename", "xcentre", "ycentre", "class", "area"])
    writer.writeheader()
    writer.writerows(instances)