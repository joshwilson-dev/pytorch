import json
import csv
import os
from shapely.geometry import Polygon

root = "datasets/bird-mask-only/input"
results = {"pixels": [], "gsd": [], "area": [], "px_5mmgsd": [], "name": [], "pose": [], "age": []}
for file in os.listdir(root):
    if file.endswith(".json"):
        annotation = json.load(open(os.path.join(root, file)))
        gsd = annotation["gsd"]
        for shape in annotation["shapes"]:
            if shape["shape_type"] == "polygon":
                instance_mask = tuple((point[0], point[1]) for point in shape["points"])
                pixels = Polygon(instance_mask).area
                area = pixels * gsd**2
                results["pixels"].append(pixels)
                results["gsd"].append(gsd)
                results["area"].append(area)
                results["px_5mmgsd"].append(area / 0.005**2)
                instance_label = json.loads(shape["label"].replace("'", '"'))
                results["name"].append(instance_label["name"])
                results["pose"].append(instance_label["pose"])
                results["age"].append(instance_label["age"])

with open("Pre-augmentation_Instance_Summary.csv", "w", newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(results.keys())
    writer.writerows(zip(*results.values()))