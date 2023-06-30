import json
import csv

results = {"areas": [], "name": [], "age": [], "dataset": []}

for dataset_name in ["test", "train"]:
    instances_path = "datasets/bird-mask-only/dataset/annotations/instances_" + dataset_name + ".json"
    instances = json.load(open(instances_path))

    categories = instances["categories"]
    classes = {}
    for category in categories:
        classes[category["id"]] = category["name"]

    for annotation in instances["annotations"]:
        results["areas"].append(annotation["area"])
        label = json.loads(classes[annotation["category_id"]].replace("'", '"'))
        results["name"].append(label["name"])
        results["age"].append(label["age"])
        results["dataset"].append(dataset_name)

with open("Post-augmentation_Instance_Summary.csv", "w", newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(results.keys())
    writer.writerows(zip(*results.values()))