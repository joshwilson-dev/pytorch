import os
import json
import csv
import copy

root = "data/balanced-2024_04_24/train"

results = {
    "base": [], "file": [], "datetime": [], "shape_type": [],
    "common_name": [], "class": [], "order": [], "family": [], "genus": [],
    "species": [], "pose": [], "age": [], "latitude": [], "longitude": [],
    "gsd": [], "area": [], "obscured": []}

results = {"base": [], "file": [], "common_name": [], "age": []}

for base, dirs, files in os.walk(root):
    for file in files:
        # if "fully annotated" in base or "background" in base:
            if file.endswith(".json"):
                annotation = json.load(open(os.path.join(base, file)))
                # gsd = annotation["gsd"]
                # gsd_cat = str(round(gsd / 0.005) * 0.005)
                # latitude = annotation["latitude"]
                # longitude = annotation["longitude"]
                # location = str(round(latitude, 1)) + ", " + str(round(longitude, 1))
                # datetime = annotation["datetime"]
                # camera = annotation["camera"]

                # datetime = ""
                # latitude = ""
                # longitude = ""
                # gsd = ""

                if len(annotation["shapes"]) == 0:
                    results["base"].append(base)
                    results["file"].append(file)
                    # results["datetime"].append(datetime)
                    # results["shape_type"].append("background")
                    results["common_name"].append("background")
                    # results["class"].append("background")
                    # results["order"].append("background")
                    # results["family"].append("background")
                    # results["genus"].append("background")
                    # results["species"].append("background")
                    results["age"].append("background")
                    # results["pose"].append("background")
                    # results["latitude"].append(latitude)
                    # results["longitude"].append(longitude)
                    # results["gsd"].append(gsd)
                    # results["area"].append(0)
                    # results["obscured"].append("background")
                else:
                    for shape in annotation["shapes"]:
                        try:
                            points = tuple((point[0], point[1]) for point in shape["points"])
                            min_x = min(p[0] for p in points)
                            min_y = min(p[1] for p in points)
                            max_x = max(p[0] for p in points)
                            max_y = max(p[1] for p in points)
                            area = (max_x - min_x) * (max_y - min_y)
                            label = json.loads(shape["label"].replace("'", '"'))
                            results["base"].append(base)
                            results["file"].append(file)
                            # results["datetime"].append(datetime)
                            # results["shape_type"].append(shape["shape_type"])
                            results["common_name"].append(label["name"])
                            results["age"].append(label["age"])
                            # results["class"].append("aves")

                            # results["order"].append(label["order"])
                            # results["family"].append(label["family"])
                            # results["genus"].append(label["genus"])
                            # results["species"].append(label["species"])

                            # results["order"].append("")
                            # results["family"].append("")
                            # results["genus"].append("")
                            # results["species"].append("")
                            # results["pose"].append(label["pose"])
                            # results["obscured"].append(label["obscured"])

                            # results["latitude"].append(latitude)
                            # results["longitude"].append(longitude)
                            # results["gsd"].append(gsd)
                            # results["area"].append(area)

                        except Exception as error:
                            print("An exception occurred:", error)
                            print(file)
                            print(shape["label"])

with open("counts.csv", "w", newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(results.keys())
    writer.writerows(zip(*results.values()))