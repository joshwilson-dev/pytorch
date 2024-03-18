import os
import json
import csv
import copy

# base = "C:/Users/uqjwil54/OneDrive - The University of Queensland/DBBD"
# base = "datasets/bird_2024_02_13/raw"
base = "datasets/bird_2024_02_13/balanced/train"
os.chdir(base)

results = {"root": [], "dir": [], "file": [], "datetime": [], "shape_type": [], "common_name": [], "class": [], "order": [], "family": [], "genus": [], "species": [], "pose": [], "age": [], "latitude": [], "longitude": [], "gsd": [], "area": [], "obscured": []}
for root, dirs, files in os.walk(os.getcwd()):
    for file in files:
        # if "fully annotated" in root or "background" in root:
            if file.endswith(".json"):
                # print(file)
                annotation = json.load(open(os.path.join(root, file)))
                # gsd = annotation["gsd"]
                # gsd_cat = str(round(gsd / 0.005) * 0.005)
                # latitude = annotation["latitude"]
                # longitude = annotation["longitude"]
                # location = str(round(latitude, 1)) + ", " + str(round(longitude, 1))
                # datetime = annotation["datetime"]

                datetime = ""
                latitude = ""
                longitude = ""
                gsd = ""
                if len(annotation["shapes"]) == 0:
                    results["root"].append(root.split("\\")[5])
                    results["dir"].append(root.split("\\")[6])
                    results["file"].append(file)
                    results["datetime"].append(datetime)
                    results["shape_type"].append("background")
                    results["common_name"].append("background")
                    results["class"].append("background")
                    results["order"].append("background")
                    results["family"].append("background")
                    results["genus"].append("background")
                    results["species"].append("background")
                    results["age"].append("background")
                    results["pose"].append("background")
                    results["latitude"].append(latitude)
                    results["longitude"].append(longitude)
                    results["gsd"].append(gsd)
                    results["area"].append(0)
                    results["obscured"].append("background")
                else:
                    for shape in annotation["shapes"]:
                        # print(shape["label"])
                        try:
                            points = tuple((point[0], point[1]) for point in shape["points"])
                            min_x = min(p[0] for p in points)
                            min_y = min(p[1] for p in points)
                            max_x = max(p[0] for p in points)
                            max_y = max(p[1] for p in points)
                            area = (max_x - min_x) * (max_y - min_y)
                            label = json.loads(shape["label"].replace("'", '"'))
                            results["root"].append(root.split("\\")[5])
                            results["dir"].append(root.split("\\")[6])
                            results["file"].append(file)
                            results["datetime"].append(datetime)
                            results["shape_type"].append(shape["shape_type"])
                            results["common_name"].append(label["name"])
                            results["age"].append(label["age"])
                            results["class"].append("aves")

                            results["order"].append(label["order"])
                            results["family"].append(label["family"])
                            results["genus"].append(label["genus"])
                            results["species"].append(label["species"])
                            results["pose"].append("")
                            results["obscured"].append("")

                            # results["order"].append("")
                            # results["family"].append("")
                            # results["genus"].append("")
                            # results["species"].append("")
                            # results["pose"].append(label["pose"])
                            # results["obscured"].append(label["obscured"])

                            results["latitude"].append(latitude)
                            results["longitude"].append(longitude)
                            results["gsd"].append(gsd)
                            results["area"].append(area)

                        except Exception as error:
                            print("An exception occurred:", error)
                            print(file)
                            print(shape["label"])

# with open("tools/counts.csv", "w", newline='') as outfile:
with open("counts.csv", "w", newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(results.keys())
    writer.writerows(zip(*results.values()))