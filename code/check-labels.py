import os
import json
import csv

base = "C:/Users/uqjwil54/Nextcloud/AIRBIRDDAT-Q5041"
os.chdir(base)

results = {"root": [], "file": [], "shape_type": [], "name": [], "pose": [], "age": [], "sex": [], "obscured": []}
for root, dirs, files in os.walk(os.getcwd()):
    for file in files:
        if "fully annotated" in root or "partially annotated" in root:
            if file.endswith(".json"):
                annotation = json.load(open(os.path.join(root, file)))
                # print(file)
                for shape in annotation["shapes"]:
                    # print(shape["label"])
                    try:
                        instance_label = json.loads(shape["label"].replace("'", '"'))
                        results["root"].append(root.split("\\")[5])
                        results["file"].append(file)
                        results["shape_type"].append(shape["shape_type"])
                        results["name"].append(instance_label["name"])
                        results["pose"].append(instance_label["pose"])
                        results["age"].append(instance_label["age"])
                        results["sex"].append(instance_label["sex"])
                        results["obscured"].append(instance_label["obscured"])
                    except:
                        print(file)
                        print(shape["label"])

with open("tools/counts.csv", "w", newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(results.keys())
    writer.writerows(zip(*results.values()))