import os
import json

root = "C:/Users/uqjwil54/Documents/Projects/DBBD/Dryad"
os.chdir(root)

# iterate through files in dir
for root, dirs, files in os.walk(os.getcwd()):
    for file in files:
            if file.endswith(".json"):
                print("Remove original image name and convert polygons to boxes in", file)
                annotation_path = os.path.join(root, file)
                annotation = json.load(open(annotation_path))
                # Remove original image path from labels
                try:
                    del annotation["originalimagePath"]
                except: pass
                # Convert all polygons to boxes
                for i in range(len(annotation["shapes"])):
                     if annotation["shapes"][i]["shape_type"] == "polygon":
                        points = annotation["shapes"][i]["points"]
                        xmin = min(points, key=lambda x: x[0])[0]
                        xmax = max(points, key=lambda x: x[0])[0]
                        ymin = min(points, key=lambda x: x[1])[1]
                        ymax = max(points, key=lambda x: x[1])[1]
                        box = [[xmin, ymin], [xmax, ymax]]
                        annotation["shapes"][i]["points"] = box
                        annotation["shapes"][i]["shape_type"] = "rectangle"
                annotation_str = json.dumps(annotation, indent = 2).replace('"null"', 'null')
                with open(annotation_path, 'w') as annotation_file:
                    annotation_file.write(annotation_str)