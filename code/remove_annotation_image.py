import os
import json

root = "C:/Users/uqjwil54/OneDrive - The University of Queensland/DBBD"
os.chdir(root)
# Iterate through files in dir and remove image data from annotation
for root, dirs, files in os.walk(os.getcwd()):
    for file in files:
        if file.endswith(".json"):
            annotation_path = os.path.join(root, file)
            annotation = json.load(open(annotation_path))
            if annotation["imageData"] != None:
                annotation["imageData"] = 'null'
                annotation_str = json.dumps(annotation, indent = 2).replace('"null"', 'null')
                with open(annotation_path, 'w') as annotation_file:
                    annotation_file.write(annotation_str)
                print("Removed image data from: ", file)