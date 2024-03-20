import os
import shutil
import json

root = ""
os.chdir(root)

# iterate through files in dir
for root, dirs, files in os.walk(os.getcwd()):
    for file in files:
        # if "fully annotated" in root or "partially annotated" in root:
            if file.endswith(".json"):
                print("Renaming instances in", file)
                annotation_path = os.path.join(root, file)
                annotation = json.load(open(annotation_path))
                for i in range(len(annotation["shapes"])):
                    label = json.loads(annotation["shapes"][i]["label"].replace("'", '"'))

                    # Do something

                    annotation["shapes"][i]["label"] = json.dumps(label).replace('"', "'")
                annotation_str = json.dumps(annotation, indent = 2).replace('"null"', 'null')
                with open(annotation_path, 'w') as annotation_file:
                    annotation_file.write(annotation_str)