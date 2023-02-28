# import package
import labelme2coco
import json
import os
import shutil

# set directory that contains labelme annotations and image files
labelme_folder = "datasets/seed-candice/dataset/test"

# set export dir
export_dir = "datasets/seed-candice/dataset/annotations"

# convert labelme annotations to coco
labelme2coco.convert(labelme_folder, export_dir)

# add segmentation
original_export = os.path.join(export_dir, "dataset.json")
export_file = os.path.join(export_dir, "instances_test.json")
annotations = json.load(open(original_export))

for i in range(len(annotations["annotations"])):
    bbox = annotations["annotations"][i]["bbox"]
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = xmin + bbox[2]
    ymax = ymin + bbox[3]
    segmentation = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]]
    annotations["annotations"][i]["segmentation"] = segmentation

annotations = json.dumps(annotations, indent = 2)
with open(export_file, 'w') as file:
    file.write(annotations)
os.remove(original_export)

# duplicate test as train
shutil.copyfile(export_file, os.path.join(export_dir, "instances_train.json"))