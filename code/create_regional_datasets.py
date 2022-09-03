################
#### Header ####
################

# Title: Crop Dataset
# Author: Josh Wilson
# Date: 29-07-2022
# Description: 
# This script takes an object detection dataset and crops the images
# and annotations to create an image classification dataset 

###############
#### Setup ####
###############
import json
import os
from PIL import Image
import hashlib
import tkinter
from tkinter import filedialog
from tkinter import messagebox
import piexif
import torchvision.transforms as T

#################
#### Content ####
#################

# create function for user to select dir
root = tkinter.Tk()
root.withdraw()

def search_for_file_path ():
    currdir = os.getcwd()
    tempdir = filedialog.askdirectory(
        parent=root,
        initialdir=currdir,
        title='Please select a directory')
    if len(tempdir) > 0:
        print ("You chose: %s" % tempdir)
    return tempdir

file_path_variable = search_for_file_path()

# did the user select a dir or cancel?
if len(file_path_variable) > 0:
    # confirm dir with user
    check = messagebox.askquestion(
        "CONFIRM",
        "Are you sure you want to create an hierarchical image classification dataset from the files in:\n" + file_path_variable)
    if check =="yes":
        os.chdir(file_path_variable)
        # load regional csv
        birds_by_region = json.load(open("birds_by_region.json"))
        regions = birds_by_region.keys()
        for region in regions:
            if not os.path.exists(region):
                os.makedirs(region)
        labels_dict = {}
        for region in regions:
            labels_dict[region] = {"images": [], "labels": []}
        if not os.path.exists("images"):
            os.makedirs("images")
        # walk through image files and crop
        for file in os.listdir():
            if file.endswith(".json") and file != "dataset.json" and file != "birds_by_region.json":
                print("Creating regional classifier data for", file)
                # read annotation file
                annotations = json.load(open(file))
                image_file = os.path.splitext(file)[0] + '.JPG'
                image = Image.open(image_file, mode="r")
                for index1 in range(len(annotations["shapes"])):

                    # crop image to instance
                    points = annotations["shapes"][index1]["points"]
                    box_x1 = points[0][0]
                    box_x2 = points[1][0]
                    box_y1 = points[0][1]
                    box_y2 = points[1][1]
                    box_left = min(box_x1, box_x2)
                    box_right = max(box_x1, box_x2)
                    box_top = min(box_y1, box_y2)
                    box_bottom = max(box_y1, box_y2)
                    instance = image.crop((box_left, box_top, box_right, box_bottom))
                    
                    # pad crop to square
                    width, height = instance.size
                    if height > width: pad = [int((height - width) / 2), 0]
                    else: pad = [0, int((width - height)/2)]
                    instance = T.Pad(padding=pad)(instance)

                    # resize crop to 224
                    instance = T.transforms.Resize(224)(instance)
                    
                    # determine image hash
                    md5hash = hashlib.md5(instance.tobytes()).hexdigest()
                    image_name = md5hash + ".JPG"

                    # get image exif
                    exif_dict = piexif.load(image.info["exif"])
                    exif_bytes = piexif.dump(exif_dict)
                    #TODO update exif image size

                    # save instance to images
                    instance.save(os.path.join("images", image_name), exif = exif_bytes)

                    # get species name and age from label
                    label = annotations["shapes"][index1]["label"].split("_")
                    species = label[-3] + " " + label[-2]
                    age = label[-1]

                    # check if species is unknown
                    if "unknown" not in species:
                        species_in_region = 0
                        # check which regions species occurs in
                        for region in regions:
                            if species in birds_by_region[region]:
                                # save label and image name into relveant label dict key
                                species_in_region = 1
                                labels_dict[region]["images"].append(image_name)
                                labels_dict[region]["labels"].append(species + " " + age)
                        if species_in_region == 0:
                            print(species, " not in any region")
        for region in regions:
            dataset_path = os.path.join(region, "dataset.json")
            instance_annotations = json.dumps(labels_dict[region], indent=2)
            with open(dataset_path, "w") as instance_annotation_file:
                instance_annotation_file.write(instance_annotations)