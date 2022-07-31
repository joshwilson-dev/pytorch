################
#### Header ####
################

# Title: Balance Dataset
# Author: Josh Wilson
# Date: 07-07-2022
# Description: 
# This script balances the number of images of species at each gsd

###############
#### Setup ####
###############
import json
import os
from pickle import FALSE, TRUE
from PIL import Image
import tkinter
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image
import piexif
import pandas as pd
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

save_dir = "./balanced/"

# did the user select a dir or cancel?
if len(file_path_variable) > 0:
    # confirm dir with user
    check = messagebox.askquestion(
        "CONFIRM",
        "Are you sure you want to create a dataset from the files in:\n" + file_path_variable)
    if check =="yes":
        os.chdir(file_path_variable)
        # walk through image/label files and get gsd and species
        dataset_content = pd.DataFrame()
        for file in os.listdir():
            if file.endswith(".JPG"):
                # parse image exif data
                image = Image.open(file)
                exif_dict = piexif.load(image.info['exif'])
                comments = json.loads("".join(map(chr, [i for i in exif_dict["0th"][piexif.ImageIFD.XPComment] if i != 0])))
                gsd = float(comments["gsd"])
                # parse annotations
                annotation = os.path.splitext(file)[0] + ".json"
                with open(annotation) as anns:
                    annotations = json.load(anns)
                data = pd.DataFrame({"image": file, "gsd": gsd}, index=[0])
                for instance in annotations['shapes']:
                    if instance["label"] not in data:
                        data[instance["label"]] = 1
                    else: data[instance["label"]] += 1
                dataset_content = pd.concat([dataset_content, data], ignore_index=True)

instances_species = dataset_content.sum().reset_index().rename(columns={"index": "species", 0: 'count'})
instances_species = instances_species[instances_species["species"] != "image"]
instances_species = instances_species[instances_species["species"] != "gsd"]
print(instances_species)

blackout = []
for species, count in instances_species.itertuples(index=False):
    if count < 30:
        instances_species = instances_species[instances_species["species"] != species]
        blackout.append(species)
print(instances_species)
print(blackout)

images = dataset_content.filter(items=['image'])
images['species'] = "all"
for species in instances_species["species"]:
    min_instances = max(instances_species["count"])
    instances = int(instances_species[instances_species["species"] == species]["count"])
    print(species)
    while instances < min_instances:
        acceptible_images = dataset_content[dataset_content[str(species)] > 0]
        image = list(acceptible_images['image'].sample(n = 1))[0]
        data = pd.DataFrame({"image": image, "species": species}, index=[0])
        instance_counts = int(dataset_content[dataset_content["image"] == image][str(species)])
        instances = instances + instance_counts
        images = pd.concat([images, data], ignore_index=True)

count = 1
# check if save directory exists and if no create one 
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for file, species in images.itertuples(index=False):
    image = Image.open(file)
    annotation = os.path.splitext(file)[0] + ".json"
    with open(annotation) as anns:
        annotations = json.load(anns)
    annotations_removed = 0
    for i in range(0, len(annotations["shapes"])):
        i = i - annotations_removed
        if species != "all" or annotations["shapes"][i]["label"] in blackout:
            if (species != "all" and annotations["shapes"][i]["label"] != species) or annotations["shapes"][i]["label"] in blackout:
                # blackout part of image
                box_x1 = annotations["shapes"][i]["points"][0][0]
                box_x2 = annotations["shapes"][i]["points"][1][0]
                box_y1 = annotations["shapes"][i]["points"][0][1]
                box_y2 = annotations["shapes"][i]["points"][1][1]
                width = abs(round(box_x2 - box_x1))
                height = abs(round(box_y2 - box_y1))
                topleft = (round(min(box_x1, box_x2)), round(min(box_y1, box_y2)))
                black_box = Image.new("RGB", (width, height))
                image.paste(black_box, topleft)
                # remove label
                del annotations["shapes"][i]
                annotations_removed += 1
    # save crop
    if len(annotations["shapes"]) > 0:
        exif_dict = piexif.load(image.info["exif"])
        exif_bytes = piexif.dump(exif_dict)
        annotation_output = os.path.join(save_dir, annotation)
        if not os.path.exists(os.path.join(save_dir, file)):
            image.save(save_dir + file, exif = exif_bytes)
            with open(annotation_output, 'w') as new_annotation:
                json.dump(annotations, new_annotation, indent=2)
        else:
            image_new = os.path.splitext(file)[0] + str(count) + ".JPG"
            image.save(save_dir + image_new, exif = exif_bytes)
            annotation_output = os.path.join(save_dir, os.path.splitext(file)[0] + str(count) + ".json")
            annotations["imagePath"] = image_new
            with open(annotation_output, 'w') as new_annotation:
                json.dump(annotations, new_annotation, indent=2)
            count += 1