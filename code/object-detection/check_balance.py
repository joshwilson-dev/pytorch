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
                # comments = json.loads("".join(map(chr, [i for i in exif_dict["0th"][piexif.ImageIFD.XPComment] if i != 0])))
                # gsd = float(comments["gsd"])
                # parse annotations
                annotation = os.path.splitext(file)[0] + ".json"
                with open(annotation) as anns:
                    annotations = json.load(anns)
                # data = pd.DataFrame({"image": file, "gsd": gsd}, index=[0])
                data = pd.DataFrame({"image": file}, index=[0])
                for instance in annotations['shapes']:
                    if instance["label"] not in data:
                        data[instance["label"]] = 1
                    else: data[instance["label"]] += 1
                dataset_content = pd.concat([dataset_content, data], ignore_index=True)

instances_species = dataset_content.sum().reset_index().rename(columns={"index": "species", 0: 'count'})
instances_species = instances_species[instances_species["species"] != "image"]
instances_species = instances_species[instances_species["species"] != "gsd"]
print(instances_species)
# for species in instances_species["species"]:
    # print(species)