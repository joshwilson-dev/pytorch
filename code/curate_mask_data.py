################
#### Header ####
################

# Title: curate data within sub directory
# Author: Josh Wilson
# Date: 02-06-2022
# Description: 
# This script runs through sub-dirs of the selected directory
# looking for mask annotations and creates a csv with the gsd
# of each mask instance

###############
#### Setup ####
###############

import os
import csv
import tkinter
from tkinter import filedialog
from tkinter import messagebox
import json
from requests import get
from pandas import json_normalize
from PIL import Image
import piexif

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

def is_float(string):
    try:
        string = float(string)
        return string
    except: return string

file_path_variable = search_for_file_path()

# did the user select a dir or cancel?
if len(file_path_variable) > 0:
    # confirm dir with user
    check = messagebox.askquestion(
        "CONFIRM",
        "Are you sure you want to curate the files in:\n" + file_path_variable)
    if check =="yes":
        os.chdir(file_path_variable)
        # iterate through files in dir
        header = False
        dataset = {"label": [], "gsd": [], "image": []}
        for root, dirs, files in os.walk(os.getcwd()):
            for file in files:
                if "mask" in root:
                    if file.endswith(".json"):
                        print(file)
                        # load annotation
                        annotation_path = os.path.join(root, file)
                        annotation = json.load(open(annotation_path))
                        # load image
                        image_name = annotation["imagePath"]
                        image_path = os.path.join(root, image_name)
                        image = Image.open(image_path)
                        image_width, image_height = image.size
                        # read exif data
                        exif_dict = piexif.load(image.info['exif'])
                        # get gsd
                        comments = json.loads("".join(map(chr, [i for i in exif_dict["0th"][piexif.ImageIFD.XPComment] if i != 0])))
                        gsd = is_float(comments["gsd"])
                        for instance in annotation["shapes"]:
                            dataset["label"].append(instance["label"])
                            dataset["gsd"].append(gsd)
                            dataset["image"].append(image_name)
        # save dict to csv
        with open("masks.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(dataset.keys())
            writer.writerows(zip(*dataset.values()))