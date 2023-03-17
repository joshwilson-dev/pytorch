################
#### Header ####
################

# Title: curate data within sub directory
# Author: Josh Wilson
# Date: 02-06-2022
# Description: 
# This script takes a set of images with annotation files and creates
# a dataset with a balance of each class and background

###############
#### Setup ####
###############

import os
import tkinter
from tkinter import filedialog
from tkinter import messagebox
import json
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
# did the user select a dir or cancel?
if len(file_path_variable) > 0:
    # confirm dir with user
    check = messagebox.askquestion(
        "CONFIRM",
        "Do you want to create a dataset from:\n" + file_path_variable)
    if check =="yes":
        os.chdir(file_path_variable)
        # create dictionaries to store data
        dataset_keys = ["image_path", "instance_class", "instance_shape_type"]
        data = {k:[] for k in dataset_keys}
        # iterate through images
        for root, dirs, files in os.walk(os.getcwd()):
            for file in files:
                if file.endswith(".json"):
                    # print("Recording Patch Data From: ", file)
                    # load the annotation
                    annotation_name = file
                    annotation_path = os.path.join(root, annotation_name)
                    annotation = json.load(open(annotation_path))
                    instance_in_patch = False
                    for shape in annotation["shapes"]:
                        # save instance to temp data
                        data["image_path"].append(file)
                        data["instance_class"].append(shape["label"])
                        data["instance_shape_type"].append(shape["shape_type"])
                        instance_in_patch = True
                        label = json.loads(shape["label"].replace("'", '"'))
                        if label["name"] == "Masked Lapwing" and label["order"] == '':
                            print(file)
                    # if there was no instances in the patch add it as background
                    if instance_in_patch == False:
                        data["image_path"].append(file)
                        data["instance_class"].append("background")
                        data["instance_shape_type"].append("null")

        # convert dictionary to dataframe
        data = pd.DataFrame(data=data)
        # class abundance
        total_class_count = (
            data
            .instance_class
            .value_counts())
        print("\tClass Count:\n{}\n".format(total_class_count))
        # polygon abundance
        polygon_class_count = (
            data
            .query("instance_shape_type == 'polygon'")
            .instance_class
            .value_counts())
        print("\tPolygon Class Count:\n{}\n".format(polygon_class_count))
        for name in data["instance_class"].unique():
            print(name)