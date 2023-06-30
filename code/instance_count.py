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
        dataset_keys = ["Image Path", "Common Name", "Scientific Name", "Age", "Pose", "Shape Type"]
        data = {k:[] for k in dataset_keys}
        # iterate through images
        for root, dirs, files in os.walk(os.getcwd()):
            for file in files:
                if file.endswith(".json"):
                    # if 'fully annotated' in root or 'partially annotated' in root:
                        print("Recording Patch Data From: ", root, file)
                        # load the annotation
                        annotation_name = file
                        annotation_path = os.path.join(root, annotation_name)
                        annotation = json.load(open(annotation_path))
                        instance_in_patch = False
                        for shape in annotation["shapes"]:
                            try:
                                label = json.loads(shape["label"].replace("'", '"'))
                            except:
                                print(shape["label"])
                                import sys
                                sys.exit("here")
                            common_name = label["name"]
                            scientific_name = label["genus"].capitalize() + " " + label["species"]
                            age = label["age"]
                            pose = label["pose"]
                            # save instance to temp data
                            data["Image Path"].append(file)
                            data["Common Name"].append(common_name)
                            data["Scientific Name"].append(scientific_name)
                            data["Age"].append(age)
                            data["Pose"].append(pose)
                            data["Shape Type"].append(shape["shape_type"])
                            instance_in_patch = True
                        # if there was no instances in the patch add it as background
                        if instance_in_patch == False:
                            data["Image Path"].append(file)
                            data["Common Name"].append("background")
                            data["Scientific Name"].append("background")
                            data["Age"].append("background")
                            data["Pose"].append("background")
                            data["Shape Type"].append("background")

        # convert dictionary to dataframe
        data = pd.DataFrame(data=data)
        # Image Count
        print(len(data["Image Path"].unique()))
        # Fine Count
        fine_instance_summary = (
            data
            .drop(["Image Path"], axis=1)
            .value_counts()
            .reset_index(name = "Instances")
            .sort_values(by=['Common Name', "Age", "Pose", "Shape Type"]))
        # Coarse Count
        coarse_instance_summary = (
            data
            .drop(["Image Path", "Pose", "Shape Type"], axis=1)
            .value_counts()
            .reset_index(name = "Instances")
            .sort_values(by=["Common Name", "Age"]))
        # save data to csv
        fine_instance_summary.to_csv('Fine_Instance_Summary.csv', index=False)
        coarse_instance_summary.to_csv('Coarse_Instance_Summary.csv', index=False)