################
#### Header ####
################

# Title: consolidate labelled images into dataset 
# Author: Josh Wilson
# Date: 01-06-2022
# Description: 
# This script runs through sub-dir of the selected directory
# looking for annotated image files and copying them to
# either a private or public dataset folder.

###############
#### Setup ####
###############

import os
import shutil
import tkinter
from tkinter import filedialog
from tkinter import messagebox
import json

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
        "Are you sure you want to create a dataset from the files in:\n" + file_path_variable)
    if check =="yes":
        os.chdir(file_path_variable)
        # iterate through files in dir
        for root, dirs, files in os.walk(os.getcwd()):
            for file in files:
                if file.endswith(tuple(["jpg", "jpeg", "png", "tif", "tiff"])):
                    print("Incorrect Extension:", file)
                    if file.endswith(tuple(["jpg", "jpeg"])):
                        print("renaming:", file)
                        file_path = os.path.join(root, file)
                        new_name = os.path.splitext(file)[0] + ".JPG"
                        new_file_path = os.path.join(root, new_name)
                        annotation_name = os.path.splitext(file)[0] + ".json"
                        annotation_path = os.path.join(root, annotation_name)
                        if os.path.exists(annotation_path):
                            annotation = json.load(open(annotation_path))
                            annotation["imagePath"] = new_name
                            new_annotation = json.dumps(annotation, indent=4)
                            with open(annotation_path, "w") as outfile:
                                outfile.write(new_annotation)
                        os.rename(file_path, new_file_path)
                    