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
# image_file_path = os.path.abspath("C:/Users/uqjwil54/Documents/trial/2022-10-11 1020/backgrounds/fc61990811f490df4b32a7780fff688d.JPG")
# background_dataset = "C:/Users/uqjwil54/Documents/trial/dataset/backgrounds/fc61990811f490df4b32a7780fff688d.JPG"
# shutil.copyfile(image_file_path, background_dataset)

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
        mask_dataset = "dataset/masks"
        background_dataset = "dataset/backgrounds"
        for path in mask_dataset, background_dataset:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
        # iterate through files in dir
        for root, dirs, files in os.walk(os.getcwd()):
            for file in files:
                if "dataset" not in root:
                    if "masks" in root:
                        if file.endswith(".json"):
                            annotation_file_name = file
                            annotation_file_path = os.path.join(root, annotation_file_name)
                            annotation = json.load(open(annotation_file_path))
                            image_file_name = annotation["imagePath"]
                            image_file_path = os.path.join(root, image_file_name)
                            print(image_file_path)
                            shutil.copyfile(image_file_path, os.path.join(mask_dataset, image_file_name))
                            shutil.copyfile(annotation_file_path, os.path.join(mask_dataset, annotation_file_name))
                    if "backgrounds" in root:
                        image_file_path = os.path.join(root, file)
                        print(image_file_path)
                        shutil.copyfile(image_file_path, os.path.join(background_dataset, file))