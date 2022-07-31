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
            dataset = os.path.join(os.getcwd(), "dataset")
            if not os.path.exists(dataset):
                os.makedirs(dataset, exist_ok = True)
            for file in files:
                if file.endswith(".json"):
                    annotation_file_name = file
                    image_file_name = os.path.splitext(file)[0] + ".JPG"
                    annotation_file_path = os.path.join(root, file)
                    image_file_path = os.path.splitext(os.path.abspath(annotation_file_path))[0] + ".JPG"
                    if "tool" not in image_file_path:
                        print(image_file_path)
                        shutil.copyfile(image_file_path, os.path.join(dataset, image_file_name))
                        shutil.copyfile(annotation_file_path, os.path.join(dataset, annotation_file_name))