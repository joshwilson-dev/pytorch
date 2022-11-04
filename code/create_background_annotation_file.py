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
from PIL import Image
import json
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
        "Are you sure you want to create an empty annotation for the files in:\n" + file_path_variable)
    if check =="yes":
        os.chdir(file_path_variable)
        # iterate through files in dir
        for root, dirs, files in os.walk(os.getcwd()):
            for file in files:
                image = Image.open(file)
                width, height = image.size
                annotation = {
                    "version": "5.0.1",
                    "flags": {},
                    "shapes": [],
                    "imagePath": file,
                    "imageData": 'null',
                    "imageHeight": height,
                    "imageWidth": width}
                annotation_str = json.dumps(annotation, indent = 2).replace('"null"', 'null')
                annotation_path = os.path.join(root, os.path.splitext(file)[0] + ".json")
                with open(annotation_path, 'w') as annotation_file:
                    annotation_file.write(annotation_str)