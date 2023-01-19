################
#### Header ####
################

# Title: Add GSD to Annotation
# Author: Josh Wilson
# Date: 26-12-2022
# Description: 
# This script runs through sub-dirs of the selected directory
# looking for images with annotation files and adds the image
# gsd to the flag key of the annotation

###############
#### Setup ####
###############

import os
import tkinter
from tkinter import filedialog
from tkinter import messagebox
import json
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

file_path_variable = search_for_file_path()

# did the user select a dir or cancel?
if len(file_path_variable) > 0:
    # confirm dir with user
    check = messagebox.askquestion(
        "CONFIRM",
        "Are you sure you want to add an empty annotation too the files in:\n" + file_path_variable)
    if check =="yes":
        os.chdir(file_path_variable)
        # iterate through files in dir
        for root, dirs, files in os.walk(os.getcwd()):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    annotation_file = os.path.splitext(file)[0] + ".json"
                    annotation_file_path = os.path.join(root, annotation_file)
                    if os.path.exists(annotation_file_path):
                        continue
                    else:
                        print("Adding empty annotation to: ", file)
                        image_file_path = os.path.join(root, file)
                        image = Image.open(image_file_path)
                        width, height = image.size 
                        annotation = {
                            "version": "5.0.1",
                            "flags": {},
                            "shapes": [],
                            "imagePath": file,
                            "imageData": "null",
                            "imageHeight": height,
                            "imageWidth": width}

                        annotation = json.dumps(annotation, indent = 2).replace('"null"', 'null')
                        with open(annotation_file_path, 'w') as annotation_file:
                            annotation_file.write(annotation)