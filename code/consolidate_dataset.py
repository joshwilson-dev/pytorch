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
from PIL import Image
import hashlib
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
        "Are you sure you want to create a dataset from the files in:\n" + file_path_variable)
    if check =="yes":
        os.chdir(file_path_variable)
        input = "input"
        if os.path.exists(input):
            shutil.rmtree(input)
        os.makedirs(input)
        # iterate through files in dir
        for root, dirs, files in os.walk(os.getcwd()):
            for file in files:
                if "fully annotated" in root or "partially annotated" in root or "backgrounds" in root:
                    if file.endswith(".json"):
                        print(file)

                        # Input paths
                        annotation_input_name = file
                        annotation_input_path = os.path.join(root, annotation_input_name)
                        annotation = json.load(open(annotation_input_path))
                        image_input_name = annotation["imagePath"]
                        image_input_path = os.path.join(root, image_input_name)

                        # Calculate md5
                        image = Image.open(image_input_path)
                        md5 = hashlib.md5(image.tobytes()).hexdigest()

                        # Output paths
                        annotation_output_name = md5 + '.json'
                        annotation_output_path = os.path.join(input, annotation_output_name)
                        image_output_name = md5 + '.jpg'
                        image_output_path = os.path.join(input, image_output_name)

                        # Update annotation
                        annotation["imagePath"] = image_output_name
                        annotation = json.dumps(annotation, indent = 2).replace('"null"', 'null')

                        # Write annotation
                        with open(annotation_output_path, 'w') as annotation_file:
                            annotation_file.write(annotation)

                        # Write original image path to exif comments
                        exif = piexif.load(image.info['exif'])
                        exif["0th"][piexif.ImageIFD.XPComment] = json.dumps(image_input_name).encode('utf-16le')
                        exif = piexif.dump(exif)

                        # Save image
                        image.save(image_output_path, exif = exif)