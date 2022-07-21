################
#### Header ####
################

# Title: md5 image and log renamer 
# Author: Josh Wilson
# Date: 26-07-2022

###############
#### Setup ####
###############

import hashlib
import os
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
        "Are you sure you want to rename the files in:\n" + file_path_variable)
    if check =="yes":
        os.chdir(file_path_variable)
        # iterate through files in dir
        for file in os.listdir():
            if file.endswith(".JPG"):
                # calculate md5 and rename image
                image_file = file
                hash_md5 = hashlib.md5()
                with open(image_file, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
                output = hash_md5.hexdigest()
                image_output = output + ".JPG"
                os.rename(image_file, image_output)
                # rename associated annotation file if it exists
                annotation_file = os.path.splitext(file)[0] + ".json"
                if os.path.isfile(annotation_file):
                    annotation_output = output + ".json"
                    os.rename(annotation_file, annotation_output)

        # update annotation file to point to new name
        for file in os.listdir():
            if file.endswith(".json"):
                with open(file, 'r') as f:
                    data = json.load(f)
                    data["imagePath"] = os.path.splitext(file)[0] + '.JPG'
                with open(file, 'w') as f:
                    json.dump(data, f, indent=2)