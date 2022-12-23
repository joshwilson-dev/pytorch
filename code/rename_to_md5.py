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
        for root, dirs, files in os.walk(os.getcwd()):
            for file in files:
                if file.endswith(".JPG"):
                    # calculate md5 and rename image
                    image_path = os.path.join(root, file)
                    hash_md5 = hashlib.md5()
                    with open(image_path, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_md5.update(chunk)
                    output = hash_md5.hexdigest()
                    image_output = os.path.join(root, output + ".JPG")
                    os.rename(image_path, image_output)
                    # rename associated annotation file if it exists
                    annotation_file = os.path.splitext(file)[0] + ".json"
                    if os.path.isfile(annotation_file):
                        annotation_output = os.path.join(root, output + ".json")
                        os.rename(annotation_file, annotation_output)

        # update annotation file to point to new name
        for root, dirs, files in os.walk(os.getcwd()):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        data["imagePath"] = os.path.splitext(file)[0] + '.JPG'
                    with open(file_path, 'w') as f:
                        json.dump(data, f, indent=2)