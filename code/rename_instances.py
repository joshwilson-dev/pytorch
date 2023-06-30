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

label_dict = {
    "{'name': 'Masked Lapwing', 'class': 'aves', 'order': '', 'family': '', 'genus': '', 'species': '', 'age': 'adult'}": "{'name': 'Masked Lapwing', 'class': 'aves', 'order': 'charadriiformes', 'family': 'charadriidae', 'genus': 'vanellus', 'species': 'miles', 'age': 'adult'}"
}
poses = ["flying", "preening", "resting", "lying"]
# did the user select a dir or cancel?
if len(file_path_variable) > 0:
    # confirm dir with user
    check = messagebox.askquestion(
        "CONFIRM",
        "Are you sure you want to rename the instances in:\n" + file_path_variable)
    if check =="yes":
        os.chdir(file_path_variable)
        # iterate through files in dir
        for root, dirs, files in os.walk(os.getcwd()):
            for file in files:
                if "fully annotated" in root or "partially annotated" in root:
                    if file.endswith(".json"):
                        print("Renaming instances in", file)
                        annotation_path = os.path.join(root, file)
                        print(annotation_path)
                        annotation = json.load(open(annotation_path))
                        check = False

                        for i in range(len(annotation["shapes"])):
                            label = json.loads(annotation["shapes"][i]["label"].replace("'", '"'))
                            if label["pose"] == "unknown":
                                label["pose"] = "resting"
                            try:
                                label["sex"]
                            except:
                                label["sex"] = "unknown"
                            try:
                                label["obscured"]
                            except:
                                label["obscured"] = "no"
                            annotation["shapes"][i]["label"] = json.dumps(label).replace('"', "'")
                        annotation_str = json.dumps(annotation, indent = 2).replace('"null"', 'null')
                        with open(annotation_path, 'w') as annotation_file:
                            annotation_file.write(annotation_str)