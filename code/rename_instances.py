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
                if file.endswith(".json"):
                    annotation_path = os.path.join(root, file)
                    annotation = json.load(open(annotation_path))
                    print("Renaming instances in", file)
                    for i in range(len(annotation["shapes"])):
                        label = json.loads(annotation["shapes"][i]["label"].replace("'", '"'))
                        # if label["pose"] not in poses and annotation["shapes"][i]["shape_type"] == "polygon":
                        #     print("Bad pose", file)
                        if "'pose': 'unknown'" in annotation["shapes"][i]["label"]:
                            annotation["shapes"][i]["label"] = annotation["shapes"][i]["label"].replace("'pose': 'unknown'", "'pose': 'resting'")
                        # old_label = annotation["shapes"][i]["label"]
                        # try:
                        #     new_label = label_dict[old_label]
                        # except:
                        #     print(old_label, "not in label dictionary")
                        #     continue
                        # annotation["shapes"][i]["label"] = new_label
                    annotation_str = json.dumps(annotation, indent = 2).replace('"null"', 'null')
                    with open(annotation_path, 'w') as annotation_file:
                        annotation_file.write(annotation_str)