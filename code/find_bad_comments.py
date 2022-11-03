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
import piexif

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

def is_float(string):
    try:
        string = float(string)
        return string
    except: return string

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
                    if "fully annotated" in root or "background" in root:
                        if "masks" in root:
                            print(os.path.join(root, file))
                        # if not file.endswith(".json"):
                        #     if not file.endswith(".ini"):
                        #         image_file_path = os.path.join(root, file)
                        #         image = Image.open(image_file_path)
                        #         exif_dict = piexif.load(image.info['exif'])
                        #         try:
                        #             comments = json.loads("".join(map(chr, [i for i in exif_dict["0th"][piexif.ImageIFD.XPComment] if i != 0])))
                        #             gsd = is_float(comments["gsd"])
                        #             substrate = is_float(comments["ecosystem typology"])
                        #         except:
                        #             print(os.path.join(root, file))