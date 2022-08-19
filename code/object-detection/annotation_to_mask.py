################
#### Header ####
################

# Title: Create Mask
# Author: Josh Wilson
# Date: 11-08-2022
# Description: 
# This script creates the png masks file from labelme polygon annotaitons

###############
#### Setup ####
###############
import json
import os
import tkinter
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageDraw

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
        "Are you sure you want to create masks from the files in:\n" + file_path_variable)
    if check =="yes":
        os.chdir(file_path_variable)
        # walk through image files and crop
        for file in os.listdir():
            if file.endswith(".JPG"):
                # check if image is labelled
                annotation = os.path.splitext(file)[0] + '.json'
                print(annotation)
                if os.path.exists(annotation):
                    print("Creating Masks...", file)
                    image = Image.open(file)
                    width = image.width
                    height = image.height
                    img = Image.new("P", (width, height))
                    mask = ImageDraw.Draw(img)
                    mask.polygon([(0,0), (width,0), (width,height), (0, height)], fill = (0, 0, 0))
                    with open(annotation) as anns:
                        dictionary = json.load(anns)
                    for index in range(len(dictionary['shapes'])):
                        segmentation = dictionary["shapes"][index]['points']
                        points = list(tuple(point) for point in segmentation)
                        mask.polygon(points, fill = (index + 1, index + 1, index + 1))
                    mask_name = os.path.join(os.path.splitext(file)[0] + "-mask.png")
                    img.save(mask_name)