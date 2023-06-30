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
from PIL import Image
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

                    # Inputs
                    image_input_name = file
                    image_input_path = os.path.join(root, image_input_name)
                    annotation_input_name = os.path.splitext(image_input_name)[0] + ".json"
                    annotation_input_path = os.path.join(root, annotation_input_name)

                    # Calculate md5
                    image = Image.open(image_input_path)
                    md5 = hashlib.md5(image.tobytes()).hexdigest()

                    # Outputs
                    image_output_name = md5 + ".jpg"
                    image_output_path = os.path.join(root, image_output_name)
                    annotation_output_name = md5 + ".json"
                    annotation_output_path = os.path.join(root, annotation_output_name)

                    # Update image name
                    image.close()
                    os.rename(image_input_path, image_output_path)
                    
                    # Check for annotation file
                    if os.path.isfile(annotation_input_name):

                        # Rename annotation
                        os.rename(annotation_input_path, annotation_output_path)

                        # Update annotation
                        annotation = json.load(open(annotation_output_path))
                        annotation["imagePath"] = image_output_name
                        annotation = json.dumps(annotation, indent = 2)

                        # Write annotation
                        with open(annotation_output_path, 'w') as annotation_file:
                            annotation_file.write(annotation)