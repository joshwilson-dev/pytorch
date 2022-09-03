################
#### Header ####
################

# Title: Balance Dataset
# Author: Josh Wilson
# Date: 07-07-2022
# Description: 
# This script balances the number of images of species at each gsd

###############
#### Setup ####
###############
import json
import os
from pickle import FALSE, TRUE
from PIL import Image
import tkinter
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image
import piexif
import shutil
import torchvision.transforms as T
import random
import hashlib
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
        "Are you sure you want to split the dataset in:\n" + file_path_variable)
    if check =="yes":
        os.chdir(file_path_variable)
        # walk through image/label files and get gsd and species
        for file in os.listdir():
            if file.endswith(".JPG"):
                # parse image exif data
                image = Image.open(file)
                exif_dict = piexif.load(image.info['exif'])
                exif_bytes = piexif.dump(exif_dict)
                try:
                    comments = json.loads("".join(map(chr, [i for i in exif_dict["0th"][piexif.ImageIFD.XPComment] if i != 0])))
                except:
                    break
                gsd = float(comments["gsd"])
                annotation_name = os.path.splitext(file)[0] + ".json"
                gsd_range = [0, 0.0075, 0.015]
                gsd_class = ["fine", "coarse"]
                for gsd_index in range(len(gsd_range) - 1):
                    gsd_min = gsd_range[gsd_index]
                    gsd_max = gsd_range[gsd_index + 1]
                    dir = gsd_class[gsd_index]
                    if not os.path.exists("gsd_split/" + dir):
                        os.makedirs("gsd_split/" + dir)
                    if gsd_min < gsd < gsd_max:
                        print("Copying:", file)
                        # save image to directory
                        image_path = os.path.join("gsd_split", dir, file)
                        annotation_path = os.path.join("gsd_split", dir, annotation_name)
                        shutil.copy(file, image_path)
                        shutil.copy(annotation_name, annotation_path)
                    elif gsd < gsd_min:
                        # downscale image to lower resolution
                        gsd_req = random.uniform(gsd_min, gsd_max)
                        scale_req = gsd/gsd_req
                        width, height = image.size
                        comments["gsd"] = str(gsd_req)
                        comments["image_width"] = str(width*scale_req)
                        comments["image_length"] = str(width*scale_req)
                        exif_dict["0th"][piexif.ImageIFD.XPComment] = json.dumps(comments).encode('utf-16le')
                        exif_bytes = piexif.dump(exif_dict)
                        size = int(scale_req*min(width, height))
                        transformed_img = T.Resize(size=size)(image)
                        annotation = json.load(open(annotation_name))
                        for shape_index in range(len(annotation["shapes"])):
                            for point_index in range(len(annotation["shapes"][shape_index]["points"])):
                                annotation["shapes"][shape_index]["points"][point_index][0] *= scale_req
                                annotation["shapes"][shape_index]["points"][point_index][1] *= scale_req
                        md5hash = hashlib.md5(transformed_img.tobytes()).hexdigest()
                        print("Downscaling:", file, "as ", md5hash + ".JPG")
                        image_path = os.path.join("gsd_split", dir, md5hash + ".JPG")
                        transformed_img.save(image_path, exif = exif_bytes)
                        annotation_output = os.path.join("gsd_split", dir, md5hash + ".json")
                        annotation["imagePath"] = md5hash + ".JPG"
                        with open(annotation_output, 'w') as new_annotation:
                            json.dump(annotation, new_annotation, indent=2)
