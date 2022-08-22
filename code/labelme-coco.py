# import package
import labelme2coco
import os
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

# did the user select a dir or cancel?
if len(file_path_variable) > 0:
    # confirm dir with user
    check = messagebox.askquestion(
        "CONFIRM",
        "Are you sure you want to create a coco annotation file from the label files in:\n" + file_path_variable)
    if check =="yes":
        os.chdir(file_path_variable)

        # set directory that contains labelme annotations and image files
        labelme_folder = file_path_variable

        # set path for coco json to be saved
        save_json_path = file_path_variable

        # convert labelme annotations to coco
        labelme2coco.convert(labelme_folder, save_json_path)

        # fix indexing error
        annotation_path = os.path.join(file_path_variable, "dataset.json")
        annotation_file = open(annotation_path)
        annotation = json.load(annotation_file)
        for ann in annotation["annotations"]:
            for index in reversed(range(0, len(annotation["categories"]))):
                if ann["category_id"] == index:
                    ann["category_id"] = index + 1
        for ann in annotation["categories"]:
            for index in reversed(range(0, len(annotation["categories"]))):
                if ann["id"] == index:
                    ann["id"] = index + 1
        # add 5 pixels to bounding boxes
        # for index in range(len(annotation["annotations"])):
        #     box = annotation["annotations"][index]["bbox"]
        #     box[0] += -5
        #     box[1] += -5
        #     box[2] += 10
        #     box[3] += 10
        #     area = box[2] * box[3]
        #     annotation["annotations"][index]["bbox"] = box
        #     annotation["annotations"][index]["area"] = area

        # create mask from bounding box
        for index in range(len(annotation["annotations"])):
            box = annotation["annotations"][index]["bbox"]
            segmentation = [
                [
                    box[0],
                    box[1],
                    box[0] + box[2],
                    box[1],
                    box[0] + box[2],
                    box[1] + box[3],
                    box[0],
                    box[1] + box[3]
                    ]
                ]
            annotation["annotations"][index]["segmentation"] = segmentation
        
        # make all classes bird
        for index in range(len(annotation["annotations"])):
            annotation["annotations"][index]["category_id"] = 1
        annotation["categories"] = [{"id": 1, "name": "Bird","supercategory": "Bird"}]

        # save to file
        annotation = json.dumps(annotation, indent=4)

        with open(annotation_path, "w") as outfile:
            outfile.write(annotation)