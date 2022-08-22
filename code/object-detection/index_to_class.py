import json
import os
import tkinter
from tkinter import filedialog

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

with open(os.path.join(file_path_variable, "dataset.json")) as anns:
    annotations = json.load(anns)
index_to_class = {}
for count in range(len(annotations["categories"])):
    species = annotations["categories"][count]["name"].split("_")[0]
    lifestage = annotations["categories"][count]["name"].split("_")[-1]
    index_to_class[str(count + 1)] = species + "-" + lifestage
with open('../../models/temp/index_to_class.txt', 'w') as file:
    file.write(json.dumps(index_to_class, indent = 4))