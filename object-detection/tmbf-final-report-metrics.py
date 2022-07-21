import os
import json
import pandas as pd

with open(os.path.join("./dataset","dataset.json")) as anns:
    annotations = json.load(anns)

label_id_to_name = {}
ann = []

for category in annotations['categories']:
    # print(category['name'])
    label_id_to_name[category['id']] = category['name']

for annotation in annotations['annotations']:  
    ann.append(label_id_to_name[annotation['category_id']])

curator = pd.DataFrame(ann, columns =['ann'])

curator["class"] = curator["ann"].str.split("-").str[0]
curator["order"] = curator["ann"].str.split("-").str[1]
curator["family"] = curator["ann"].str.split("-").str[2]
curator["genus"] = curator["ann"].str.split("-").str[3]
curator["species"] = curator["ann"].str.split("-").str[4]

print(curator['class'].value_counts())
print(curator['genus'].value_counts())
print(curator['species'].value_counts())