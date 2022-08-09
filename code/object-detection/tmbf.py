import json
with open("Z:\drone-based-bird-survey\dataset\dataset.json") as anns:
    annotations = json.load(anns)

label_id_to_name = {}

for category in annotations['categories']:
    label_id_to_name[category['id']] = category['name']

curator = []

for annotation in annotations['annotations']:
    species = label_id_to_name[annotation['category_id']]
    curator.append(str([species]))

import numpy as np
values, counts = np.unique(curator, return_counts=True)
print(values)
print(counts)