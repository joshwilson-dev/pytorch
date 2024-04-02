import json

root = "datasets/bird_2024_02_13/balanced/annotations/"
index_to_class = json.load(open(root + "index_to_class.json"))
test_annotation = json.load(open(root + "test.json"))
train_annotation = json.load(open(root + "train.json"))

test_counts = {}
train_counts = {}

for label in index_to_class.items():
    test_counts[label[0]] = 0
    train_counts[label[0]] = 0
    print(label[1])
print("All\nTEST")

for annotation in test_annotation["annotations"]:
    test_counts[str(annotation["category_id"])] += 1

for annotation in train_annotation["annotations"]:
    train_counts[str(annotation["category_id"])] += 1

for label, count in test_counts.items():
    print(count)
print(sum(test_counts.values()))

print("\nTRAIN")
for label, count in train_counts.items():
    print(count)
print(sum(train_counts.values()))

# define area ranges
# import numpy as np
# min_dim = 100
# max_dim = 1500
# steps = 14
# step_size = (max_dim - min_dim)/steps
# dims = np.arange(min_dim, max_dim, step_size)
# areaRng = [[0, 1e5 ** 2]] + [[dim, 1e5 ** 2] for dim in dims]

# area_count = {}
# for Rng in areaRng:
#     area_count[str(Rng)] = 0

# for annotation in test_annotation["annotations"]:
#     if annotation["category_id"] == 2:
#         for Rng in areaRng:
#             if Rng[0] < annotation["area"] and Rng[1] > annotation["area"]:
#                 area_count[str(Rng)] += 1
# print("AREAS")
# for i in area_count.keys():
#     print(i)
# for i in area_count.values():
#     print(i)