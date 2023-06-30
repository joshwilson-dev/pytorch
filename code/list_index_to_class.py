import json

root = "./datasets/bird-mask/dataset/annotations/"
index_to_class = json.load(open(root + "index_to_class.json"))
test_annotation = json.load(open(root + "instances_test.json"))

counts = {}

for label in index_to_class.items():
    counts[label[0]] = 0
    print(label[1]["name"])
print("All")
for annotation in test_annotation["annotations"]:
    counts[str(annotation["category_id"])] += 1

for count in counts.values():
    print(count)
print(sum(counts.values()))