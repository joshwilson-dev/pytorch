import os
import shutil
import json

index_to_class = json.load(open("resources/index_to_class.json"))
class_to_index = {value["name"]: key for key, value in index_to_class.items()}

root = "data/original"
os.chdir(root)

if os.path.exists("consolidated"):
    shutil.rmtree("consolidated")
os.makedirs("consolidated")

# iterate through files in dir
count = 0
for root, dirs, files in os.walk(os.getcwd()):
    for file in files:
        if "fully annotated" in root or "backgrounds" in root:
            if file.endswith(".json"):
                print(file)

                # Input paths
                annotation_input_name = file
                annotation_input_path = os.path.join(root, annotation_input_name)
                annotation = json.load(open(annotation_input_path))
                image_input_name = annotation["imagePath"]
                image_input_path = os.path.join(root, image_input_name)
                
                # Output paths
                annotation_output_name = str(count) + '.json'
                annotation_output_path = os.path.join("consolidated", annotation_output_name)
                image_output_name = str(count) + '.jpg'
                image_output_path = os.path.join("consolidated", image_output_name)
                count += 1

                # Update annotation
                annotation["imagePath"] = image_output_name
                annotation["originalimagePath"] = image_input_name
                for i in range(len(annotation["shapes"])):
                    label = json.loads(annotation["shapes"][i]["label"].replace("'", '"'))
                    if label["name"] != "bird":
                        new_label = index_to_class[class_to_index[label["name"].lower()]]
                        label["class"] = "aves"
                        label["order"] = new_label["order"]
                        label["family"] = new_label["family"]
                        label["genus"] = new_label["genus"]
                        label["species"] = new_label["species"]
                    else:
                        label["class"] = "aves"
                        label["order"] = "unknown"
                        label["family"] = "unknown"
                        label["genus"] = "unknown"
                        label["species"] = "unknown"
                    annotation["shapes"][i]["label"] = json.dumps(label).replace('"', "'")
                annotation = json.dumps(annotation, indent = 2).replace('"null"', 'null')

                # Write annotation
                with open(annotation_output_path, 'w') as annotation_file:
                    annotation_file.write(annotation)

                # Copy image
                import shutil
                shutil.copyfile(image_input_path, image_output_path)