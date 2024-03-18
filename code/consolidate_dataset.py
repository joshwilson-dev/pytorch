import os
import shutil
import json
from PIL import Image
import hashlib
import piexif

index_to_class = json.load(open("resources/index_to_class.json"))
class_to_index = {value["name"]: key for key, value in index_to_class.items()}

root = "data/original"
os.chdir(root)

if os.path.exists("input"):
    shutil.rmtree("input")
os.makedirs("input")

# iterate through files in dir
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

                # Calculate md5
                image = Image.open(image_input_path)
                md5 = hashlib.md5(image.tobytes()).hexdigest()

                # Output paths
                annotation_output_name = md5 + '.json'
                annotation_output_path = os.path.join("input", annotation_output_name)
                image_output_name = md5 + '.jpg'
                image_output_path = os.path.join("input", image_output_name)

                # Update annotation
                annotation["imagePath"] = image_output_name
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

                # Write original image path to exif comments
                exif = piexif.load(image.info['exif'])
                exif["0th"][piexif.ImageIFD.XPComment] = json.dumps(image_input_name).encode('utf-16le')
                exif = piexif.dump(exif)

                # Save image
                image.save(image_output_path, exif = exif)