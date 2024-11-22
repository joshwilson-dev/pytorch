import os
import json
from PIL import Image

# root = "C:/Users/uqjwil54/OneDrive - The University of Queensland/DBBS/Surveys/2024-03-12_1546_Adams Beach/T4"
root = "C:/Users/uqjwil54/OneDrive - The University of Queensland/Events/DEMO/Seven Mile Lagoon/SML-2024_10_24/Subset/background"
os.chdir(root)
# iterate through files in dir
for root, dirs, files in os.walk(os.getcwd()):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            annotation_file = os.path.splitext(file)[0] + ".json"
            annotation_file_path = os.path.join(root, annotation_file)
            if os.path.exists(annotation_file_path):
                continue
            else:
                print("Adding empty annotation to: ", file)
                image_file_path = os.path.join(root, file)
                image = Image.open(image_file_path)
                width, height = image.size 
                annotation = {
                    "version": "5.0.1",
                    "flags": {},
                    "shapes": [],
                    "imagePath": file,
                    "imageData": "null",
                    "imageHeight": height,
                    "imageWidth": width}

                annotation = json.dumps(annotation, indent = 2).replace('"null"', 'null')
                with open(annotation_file_path, 'w') as annotation_file:
                    annotation_file.write(annotation)