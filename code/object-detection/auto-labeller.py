################
#### Header ####
################

# Title: Automated Labeller
# Author: Josh Wilson
# Date: 07-07-2022
# Description: 
# This script uses an onnx detection model to create an
# annotation file images in a directory

###############
#### Setup ####
###############
import json
import os
import custom_detector
import numpy
import torch
import torchvision
import PIL


# required packages
# - pytorch cpu
#################
#### Content ####
#################

# model
os.chdir(os.path.dirname(os.path.abspath(__file__))) # delete this for HPC runtime
input_dir = "../inputs"
model_path = os.path.join(input_dir, "model_final_state_dict.pth")
index_path = os.path.join(input_dir, "index_to_class.txt")
model = custom_detector.CustomDetector()
device = torch.device("cuda")
model.load_state_dict(torch.load(model_path, map_location=device))
index_to_class = json.load(open(index_path))
model.eval()
model = model.to(device)


def prepare_image(image_path):
    image = PIL.Image.open(image_path).convert('RGB')
    image = torchvision.transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    width = image.width
    height = image.height
    return image, width, height

def label(image_path, image_name):
    image, width, height = prepare_image(os.path.join(image_path, image_name))
    with torch.no_grad():
        prediction = model(image)
    score_threshold = 0.1
    mask_threshold = 0.1
    boxes = prediction[0]["boxes"]
    labels = prediction[0]["labels"]
    masks = prediction[0]['masks']
    print(masks)
    # loop through each rw of image and keep first and last pixel position

    
    # points = []
    # for box in boxes:
    #     points.append([
    #         [float(box[0]), float(box[1])],
    #         [float(box[2]), float(box[3])]])
    # named_labels = [index_to_class[str(i)] for i in labels]
    # label_name = os.path.splitext(image_name)[0] + '.json'
    # label_path = os.path.join("../outputs", label_name)
    # shapes = []
    # for i in range(0, len(named_labels)):
    #     shapes.append({
    #         "label": named_labels[i],
    #         "points": points[i],
    #         "group_id": 'null',
    #         "shape_type": "rectangle",
    #         "flags": {}})
    # annotation = {
    #     "version": "5.0.1",
    #     "flags": {},
    #     "shapes": shapes,
    #     "imagePath": image_name,
    #     "imageData": 'null',
    #     "imageHeight": height,
    #     "imageWidth": width}
    # annotation_str = json.dumps(annotation, indent = 2).replace('"null"', 'null')
    # with open(label_path, 'w') as annotation_file:
    #     annotation_file.write(annotation_str)

def main():
    # walk through image files and label
    for file in os.listdir(input_dir):
        if file.endswith(".JPG"):
            label(input_dir, file)

main()