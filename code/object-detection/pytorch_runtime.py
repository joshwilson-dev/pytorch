import torch
import torchvision
import custom_dataloader
import PIL
import numpy as np
from matplotlib import pyplot as plt
import os
import importlib
importlib.reload(custom_dataloader)
import json
from torchvision.io import read_image

def prepare_image(image_path):
    image = PIL.Image.open(image_path).convert('RGB')
    image = torchvision.transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    return image

def draw_boxes(image_path, model):
    image = prepare_image(image_path)
    with torch.no_grad():
        prediction = model(image)
    image = read_image(image_path)
    image_boxes = prediction[0]["boxes"]
    labels = prediction[0]["labels"].tolist()
    scores = prediction[0]["scores"].tolist()
    string_scores = ['{0:.2f}'.format(score) for score in scores]
    named_labels = [index_to_class[str(i)] for i in labels]
    # if score less than threshold then just say bird
    for i in range(len(scores)):
        if scores[i] < 0.9:
            named_labels[i] = "Bird"
    named_labels_with_scores = [named_labels[i] + ": " + string_scores[i] for i in range(len(scores))]
    ouput_image = torchvision.utils.draw_bounding_boxes(image = image, boxes = image_boxes, labels = named_labels_with_scores)
    ouput_image = torchvision.transforms.ToPILImage()(ouput_image)
    ouput_image.show()

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# device = torch.device("cuda")
device = torch.device("cpu")
model = custom_dataloader.FRCNNObjectDetector()
model_root = "../../models/bird-object-detection"
model_path = model_root + "/model_final_state_dict.pth"
index_path = model_root + "/index_to_class.txt"
model.load_state_dict(torch.load(model_path, map_location=device))
index_to_class = json.load(open(index_path))
model.eval()
model = model.to(device)
prediction_1 = draw_boxes("../../datasets/bird-detector/test/002mm_2018-01-23_18-14-41_deception_chinstrap_penguins.jpg", model)
prediction_1

# for converting checkpoint
# model_path = "../../models/temp/model_2.pth"
# checkpoint = torch.load(model_path, map_location="cpu")
# model.load_state_dict(checkpoint["model"])
# utils.save_on_master(model.state_dict(), "../../models/temp/model_final_state_dict.pth")