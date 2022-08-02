import torch
import torchvision
import custom_dataloader
import PIL
import numpy as np
from matplotlib import pyplot as plt
import os
import importlib
import utils
import json
importlib.reload(custom_dataloader)

def prepare_image(image_path, goal, device):
    image = PIL.Image.open(image_path).convert('RGB')
    image = torchvision.transforms.ToTensor()(image)
    if goal == "classify":
        image = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        image = torchvision.transforms.Resize(256)(image)
        image = torchvision.transforms.CenterCrop(224)(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    return image

def detect_birds(image_path, model, device):
    image = prepare_image(image_path, device)
    with torch.no_grad():
        prediction = model(image)
    image = (image*255).type(torch.uint8)[0]
    boxes = prediction[0]["boxes"]
    labels = prediction[0]["labels"].tolist()
    scores = prediction[0]["scores"].tolist()
    return {"boxes": boxes, "labels": labels, "scores": scores}

def create_classifier(device, num_classes, model_path):
    model = torchvision.models.resnet18()
    features = model.fc.in_features
    model.fc = torch.nn.Linear(features, num_classes)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

def main():
    device = torch.device("cuda")
    # device = torch.device("cpu")
    models = []
    model_paths = []
    class_index_paths = []
    for root, dirs, files in os.walk("..\\..\\models\\bird-detector"):
        for file in files:
            if file == "model_final_state_dict.pth":
                models.append(os.path.split(root)[1])
                model_paths.append(root + "\\" + file)
                class_index_paths.append(root + "\\" + "index_to_class.txt") 
    model_dict = {"models": models, "model_paths": model_paths, "class_index_paths": class_index_paths}
    # detect birds
    model = custom_dataloader.FRCNNObjectDetector()
    model_path = model_dict[model_paths][0]
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model = model.to(device)
    image = "../../datasets/bird-detector/train/0a2c34e5cecebc5c5c1cd3e6eb08f243.jpg"
    predictions = detect_birds(image, model, device)
    for box in predictions["boxes"]:
        best_score = 1
        model = "aves"
        while best_score > 0.5:
            index = model_dict["models"].index("aves")
            model_path = model_dict["model_path"][index]
            class_index_path = model_dict["class_index_paths"][index]
            f = open(class_index_path)
            class_to_index = json.load(f)
            #HERE
            model = create_classifier()






if __name__ == "__main__":
    main()
    #             models.append()
    # models = {"model": [], "model_path": [], "index_to_class": []}
    # model = custom_dataloader.FRCNNObjectDetector()

    # model_path = "../../models/bird-detector/bird-detector/model_final_state_dict.pth"
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # model.eval()
    # model = model.to(device)
    # prediction_1 = draw_boxes("../../datasets/bird-detector/train/0a2c34e5cecebc5c5c1cd3e6eb08f243.jpg", model)
    # prediction_1