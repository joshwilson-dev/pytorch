import torch
import torchvision
import custom_dataloader
import PIL
import os
import importlib
import json
from torchvision.io import read_image
importlib.reload(custom_dataloader)
import csv

def prepare_image(image, goal, device):
    if goal == "classify":
        image = image.convert('RGB')
        image = torchvision.transforms.ToTensor()(image)
        image = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        image = torchvision.transforms.Resize(256)(image)
        image = torchvision.transforms.CenterCrop(224)(image)
    else:
        image = image.convert('RGB')
        image = torchvision.transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    return image

def detect_birds(image_path, model, device):
    image = prepare_image(image_path, "detect", device)
    with torch.no_grad():
        prediction = model(image)
    boxes = prediction[0]["boxes"]
    labels = prediction[0]["labels"].tolist()
    scores = prediction[0]["scores"].tolist()
    return boxes, labels, scores

def create_classifier(device, num_classes, model_path):
    model = torchvision.models.resnet50()
    features = model.fc.in_features
    model.fc = torch.nn.Linear(features, num_classes)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def create_image(image_path, output_dict):
    image = read_image(image_path)
    boxes = output_dict["boxes"][0]
    labels = ["object"] * len(boxes)
    for i in range(len(boxes)):
        for rank in output_dict.keys():
            if rank != "boxes" and rank != "lifestage":
                rank_scores = output_dict[rank][i]
                if max(rank_scores.values()) > 0.5:
                    labels[i] = max(rank_scores, key=rank_scores.get)
    ouput_image = torchvision.utils.draw_bounding_boxes(image = image, boxes = boxes, labels = labels, font = "../model/bird-species-detector/BKANT.TTF", font_size = 30, width=5)
    ouput_image = torchvision.transforms.ToPILImage()(ouput_image)
    ouput_image.save("outputs/trial.jpg")

def main(image_path):
    device = torch.device("cuda")
    # device = torch.device("cpu")
    models = []
    model_paths = []
    index_class_paths = []
    for root, _, _ in os.walk("..\\..\\models\\bird-detector\\object"):
        models.append(os.path.split(root)[1])
        model_paths.append(root + "\\" + "model_final_state_dict.pth")
        index_class_paths.append(root + "\\" + "index_to_class.txt")
    model_dict = {"models": models, "model_paths": model_paths, "index_class_paths": index_class_paths}
    # detect birds
    model = custom_dataloader.FRCNNObjectDetector()
    model_path = model_dict["model_paths"][0]
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model = model.to(device)
    image = PIL.Image.open(image_path)
    boxes, labels, scores = detect_birds(image, model, device)
    inverse_labels = [str(1 - label) for label in labels]
    inverse_scores = [1 - score for score in scores]
    index = model_dict["models"].index("object")
    index_class_path = model_dict["index_class_paths"][index]
    f = open(index_class_path)
    index_to_class = json.load(f)
    labels = [str(i) for i in labels]
    named_labels = [index_to_class[i] for i in labels]
    inverse_named_labels = [index_to_class[i] for i in inverse_labels]
    output_dict = {"boxes": [], "bird": [], "order": [], "family": [], "genus": [], "species": [], "lifestage": []}
    output_dict["boxes"].append(boxes)
    for i in range(len(named_labels)):
        named_labels_with_scores = {named_labels[i]: scores[i], inverse_named_labels[i]: inverse_scores[i]}
        output_dict["bird"].append(named_labels_with_scores)
    for box in boxes:
        model = "aves"
        for rank in output_dict.keys():
            if rank != "boxes" and rank != "bird":
                index = model_dict["models"].index(model)
                model_path = model_dict["model_paths"][index]
                index_class_path = model_dict["index_class_paths"][index]
                f = open(index_class_path)
                index_to_class = json.load(f)
                num_classes = len(index_to_class)
                if num_classes > 1:
                    # Crop image to bounding box
                    box_left = box[0].item()
                    box_right = box[2].item()
                    box_top = box[1].item()
                    box_bottom = box[3].item()
                    instance = image.crop((box_left, box_top, box_right, box_bottom))
                    # create model and predict class
                    model = create_classifier(device, num_classes, model_path)
                    scores = model(prepare_image(instance, "classify", device))[0]
                    scores = torch.nn.functional.softmax(scores, dim=0).tolist()
                else: 
                    scores = [1.0]
                named_labels = list(index_to_class.values())
                named_labels_with_scores = {}
                for i in range(len(named_labels)):
                    named_labels_with_scores[named_labels[i]] = scores[i]
                output_dict[rank].append(named_labels_with_scores)
                model = index_to_class[str(scores.index(max(scores)))]
    create_image(image_path, output_dict)
    with open('outputs/trial.csv', 'w', newline='') as csvfile:
        fieldnames = {"bird": [], "bird score": [], "order": [], "order score": [], "family": [], "family score": [], "genus": [], "genus score": [], "species": [], "species score": [], "lifestage": [], "lifestage score": []}
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(boxes)):
            csv_dict = {}
            for column in fieldnames:
                if "score" in column:
                    key = column[:-len(" score")]
                    csv_dict[column] = max(output_dict[key][i].values())
                else:
                    csv_dict[column] = max(output_dict[column][i], key=output_dict[column][i].get)
            writer.writerow(csv_dict)

if __name__ == "__main__":
    main(image_path = "../../datasets/bird-detector/test/test-5.jpg")