import torch
import torchvision
import PIL
import os
import json
from torchvision.io import read_image
import math
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import piexif
from requests import get
from pandas import json_normalize
import torch.nn as nn
from torchvision import models, transforms
import shapefile
from shapely.geometry import Point
from shapely.geometry import shape
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

def get_region(latitude, longitude):
    gps = (longitude, latitude)
    try:
        shape_path = "World_Continents/World_Continents.shp"
        shp = shapefile.Reader(shape_path)
        all_shapes = shp.shapes()
        all_records = shp.records()
        for i in range(len(all_shapes)):
            boundary = all_shapes[i]
            if Point(gps).within(shape(boundary)):
                region = all_records[i][1]
                print("The image was taken in ", region)
    except:
        print("Couldn't get continent")
    return region

def classify(instance, model_path, device):
    model = create_classification_model(model_path, device)
    scores = model(instance)[0]
    scores = torch.nn.functional.softmax(scores, dim=0).tolist()
    index_to_class = json.load(open(os.path.join(model_path, "index_to_class.txt")))
    named_labels = list(index_to_class.values())
    named_labels_with_scores = {}
    for i in range(len(named_labels)):
        named_labels_with_scores[named_labels[i]] = scores[i]
    return named_labels_with_scores

def prepare_image_for_classification(image, device):
    image = image.convert('RGB')
    # pad crop to square
    width, height = image.size
    max_dim = max(width, height)
    if height > width:
        pad = [int((height - width) / 2) + 20, 20]
    else:
        pad = [20, int((width - height)/2) + 20]
    image = torchvision.transforms.Pad(padding=pad)(image)
    image = torchvision.transforms.CenterCrop(size=max_dim)(image)
    # resize crop to 224
    image = torchvision.transforms.Resize(224)(image)
    image = torchvision.transforms.ToTensor()(image)
    image = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    return image

def create_classification_model(model_path, device):
    index_to_class = json.load(open(os.path.join(model_path, "index_to_class.txt")))
    num_classes = len(index_to_class)
    model = models.resnet101(weights='ResNet101_Weights.DEFAULT')
    features = model.fc.in_features
    model.fc = nn.Linear(features, num_classes)
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, "model_best_state_dict.pth")))
    model.eval()
    return model

def create_detection_model(model_path, device):
    index_to_class = json.load(open(os.path.join(model_path, "index_to_class.txt")))
    num_classes = len(index_to_class) + 1
    kwargs = json.load(open(os.path.join(model_path, "kwargs.txt")))
    backbone = resnet_fpn_backbone(backbone_name = "resnet101", weights = "DEFAULT")
    box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(backbone.out_channels * 4, num_classes)
    model = torchvision.models.detection.__dict__["FasterRCNN"](box_predictor = box_predictor, backbone = backbone, **kwargs)
    model.load_state_dict(torch.load(os.path.join(model_path, "model_best_state_dict.pth"), map_location=device))
    model.eval()
    model = model.to(device)
    return model

def prepare_image_for_detection(image_path, device):
    image = PIL.Image.open(image_path).convert('RGB')
    image = torchvision.transforms.PILToTensor()(image)
    image = torchvision.transforms.ConvertImageDtype(torch.float)(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    return image

def detect_birds(model_path, image_path, model, device):
    index_to_class = json.load(open(os.path.join(model_path, "index_to_class.txt")))
    image = prepare_image_for_detection(image_path, device)
    with torch.no_grad():
        prediction = model(image)
    boxes = prediction["boxes"]
    scores = prediction["scores"]
    labels = prediction["labels"]
    return boxes, scores, labels

def create_annotation(output_dict, file, height, width):
    class_to_label = json.load(open("class_to_label.json"))
    points = []
    labels = []
    for index in range(len(output_dict["boxes"][0])):
        box = output_dict["boxes"][0][index]
        points.append([
            [float(box[0]), float(box[1])],
            [float(box[2]), float(box[3])]])
        if output_dict["age"][index]["adult"] > 0.5:
            rank_scores = output_dict["species"][index]
            if max(output_dict["species"][index].values()) > 0.5:
                label = class_to_label[max(output_dict["species"][index], key=rank_scores.get)]
            else: label = class_to_label["adult"]
        else: label = class_to_label["chick"]
        labels.append(label)
    label_name = os.path.splitext(file)[0] + '.json'
    label_path = os.path.join("../images", label_name)
    shapes = []
    for i in range(0, len(labels)):
        shapes.append({
            "label": labels[i],
            "points": points[i],
            "group_id": 'null',
            "shape_type": "rectangle",
            "flags": {}})
    annotation = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": file,
        "imageData": 'null',
        "imageHeight": height,
        "imageWidth": width}
    annotation_str = json.dumps(annotation, indent = 2).replace('"null"', 'null')
    with open(label_path, 'w') as annotation_file:
        annotation_file.write(annotation_str)
    return

def main(**kwargs):
    # walk through image files and label
    for file in os.listdir(kwargs["datapath"]):
        if file.lower().endswith((".jpg")):
            print("Labelling: ", file)
            # get exif data and calculate gsd and region
            image_path = os.path.join("../images", file)
            image = PIL.Image.open(image_path)
            # try getting gsd from exif comments
            # get exif data
            exif_dict = piexif.load(image.info['exif'])
            exif_bytes = piexif.dump(exif_dict)
            comments = json.loads("".join(map(chr, [i for i in exif_dict["0th"][piexif.ImageIFD.XPComment] if i != 0])))
            gsd = float(comments["gsd"])
            latitude = float(comments["latitude"])
            longitude = float(comments["longitude"])
            region = get_region(latitude, longitude)
            device = torch.device("cuda")
            detection_model_path = "models/temp"
            age_model_path = "../models/temp/Age"
            region_model_path = os.path.join("../models/temp", region)
            detection_model = create_detection_model(detection_model_path, device)
            boxes, bird_score, bird_label = detect_birds(detection_model_path, image_path, detection_model, device)
            output_dict = {"boxes": [], "label": [], "ground_truth": []}
            output_dict["boxes"].append(boxes)


            for i in range(len(bird_label)):
                named_labels_with_scores = {bird_label[i]: bird_score[i], "Background": 1 - bird_score[i]}
                output_dict["bird"].append(named_labels_with_scores)
            # classify age and species
            for i in range(len(boxes)):
                box = boxes[i]
                box_left = box[0].item()
                box_right = box[2].item()
                box_top = box[1].item()
                box_bottom = box[3].item()

                instance = image.crop((box_left, box_top, box_right, box_bottom))
                instance = prepare_image_for_classification(instance, device)
                # create Age model and predict class
                age_classification = classify(instance, age_model_path, device)
                
                output_dict["age"].append(age_classification)                   
                if age_classification["adult"] > 0.5:
                    # bird is an adult
                    species_classification = classify(instance, region_model_path, device)
                else:
                    species_classification = {"unknown": 1.0}
                output_dict["species"].append(species_classification)
            # label image
            create_image(image_path, output_dict)
            # create label file
            create_annotation(output_dict, file, image_width, image_height)

if __name__ == "__main__":
    main()

from coco_utils import get_coco, get_coco_kp
import presets
import utils
from engine import evaluate
def get_dataset(name, image_set, transform, data_path, num_classes):
    paths = {"coco": (data_path, get_coco, num_classes), "coco_kp": (data_path, get_coco_kp, 2)}
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes

def get_transform(train, args):
    if train:
        return presets.DetectionPresetTrain(data_augmentation=args.data_augmentation)
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()
        return lambda img, target: (trans(img), target)
    else:
        return presets.DetectionPresetEval()
    
dataset_test, num_classes = get_dataset(args.dataset, "train", get_transform(False, args), args.data_path, args.numclasses)
train_size = int(0.8 * len(dataset_test))
test_size = len(dataset_test) - train_size
_, dataset_test = torch.utils.data.random_split(dataset_test, [train_size, test_size])
test_sampler = torch.utils.data.SequentialSampler(dataset_test)
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn)
evaluate(model, data_loader_test, device=device)
