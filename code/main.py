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

patch_size = 1333
overlap = 150

def get_xmp(image_path):
    # get xmp information
    f = open(image_path, 'rb')
    d = f.read()
    xmp_start = d.find(b'<x:xmpmeta')
    xmp_end = d.find(b'</x:xmpmeta')
    xmp_str = (d[xmp_start:xmp_end+12]).lower()
    # Extract dji info
    dji_xmp_keys = ['relativealtitude']
    dji_xmp = {}
    for key in dji_xmp_keys:
        search_str = (key + '="').encode("UTF-8")
        value_start = xmp_str.find(search_str) + len(search_str)
        value_end = xmp_str.find(b'"', value_start)
        value = xmp_str[value_start:value_end]
        dji_xmp[key] = float(value.decode('UTF-8'))
    height = dji_xmp["relativealtitude"]
    return height

def get_altitude(exif_dict):
    altitude_tag = exif_dict['GPS'][piexif.GPSIFD.GPSAltitude]
    altitude_ref = exif_dict['GPS'][piexif.GPSIFD.GPSAltitudeRef]
    altitude = altitude_tag[0]/altitude_tag[1]
    below_sea_level = altitude_ref != 0
    altitude = -altitude if below_sea_level else altitude
    return altitude

def get_gps(exif_dict):
    latitude_tag = exif_dict['GPS'][piexif.GPSIFD.GPSLatitude]
    longitude_tag = exif_dict['GPS'][piexif.GPSIFD.GPSLongitude]
    latitude_ref = exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef].decode("utf-8")
    longitude_ref = exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef].decode("utf-8")
    latitude = degrees(latitude_tag)
    longitude = degrees(longitude_tag)
    latitude = -latitude if latitude_ref == 'S' else latitude
    longitude = -longitude if longitude_ref == 'W' else longitude
    return latitude, longitude

def degrees(tag):
    d = tag[0][0] / tag[0][1]
    m = tag[1][0] / tag[1][1]
    s = tag[2][0] / tag[2][1]
    return d + (m / 60.0) + (s / 3600.0)

def get_elevation(latitude, longitude):
    query = ('https://api.open-elevation.com/api/v1/lookup'f'?locations={latitude},{longitude}')
    # Request with a timeout for slow responses
    r = get(query, timeout = 20)
    # Only get the json response in case of 200 or 201
    if r.status_code == 200 or r.status_code == 201:
        elevation = json_normalize(r.json(), 'results')['elevation'].values[0]
    else: 
        elevation = None
    return elevation

def is_float(string):
    try:
        string = float(string)
        return string
    except: return string

sensor_size = {
    "FC220": [6.16, 4.55],
    "FC330": [6.16, 4.62],
    "FC7203": [6.3, 4.7],
    "FC6520": [17.3, 13],
    "FC6310": [13.2, 8.8],
    "L1D-20c": [13.2, 8.8],
    "Canon PowerShot G15": [7.44, 5.58],
    "NX500": [23.50, 15.70],
    "Canon PowerShot S100": [7.44, 5.58],
    "Survey2_RGB": [6.17472, 4.63104]
    }

def get_gsd(exif_dict, image_path, image_width, image_height):
    try:
        height = get_xmp(image_path)
        print("Got height from xmp")
    except:
        print("Couldn't get height from xmp")
        print("Trying to infer height from altitude and elevation a GPS")
        try:
            latitude, longitude = get_gps(exif_dict)
            print("Got GPS from exif")
        except:
            print("Couldn't get GPS")
            return
        try:
            altitude = get_altitude(exif_dict)
            print("Got altiude from exif")
        except:
            print("Couldn't get altitude")
            return
        try:
            elevation = get_elevation(latitude, longitude)
            print("Got elevation from exif")
            height = altitude - elevation
        except:
            print("Couldn't get elevation")
            return
    try:
        camera_model = exif_dict["0th"][piexif.ImageIFD.Model].decode("utf-8").rstrip('\x00')
        print("Got camera model from exif")
    except:
        print("Couldn't get camera model from exif")
        return
    try:
        sensor_width, sensor_length = sensor_size[camera_model]
        print("Got sensor dimensions")
    except:
        print("Couldn't get sensor dimensions from sensor size dict")
        return
    try:
        focal_length = exif_dict["Exif"][piexif.ExifIFD.FocalLength][0] / exif_dict["Exif"][piexif.ExifIFD.FocalLength][1]
        print("Got focal length from exif")
    except:
        print("Couldn't get focal length from exif")
        return
    pixel_pitch = max(sensor_width / image_width, sensor_length / image_height)
    gsd = height * pixel_pitch / focal_length
    print("GSD: ", gsd)
    return gsd

def get_region(exif_dict):
    try:
        latitude, longitude = get_gps(exif_dict)
        gps = (longitude, latitude)
        print("Got GPS from exif")
    except:
        print("Couldn't get GPS")
        return
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
    original_image = PIL.Image.open(image_path).convert('RGB')
    width, height = original_image.size
    n_crops_width = math.ceil((width - overlap) / (patch_size - overlap))
    n_crops_height = math.ceil((height - overlap) / (patch_size - overlap))
    padded_width = n_crops_width * (patch_size - overlap) + overlap
    padded_height = n_crops_height * (patch_size - overlap) + overlap
    pad_width = (padded_width - width) / 2
    pad_height = (padded_height - height) / 2
    left = (padded_width/2) - (width/2)
    top = (padded_height/2) - (height/2)
    image = PIL.Image.new(original_image.mode, (padded_width, padded_height), "black")
    image.paste(original_image, (int(left), int(top)))
    patches = [] 
    for height_index in range(n_crops_height):
        for width_index in range(n_crops_width):
            left = width_index * (patch_size - overlap)
            right = left + patch_size
            top = height_index * (patch_size - overlap)
            bottom = top + patch_size
            patch = image.crop((left, top, right, bottom))
            patches.append(patch)
    batch = torch.empty(0, 3, patch_size, patch_size).to(device)
    for patch in patches:
        patch = torchvision.transforms.PILToTensor()(patch)
        patch = torchvision.transforms.ConvertImageDtype(torch.float)(patch)
        patch = patch.unsqueeze(0)
        patch = patch.to(device)
        batch = torch.cat((batch, patch), 0)
    return batch, pad_width, pad_height, n_crops_height, n_crops_width

def detect_birds(model_path, image_path, model, device):
    index_to_class = json.load(open(os.path.join(model_path, "index_to_class.txt")))
    batch, pad_width, pad_height, n_crops_height, n_crops_width = prepare_image_for_detection(image_path, device)
    with torch.no_grad():
        prediction = model(batch)
    boxes = torch.empty(0, 4).to(device)
    scores = torch.empty(0).to(device)
    labels = torch.empty(0, dtype=torch.int64).to(device)
    for height_index in range(n_crops_height):
        for width_index in range(n_crops_width):
            patch_index = height_index * n_crops_width + width_index
            padding_left = (patch_size - overlap) * width_index - pad_width
            padding_top = (patch_size - overlap) * height_index - pad_height
            adjustment = torch.tensor([[padding_left, padding_top, padding_left, padding_top]]).to(device)
            adj_boxes = torch.add(adjustment, prediction[patch_index]["boxes"])
            boxes = torch.cat((boxes, adj_boxes), 0)
            scores = torch.cat((scores, prediction[patch_index]["scores"]), 0)
            labels = torch.cat((labels, prediction[patch_index]["labels"]), 0)
    nms_indices = torchvision.ops.nms(boxes, scores, 0.1)
    boxes = boxes[nms_indices]
    scores = scores[nms_indices].tolist()
    labels = labels[nms_indices].tolist()
    named_labels = [index_to_class[str(i)] for i in labels]
    return boxes, scores, named_labels

def create_image(image_path, output_dict):
    image = read_image(image_path)
    boxes = output_dict["boxes"][0]
    labels = ["object"] * len(boxes)
    for i in range(len(boxes)):
        for rank in output_dict.keys():
            if rank != "boxes":
                rank_scores = output_dict[rank][i]
                if max(rank_scores.values()) > 0.5:
                    labels[i] = max(rank_scores, key=rank_scores.get)
    ouput_image = torchvision.utils.draw_bounding_boxes(image = image, boxes = boxes, labels = labels)
    ouput_image = torchvision.transforms.ToPILImage()(ouput_image)
    ouput_image.show()
    ouput_image.save("trial.jpg")

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

def main():
    # walk through image files and label
    for file in os.listdir("../images"):
        if file.lower().endswith((".jpg", ".jpeg")):
            print("Labelling: ", file)
            # get exif data and calculate gsd and region
            image_path = os.path.join("../images", file)
            image = PIL.Image.open(image_path)
            image_width, image_height = image.size
            # try getting gsd and region from image_metadata.txt file
            image_metadata = json.load(open("image_metadata.json"))
            gsd = image_metadata["gsd"]
            region = image_metadata["region"]
            if region != "unknown":
                print("Using region from image_metadata.json")
            else:
                print("Couldn't get region from image_metadata.json")
                print("Trying to get region from image metadata")
                # load image exif data
                exif_dict = piexif.load(image.info['exif'])
                try:
                    region = get_region(exif_dict)
                    print("Region: ", region)
                except:
                    print("Couldn't determine region, so using all region model")
            if gsd != "unknown":
                print("Using gsd from image_metadata.json")
            else:
                print("Couldn't get gsd from image_metadata.json")
                print("Trying to get gsd from image metadata")
                # load image exif data
                exif_dict = piexif.load(image.info['exif'])
                try:
                    gsd = get_gsd(exif_dict, image_path, image_width, image_height)
                except:
                    gsd = 100
                    print("Couldn't determine gsd, so using all gsd model")
            if gsd <= 0.015:
                gsd_dir = "fine"
            else: gsd_dir = "all"
            # detect birds
            device = torch.device("cpu")
            detection_model_path = os.path.join("../models", gsd_dir)
            age_model_path = os.path.join("../models", gsd_dir, "Age")
            region_model_path = os.path.join("../models", gsd_dir, region)
            detection_model = create_detection_model(detection_model_path, device)
            boxes, bird_score, bird_label = detect_birds(detection_model_path, image_path, detection_model, device)
            output_dict = {"boxes": [], "bird": [], "age": [], "species": []}
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