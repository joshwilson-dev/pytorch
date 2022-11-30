import torch
import torchvision
import PIL
import os
import json
import math
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import piexif
import torch.nn as nn
from torchvision import models, transforms
import shapefile
from shapely.geometry import Point
from shapely.geometry import shape
from torchvision.models.detection.anchor_utils import AnchorGenerator
import csv
import math
import pandas as pd

def get_xmp(image_file_path):
    # get xmp information
    f = open(image_file_path, 'rb')
    d = f.read()
    xmp_start = d.find(b'<x:xmpmeta')
    xmp_end = d.find(b'</x:xmpmeta')
    xmp_str = (d[xmp_start:xmp_end+12]).lower()
    # define info to search for
    dji_xmp_keys = ['relativealtitude']
    dji_xmp = {}
    # extract info from xmp
    for key in dji_xmp_keys:
        search_str = (key + '="').encode("UTF-8")
        value_start = xmp_str.find(search_str) + len(search_str)
        value_end = xmp_str.find(b'"', value_start)
        value = xmp_str[value_start:value_end]
        dji_xmp[key] = float(value.decode('UTF-8'))
    height = dji_xmp["relativealtitude"]
    return height

def get_gsd(exif, sensor_sizes, image_width, image_height, height):
    camera_model = exif["0th"][piexif.ImageIFD.Model].decode("utf-8").rstrip('\x00')
    sensor_width, sensor_length = sensor_sizes[camera_model]
    focal_length = exif["Exif"][piexif.ExifIFD.FocalLength][0] / exif["Exif"][piexif.ExifIFD.FocalLength][1]
    pixel_pitch = max(sensor_width / image_width, sensor_length / image_height)
    # calculate gsd
    gsd = height * pixel_pitch / focal_length
    return gsd

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

def is_float(string):
    try:
        string = float(string)
        return string
    except: return string

def get_region(exif_dict):
    try:
        latitude, longitude = get_gps(exif_dict)
        gps = (longitude, latitude)
    except:
        print("Couldn't get GPS")
        return
    try:
        shape_path = "world_continents\World_Continents.shp"
        shp = shapefile.Reader(shape_path)
        all_shapes = shp.shapes()
        all_records = shp.records()
        for i in range(len(all_shapes)):
            boundary = all_shapes[i]
            if Point(gps).within(shape(boundary)):
                region = all_records[i][1]
                if region == "Australia": region = "Oceania"
                if region == "Africa": region = "Oceania"
    except Exception as e:
        print(e)
        print("Couldn't get continent")
    return region

def create_detection_model(det_index_to_class):
    num_classes = len(det_index_to_class) + 1
    backbone = resnet_fpn_backbone(backbone_name = "resnet101", weights = "DEFAULT")
    box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(backbone.out_channels * 4, num_classes)
    return box_predictor, backbone

def update_detection_model(model_path, device, box_predictor, backbone, kwargs, anchor_sizes):
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    kwargs["rpn_anchor_generator"] = rpn_anchor_generator
    model = torchvision.models.detection.__dict__["FasterRCNN"](box_predictor = box_predictor, backbone = backbone, **kwargs)
    model.load_state_dict(torch.load(os.path.join(model_path, "model_best_state_dict-artificial-new.pth"), map_location=device))
    model.eval()
    model = model.to(device)
    return model

def prepare_image_for_detection(image_path, device, overlap, patch_width, patch_height):
    original_image = PIL.Image.open(image_path).convert('RGB')
    width, height = original_image.size
    n_crops_width = math.ceil((width - overlap) / (patch_width - overlap))
    n_crops_height = math.ceil((height - overlap) / (patch_height - overlap))
    padded_width = n_crops_width * (patch_width - overlap) + overlap
    padded_height = n_crops_height * (patch_height - overlap) + overlap
    pad_width = (padded_width - width) / 2
    pad_height = (padded_height - height) / 2
    left = (padded_width/2) - (width/2)
    top = (padded_height/2) - (height/2)
    image = PIL.Image.new(original_image.mode, (padded_width, padded_height), "black")
    image.paste(original_image, (int(left), int(top)))
    patches = []
    for height_index in range(n_crops_height):
        for width_index in range(n_crops_width):
            left = width_index * (patch_width - overlap)
            right = left + patch_width
            top = height_index * (patch_height - overlap)
            bottom = top + patch_height
            patch = image.crop((left, top, right, bottom))
            patches.append(patch)
    batch = torch.empty(0, 3, patch_height, patch_width).to(device)
    for patch in patches:
        patch = transforms.PILToTensor()(patch)
        patch = transforms.ConvertImageDtype(torch.float)(patch)
        patch = patch.unsqueeze(0)
        patch = patch.to(device)
        batch = torch.cat((batch, patch), 0)
    return batch, pad_width, pad_height, n_crops_height, n_crops_width

def detect_birds(model_path, image_path, model, device, det_index_to_class, overlap, patch_width, patch_height, reject):
    kwargs = json.load(open(os.path.join(model_path, "kwargs.txt")))
    print("Patching...")
    batch, pad_width, pad_height, n_crops_height, n_crops_width = prepare_image_for_detection(image_path, device, overlap, patch_width, patch_height)
    max_batch_size = 4
    batch_length = batch.size()[0]
    sub_batch_lengths = [max_batch_size] * math.floor(batch_length/max_batch_size)
    sub_batch_lengths.append(batch_length % max_batch_size)
    sub_batches = torch.split(batch, sub_batch_lengths)
    predictions = []
    print("Detecting...")
    with torch.no_grad():
        for sub_batch in sub_batches:
            prediction = model(sub_batch)
            predictions.extend(prediction)
    boxes = torch.empty(0, 4)
    scores = torch.empty(0)
    labels = torch.empty(0, dtype=torch.int64)
    for height_index in range(n_crops_height):
        for width_index in range(n_crops_width):
            patch_index = height_index * n_crops_width + width_index
            batch_boxes = predictions[patch_index]["boxes"]
            batch_scores = predictions[patch_index]["scores"]
            batch_labels = predictions[patch_index]["labels"]
            # if box near overlapping edge, drop the entire box
            sides = []
            if width_index != 0:
                sides.append(0)
            if height_index != 0:
                sides.append(1)
            if width_index != max(range(n_crops_width)):
                sides.append(2)
            if height_index != max(range(n_crops_height)):
                sides.append(3)
            for side in sides:
                # top and left
                if side < 2: index = batch_boxes[:, side] > reject
                # bottom and right
                else: index = batch_boxes[:, side] < patch_width - reject
                batch_boxes = batch_boxes[index]
                batch_scores = batch_scores[index]
                batch_labels = batch_labels[index]
            padding_left = (patch_width - overlap) * width_index - pad_width
            padding_top = (patch_height - overlap) * height_index - pad_height
            adjustment = torch.tensor([[padding_left, padding_top, padding_left, padding_top]])
            adj_boxes = torch.add(adjustment, batch_boxes.to(torch.device("cpu")))
            boxes = torch.cat((boxes, adj_boxes), 0)
            scores = torch.cat((scores, batch_scores.to(torch.device("cpu"))), 0)
            labels = torch.cat((labels, batch_labels.to(torch.device("cpu"))), 0)
    nms_indices = torchvision.ops.nms(boxes, scores, kwargs["box_nms_thresh"])
    boxes = boxes[nms_indices]
    scores = scores[nms_indices].tolist()
    labels = labels[nms_indices].tolist()
    named_labels = [det_index_to_class[str(i)] for i in labels]
    return boxes, scores, named_labels

def create_classification_model():
    model = models.resnet101(weights='ResNet101_Weights.DEFAULT')
    features = model.fc.in_features
    return model, features

def update_classification_model(model_base, features, device, index_to_class, model_path):
    num_classes = len(index_to_class)
    model_base.fc = nn.Linear(features, num_classes)
    model = model_base.to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, "model_best_state_dict.pth")))
    model.eval()
    return model

def prepare_image_for_classification(image, device):
    image = image.convert('RGB')
    # pad crop to square
    width, height = image.size
    max_dim = max(width, height)
    if height > width:
        pad = [int((height - width) / 2) + 20, 20]
    else:
        pad = [20, int((width - height)/2) + 20]
    image = transforms.Pad(padding=pad)(image)
    image = transforms.CenterCrop(size=max_dim)(image)
    # resize crop to 224
    image = transforms.Resize(224)(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    return image

def classify(batch, model, index_to_class, output_dict):
    print("Classifying...")
    scores = model(batch)
    named_labels = list(index_to_class.values())
    for instance_index in range(len(batch)):
        named_labels_with_scores = {}
        score = torch.nn.functional.softmax(scores[instance_index], dim=0).tolist()
        for class_index in range(len(named_labels)):
            named_labels_with_scores[named_labels[class_index]] = score[class_index]
        output_dict["species"].append(named_labels_with_scores)
    return output_dict

def create_annotation(output_dict, image_name, width, height):
    print("Creating annotation...")
    points = []
    labels = []
    scores = []
    for index in range(len(output_dict["boxes"][0])):
        box = output_dict["boxes"][0][index]
        points.append([
            [float(box[0]), float(box[1])],
            [float(box[2]), float(box[3])]])
        bird_score = round(output_dict["bird"][index]['Bird'], 2)
        species_scores = output_dict["species"][index]
        species_score = round(max(species_scores.values()), 2)
        label = max(species_scores, key=species_scores.get).split("_")[0]
        labels.append(label)
        scores.append([bird_score, species_score])
    label_name = os.path.splitext(image_name)[0] + '.json'
    label_path = os.path.join("../images", label_name)
    shapes = []
    for i in range(0, len(labels)):
        shapes.append({
            "label": "Bird: " + str(scores[i][0]) + " - " + labels[i] + ": " + str(scores[i][1]),
            "points": points[i],
            "group_id": 'null',
            "shape_type": "rectangle",
            "flags": {}})
    annotation = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": 'null',
        "imageHeight": height,
        "imageWidth": width}
    annotation_str = json.dumps(annotation, indent = 2).replace('"null"', 'null')
    with open(label_path, 'w') as annotation_file:
        annotation_file.write(annotation_str)
    return

def create_csv(output_dict, image_name, header):
    print("Updating csv...")
    csv_path = "../images/results.csv"
    with open(csv_path, 'a+', newline='') as csvfile:
        bird = list(output_dict['bird'][0].keys())
        species = list(output_dict['species'][0].keys())
        fieldnames = ["image_name", "box"] + bird + species
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for index in range(len(output_dict["boxes"][0])):
            if header == False:
                writer.writeheader()
                header = True
            row = {"image_name": image_name, "box": output_dict["boxes"][0][index]}
            for fieldname in bird:
                row[fieldname] = output_dict["bird"][index][fieldname]
            for fieldname in species:
                row[fieldname] = output_dict["species"][index][fieldname]
            writer.writerow(row)

def main():
    sensor_sizes = {
        "FC220": [6.16, 4.55],
        "FC330": [6.16, 4.62],
        "FC7203": [6.3, 4.7],
        "FC6520": [17.3, 13],
        "FC6310": [13.2, 8.8],
        "L1D-20c": [13.2, 8.8]
        }

    # walk through image files and store in dataframe
    images = pd.DataFrame(columns = ["image_path", "gsd", "region"])
    for file in os.listdir("../images"):
        if file.lower().endswith((".jpg", ".jpeg")):
            print("Adding record for: ", file)
            # get exif data and determine region
            image_path = os.path.join("../images", file)
            image = PIL.Image.open(image_path)
            image_width, image_height = image.size
            # load image exif data
            exif_dict = piexif.load(image.info['exif'])
            # try calculating gsd
            print("Trying to determine gsd from image metadata")
            try:
                height = get_xmp(image_path)
                gsd = get_gsd(exif_dict, sensor_sizes, image_width, image_height, height)
                print("GSD: ", gsd)
            except:
                gsd = 0.005
                print("Couldn't determine gsd, so using 0.005 gsd to filter anchors")
            # try determining region from image metadata
            print("Trying to determine region from image metadata")
            try:
                region = get_region(exif_dict)
                print("Region: ", region)
            except:
                region = "Global"
                print("Couldn't determine region, so using global model")
            images = pd.concat([images, pd.DataFrame({"image_path": [image_path], "gsd": [gsd], "region": [region]})])
    # convert dict of lists to dataframe
    # sort by gsd, then by region
    images = images.sort_values(['gsd', 'region']).reset_index()
    
    # define constants
    overlap = 150
    patch_height = 800
    patch_width = 800
    reject = 25

    header = False

    base_anchor_sizes_px = ((32,), (64,), (128,), (256,), (512,))
    anchor_sizes_px = base_anchor_sizes_px
    detector_model_path = os.path.join("../models/full-model/bird-detector")
    det_kwargs = json.load(open(os.path.join(detector_model_path, "kwargs.txt")))
    min_anchor_m = 0.2
    max_anchor_m = 1.3

    # create base detection model
    device = torch.device("cuda")
    # device = torch.device("cpu")
    det_index_to_class = json.load(open(os.path.join(detector_model_path, "index_to_class.json")))
    box_predictor, backbone = create_detection_model(det_index_to_class)
    detection_model = update_detection_model(detector_model_path, device, box_predictor, backbone, det_kwargs, anchor_sizes_px)
    # create base classification model
    clas_model_base, class_features = create_classification_model()
    region = "Oceania"
    clas_model_path = os.path.join("../models/full-model/bird-classifier", region)
    clas_index_to_class = json.load(open(os.path.join(clas_model_path, "index_to_class.json")))
    clas_model = update_classification_model(clas_model_base, class_features, device, clas_index_to_class, clas_model_path)

    for _, row in images.iterrows():
        gsd = row["gsd"]
        temp_region = row["region"]
        image_path = row["image_path"]
        image_name = os.path.basename(row["image_path"])
        image = PIL.Image.open(image_path)
        image_width, image_height = image.size
        print("Annotating: ", image_name)
        # set up results
        output_dict = {"boxes": [], "bird": [], "species": []}
        # update detection model if new anchors
        temp_anchor_sizes_px = []
        for px in base_anchor_sizes_px:
            anchor_m = px[0] * gsd
            if anchor_m >= min_anchor_m and anchor_m <= max_anchor_m:
                temp_anchor_sizes_px.append([px[0],])
            else:
                temp_anchor_sizes_px.append([0,])
        tp_anchor_sizes_px = tuple(tuple(sub) for sub in temp_anchor_sizes_px)
        if anchor_sizes_px != tp_anchor_sizes_px:
            print("Updating detection model with new gsd bracket")
            anchor_sizes_px = tp_anchor_sizes_px
            detection_model = update_detection_model(detector_model_path, device, box_predictor, backbone, det_kwargs, anchor_sizes_px)
        print(anchor_sizes_px)
        # detect birds
        boxes, bird_score, bird_label = detect_birds(detector_model_path, image_path, detection_model, device, det_index_to_class, overlap, patch_width, patch_height, reject)
        output_dict["boxes"].append(boxes)
        for i in range(len(bird_label)):
            named_labels_with_scores = {bird_label[i]: bird_score[i], "Background": 1 - bird_score[i]}
            output_dict["bird"].append(named_labels_with_scores)
        # classify species
        if region != temp_region:
            print("Updating classifier with new region")
            region = temp_region
            clas_model_path = os.path.join("../models/full-model/bird-classifier", region)
            clas_index_to_class = json.load(open(os.path.join(clas_model_path, "index_to_class.json")))
            clas_model = update_classification_model(clas_model_base, class_features, device, clas_index_to_class, clas_model_path)
        batch = torch.empty(0, 3, 224, 224).to(device)
        for i in range(len(boxes)):
            # crop out the detected bird and pass to classifier
            box = boxes[i]
            box_left = box[0].item()
            box_right = box[2].item()
            box_top = box[1].item()
            box_bottom = box[3].item()
            instance = image.crop((box_left, box_top, box_right, box_bottom))
            # prepare the crop for classification
            instance = prepare_image_for_classification(instance, device)
            # add to batch
            batch = torch.cat((batch, instance), 0)
        # classify the species of the crops
        output_dict = classify(batch, clas_model, clas_index_to_class, output_dict)
        # create label file
        create_annotation(output_dict, image_name, image_width, image_height)
        if len(output_dict["bird"]) > 0:
            # create dictionary of results
            create_csv(output_dict, image_name, header)
        header = True
    print("Done!")

if __name__ == "__main__":
    main()