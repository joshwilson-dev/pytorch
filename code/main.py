import torch
import torchvision
import PIL
import os
import json
import math
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import piexif
import shapefile
from shapely.geometry import Point
from shapely.geometry import shape
from torchvision.models.detection.anchor_utils import AnchorGenerator
import csv
import math
import pandas as pd
from torchvision.models.detection import roi_heads
import custom_roi_heads
import custom_boxes
from torchvision.ops import boxes as box_ops
import time

# redefining roi_heads to return scores for all classes
# and filter the classes based on a supplied filter
roi_heads.RoIHeads.postprocess_detections = custom_roi_heads.postprocess_detections
roi_heads.RoIHeads.forward = custom_roi_heads.forward

# redefining boxes to do nms on all classes, not class specific
box_ops.batched_nms = custom_boxes.batched_nms
box_ops._batched_nms_coordinate_trick = custom_boxes._batched_nms_coordinate_trick

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
        shape_path = ".\code\world_continents\World_Continents.shp"
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

def create_detection_model(index_to_class, model_path, device, kwargs, target_gsd):
    num_classes = len(index_to_class) + 1
    backbone = resnet_fpn_backbone(backbone_name = "resnet101", weights = "DEFAULT")
    box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(backbone.out_channels * 4, num_classes)
    min_bird_size = 25
    max_bird_size = 125
    step_size = int(((max_bird_size - min_bird_size)/4))
    anchor_sizes = list(range(min_bird_size, max_bird_size + step_size, step_size))
    anchor_sizes = tuple((size / (target_gsd * 100),) for size in anchor_sizes)
    # anchor_sizes = ((0.25,), (0.5,), (0.75,), (1.0,), (1.25,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    kwargs["rpn_anchor_generator"] = rpn_anchor_generator
    model = torchvision.models.detection.__dict__["FasterRCNN"](box_predictor = box_predictor, backbone = backbone, **kwargs)
    model.load_state_dict(torch.load(os.path.join(model_path, "model_best_state_dict.pth"), map_location=device))
    model.eval()
    model = model.to(device)
    return model

def update_detection_model(model, regional_filter, device):
    model.roi_heads.filter = regional_filter.to(device)
    return model

def prepare_image_for_detection(image_path, overlap, patch_width, patch_height, gsd, target_gsd):
    image = PIL.Image.open(image_path).convert('RGB')
    scale = target_gsd/gsd
    width, height = image.size
    width = int(width / scale)
    height = int(height / scale)
    image = image.resize((width, height)) 
    n_crops_width = math.ceil((width - overlap) / (patch_width - overlap))
    n_crops_height = math.ceil((height - overlap) / (patch_height - overlap))
    padded_width = n_crops_width * (patch_width - overlap) + overlap
    padded_height = n_crops_height * (patch_height - overlap) + overlap
    pad_width = (padded_width - width) / 2
    pad_height = (padded_height - height) / 2
    batch = []
    for height_index in range(n_crops_height):
        for width_index in range(n_crops_width):
            left = width_index * (patch_width - overlap) - pad_width
            right = left + patch_width
            top = height_index * (patch_height - overlap) - pad_height
            bottom = top + patch_height
            patch = image.crop((left, top, right, bottom))
            patch = torchvision.transforms.PILToTensor()(patch)
            patch = torchvision.transforms.ConvertImageDtype(torch.float)(patch)
            patch = patch.unsqueeze(0)
            batch.append(patch)
    dim = torch.Tensor(0, 3, patch_height, patch_width)
    batch = torch.cat(batch, out=dim)
    return batch, pad_width, pad_height, n_crops_height, n_crops_width, scale

def detect_birds(kwargs, image_path, model, device, index_to_class, overlap, patch_width, patch_height, reject, gsd, target_gsd):
    print("\tPatching...")
    batch, pad_width, pad_height, n_crops_height, n_crops_width, scale = prepare_image_for_detection(image_path, overlap, patch_width, patch_height, gsd, target_gsd)
    max_batch_size = 15
    batch_length = batch.size()[0]
    sub_batch_lengths = [max_batch_size] * math.floor(batch_length/max_batch_size)
    sub_batch_lengths.append(batch_length % max_batch_size)
    sub_batches = torch.split(batch, sub_batch_lengths)
    predictions = []
    print("\tDetecting...")
    with torch.no_grad():
        for sub_batch in sub_batches:
            if len(sub_batch) > 0:
                prediction = model(sub_batch.to(device))
                predictions.extend(prediction)
    boxes = torch.empty(0, 4)
    scores = torch.empty(0)
    class_scores = torch.empty(0, len(index_to_class))
    labels = torch.empty(0, dtype=torch.int64)
    for height_index in range(n_crops_height):
        for width_index in range(n_crops_width):
            patch_index = height_index * n_crops_width + width_index
            batch_boxes = predictions[patch_index]["boxes"]
            batch_scores = predictions[patch_index]["scores"]
            batch_class_scores = predictions[patch_index]["class_scores"]
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
                batch_class_scores = batch_class_scores[index]
                batch_labels = batch_labels[index]
            padding_left = (patch_width - overlap) * width_index - pad_width
            padding_top = (patch_height - overlap) * height_index - pad_height
            #scale
            adjustment = torch.tensor([[padding_left, padding_top, padding_left, padding_top]])
            adj_boxes = torch.add(adjustment, batch_boxes.to(torch.device("cpu")))
            adj_boxes = torch.mul(adj_boxes, scale)
            boxes = torch.cat((boxes, adj_boxes), 0)
            scores = torch.cat((scores, batch_scores.to(torch.device("cpu"))), 0)
            class_scores = torch.cat((class_scores, batch_class_scores.to(torch.device("cpu"))), 0)
            labels = torch.cat((labels, batch_labels.to(torch.device("cpu"))), 0)
    nms_indices = box_ops.batched_nms(boxes, scores, labels, kwargs["box_nms_thresh"])
    boxes = boxes[nms_indices]
    scores = scores[nms_indices].tolist()
    class_scores = class_scores[nms_indices].tolist()
    labels = labels[nms_indices].tolist()
    return boxes, class_scores

def create_annotation(boxes, scores, image_name, width, height, index_to_class):
    print("\tCreating annotation...")
    points = []
    species_labels = []
    species_scores = []
    bird_scores = []
    for index in range(len(boxes)):
        box = boxes[index]
        points.append([
            [float(box[0]), float(box[1])],
            [float(box[2]), float(box[3])]])
        bird_score = round(sum(scores[index]), 2)
        species_score = round(max(scores[index]), 2)
        species_label = index_to_class[str(scores[index].index(max(scores[index])) + 1)]
        species_labels.append(species_label)
        bird_scores.append(bird_score)
        species_scores.append(species_score)
    label_name = os.path.splitext(image_name)[0] + '.json'
    label_path = os.path.join("./images", label_name)
    shapes = []
    for i in range(0, len(species_labels)):
        shapes.append({
            "label": ' '.join(("Bird:",str(bird_scores[i]), "-", species_labels[i] + ":", str(species_scores[i]))),
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

def create_csv(boxes, scores, image_name, header, index_to_class):
    print("\tUpdating csv...")
    csv_path = "./images/results.csv"
    with open(csv_path, 'a+', newline='') as csvfile:
        fieldnames = ["image_name", "box", "bird"] + list(index_to_class.values())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for index in range(len(boxes)):
            if header == False:
                writer.writeheader()
                header = True
            row = {"image_name": image_name, "box": boxes[index].tolist(), "bird": round(sum(scores[index]), 2)}
            for fieldname in index_to_class.values():
                row[fieldname] = scores[index][list(index_to_class.values()).index(fieldname)]
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

    # remove old results file
    if os.path.exists("./images/results.csv"):
        os.remove("./images/results.csv")
    
    # define constants
    overlap = 300
    patch_height = 800
    patch_width = 800
    reject = 10
    target_gsd = 0.005
    header = False

    # create detection model
    device = torch.device("cuda")
    model_path = os.path.join("./models/bird-detector")
    kwargs = json.load(open(os.path.join(model_path, "kwargs.txt")))
    index_to_class = json.load(open(os.path.join(model_path, "index_to_class.json")))
    birds_by_region = json.load(open(os.path.join(model_path, "birds_by_region.json")))
    model = create_detection_model(index_to_class, model_path, device, kwargs, target_gsd)

    for file in os.listdir("./images"):
        if file.lower().endswith((".jpg", ".jpeg")):
            print("Annotating: ", file)
            # get exif data and determine region
            image_path = os.path.join("./images", file)
            image = PIL.Image.open(image_path)
            image_width, image_height = image.size
            # load image exif data
            exif_dict = piexif.load(image.info['exif'])
            # try calculating gsd
            print("\tTrying to determine gsd from image metadata")
            try:
                height = get_xmp(image_path)
                gsd = get_gsd(exif_dict, sensor_sizes, image_width, image_height, height)
                print("\tGSD: ", gsd)
            except:
                gsd = target_gsd
                print("\tCouldn't determine gsd, assuming {} meters/pixel".format(target_gsd))
            # try determining region from image metadata
            print("\tTrying to determine region from image metadata")
            try:
                region = get_region(exif_dict)
                print("\tRegion: ", region)
            except:
                region = "Oceania"
                print("Couldn't determine region, so including all species")
            # if gsd too large skip image
            if gsd > 0.0075:
                print("GSD too large")
                continue
            # update model with new regional filter
            regional_filter = torch.zeros(len(index_to_class) + 1)
            regional_filter[0] = 1
            for i in range(1, len(index_to_class) + 1):
                species = ' '.join(index_to_class[str(i)].split("_")[-2:])
                if species in birds_by_region[region]:
                    regional_filter[i] = 1
            # update detection model regional filter
            model = update_detection_model(model, regional_filter, device)
            # detect birds
            boxes, scores = detect_birds(kwargs, image_path, model, device, index_to_class, overlap, patch_width, patch_height, reject, gsd, target_gsd)
            # create label file
            create_annotation(boxes, scores, file, image_width, image_height, index_to_class)
            # create dictionary of results
            if len(boxes) > 0:
                create_csv(boxes, scores, file, header, index_to_class)
            header = True
    print("\tDone!")

if __name__ == "__main__":
    main()