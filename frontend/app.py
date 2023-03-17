import gradio as gr
import os
import math
import csv
import json
import itertools
import pathlib
import torch
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import roi_heads
from torchvision.ops import boxes as box_ops
import custom_roi_heads
import custom_boxes

# redefining roi_heads to return scores for all classes
# and filter the classes based on a supplied filter
roi_heads.RoIHeads.postprocess_detections = custom_roi_heads.postprocess_detections
roi_heads.RoIHeads.forward = custom_roi_heads.forward

# redefining boxes to do nms on all classes, not class specific
box_ops.batched_nms = custom_boxes.batched_nms
box_ops._batched_nms_coordinate_trick = custom_boxes._batched_nms_coordinate_trick

# function to generate pytorch detection model
def create_detection_model(common_name, model_path, device, kwargs, target_gsd, class_filter):
    num_classes = len(common_name) + 1
    # Create backbone
    backbone = resnet_fpn_backbone(backbone_name = "resnet101", weights = "DEFAULT")
    # Create box head
    box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(backbone.out_channels * 4, num_classes)
    # Create new anchors
    min_bird_size = 25
    max_bird_size = 225
    step_size = int(((max_bird_size - min_bird_size)/4))
    anchor_sizes = list(range(min_bird_size, max_bird_size + step_size, step_size))
    anchor_sizes = tuple((size / (target_gsd * 100),) for size in anchor_sizes)
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    print(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    kwargs["rpn_anchor_generator"] = rpn_anchor_generator
    # Create model
    model = torchvision.models.detection.__dict__["FasterRCNN"](box_predictor = box_predictor, backbone = backbone, **kwargs)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.roi_heads.class_filter = class_filter.to(device)
    model.eval()
    model = model.to(device)
    return model

# function to prepare image for detection with patching to fit in GPU
def prepare_image_for_detection(image, overlap, patch_width, patch_height, scale):
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
    return batch, pad_width, pad_height, n_crops_height, n_crops_width

# perform detection
def detect_birds(kwargs, image, model, device, common_name, overlap, patch_width, patch_height, reject, scale):
    batch, pad_width, pad_height, n_crops_height, n_crops_width = prepare_image_for_detection(image, overlap, patch_width, patch_height, scale)
    max_batch_size = 15
    batch_length = batch.size()[0]
    sub_batch_lengths = [max_batch_size] * math.floor(batch_length/max_batch_size)
    sub_batch_lengths.append(batch_length % max_batch_size)
    sub_batches = torch.split(batch, sub_batch_lengths)
    predictions = []
    with torch.no_grad():
        for sub_batch in sub_batches:
            if len(sub_batch) > 0:
                prediction = model(sub_batch.to(device))
                predictions.extend(prediction)
    boxes = torch.empty(0, 4)
    scores = torch.empty(0)
    class_scores = torch.empty(0, len(common_name))
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
    # remove overlapping detections
    inds = box_ops.batched_nms(boxes, scores, labels, kwargs["box_nms_thresh"])
    boxes, scores, labels, class_scores = boxes[inds], scores[inds], labels[inds], class_scores[inds]
    # remove low score bird detections
    inds = torch.where(torch.sum(class_scores, 1) > kwargs["box_score_thresh"])
    boxes, scores, labels, class_scores = boxes[inds], scores[inds], labels[inds], class_scores[inds]
    return boxes, scores, labels, class_scores

def size_filters(boxes, scores, labels, class_scores, min_pixel_area, scale):
    areas = torch.tensor([(box[2] - box[0]) * (box[3] - box[1]) / scale for box in boxes])
    min_areas = []
    max_areas = []
    for label in labels:
        min_areas.append(index_to_class[str(label.item())]["min_area"])
        max_areas.append(index_to_class[str(label.item())]["max_area"])
    min_areas = torch.tensor(min_areas)
    max_areas = torch.tensor(max_areas)
    print("areas", areas)
    inds = torch.where((areas > min_pixel_area) & (areas > min_areas) & (areas < max_areas))
    boxes, scores, labels, class_scores = boxes[inds], scores[inds], labels[inds], class_scores[inds]
    return boxes, scores, labels, class_scores

def taxon_results(class_scores, taxons):
    keys = list(taxons.keys())[1:]
    results = {key: [{"none": 0}] * len(class_scores) for key in keys}
    for key in keys:
        for group in set(taxons[key]):
            inds = torch.tensor([i for i in range(len(taxons[key])) if taxons[key][i] == group])
            group_scores = torch.sum(torch.index_select(class_scores, 1, inds), 1)
            for i in range(len(group_scores)):
                if group_scores[i] > list(results[key][i].values())[0]:
                    results[key][i] = {group: group_scores[i]}
    return results

def label_image(image, boxes, tax_res, kwargs):
    labels = [""] * len(boxes)
    for key in tax_res.keys():
        for i in range(len(tax_res[key])):
            score = round(float(list(tax_res[key][i].values())[0].tolist()), 2)
            group = list(tax_res[key][i].keys())[0]
            if score > kwargs["box_score_thresh"]:
                labels[i] += "\n" + group + ": " + str(score)
    tensor_image = torchvision.transforms.PILToTensor()(image)
    labelled_image = draw_bounding_boxes(tensor_image, boxes, labels, colors = "blue", width = 2)
    labelled_image = torchvision.transforms.ToPILImage()(labelled_image)
    return labelled_image

# Calculate the GPS of each bird based on pixel co-ordinates
def calculate_gps(ref_latitude, ref_longitude, dx, dy):
    r_earth = 6371000.0
    latitude  = ref_latitude  + (-dy / r_earth) * (180 / math.pi)
    longitude = ref_longitude + (dx / r_earth) * (180 / math.pi) / math.cos(ref_latitude * math.pi/180)
    return(latitude, longitude)

# Create the results file
def create_csv(boxes, class_scores, tax_res, ref_latitude, ref_longitude, gsd):
    csv_path = "results.csv"
    groups = list(tax_res.keys())
    group_scores = [group + " score" for group in groups]
    group_results = list(itertools.chain(*zip(groups, group_scores)))
    with open(csv_path, 'a+', newline='') as csvfile:
        fieldnames = ["box", "latitude", "longitude"] + group_results + taxons["common_name"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(boxes)):
            box = boxes[index].tolist()
            x = (box[0] + box[2])/2
            y = (box[1] + box[3])/2
            dx = x * gsd
            dy = y * gsd
            latitude, longitude = calculate_gps(ref_latitude, ref_longitude, dx, dy)
            row = {"box": box, "latitude": latitude, "longitude": longitude}
            for item in tax_res.items():
                row[item[0]] = list(item[1][index].keys())[0]
                row[item[0] + ' score'] = list(item[1][index].values())[0].tolist()
            for fieldname in taxons["common_name"]:
                row[fieldname] = class_scores[index][taxons["common_name"].index(fieldname)].tolist()
            writer.writerow(row)
    return

def main(gpu, image, gsd, ref_latitude, ref_longitude, min_pixel_area, box_score_thresh, box_nms_thresh, included_species):  
    if os.path.exists("results.csv"):
        os.remove("results.csv")
    # Define constants
    target_gsd = 0.005
    scale = target_gsd/gsd
    width, height = image.size
    if gpu == False:
        device = torch.device("cpu")
        overlap = 0
        patch_height = int(height / scale)
        patch_width = int(width / scale)
        reject = 0
    else:
        device = torch.device("cuda")
        overlap = 400
        patch_height = 800
        patch_width = 800
        reject = 10
    max_size = max(patch_height, patch_width)
    min_size = min(patch_height, patch_width)
    # define kwargs
    kwargs = json.load(open("kwargs.json"))
    kwargs["box_score_thresh"] = 0
    kwargs["box_nms_thresh"] = box_nms_thresh
    kwargs["min_size"] = min_size
    kwargs["max_size"] = max_size
    # create class filter
    class_filter = torch.zeros(len(index_to_class) + 1)
    class_filter[0] = 1
    for i in range(len(taxons["common_name"])):
        species = taxons["common_name"][i]
        if species in included_species:
            class_filter[i + 1] = 1
    # create detection model
    model_path = "./model_best_state_dict.pth"
    model = create_detection_model(taxons["common_name"], model_path, device, kwargs, target_gsd, class_filter)
    kwargs["box_score_thresh"] = box_score_thresh

    # run detection model
    boxes, scores, labels, class_scores = detect_birds(kwargs, image, model, device, taxons["common_name"], overlap, patch_width, patch_height, reject, scale)

    # filter out boxes with less than min_pixel_area or wrong sized birds
    boxes, scores, labels, class_scores = size_filters(boxes, scores, labels, class_scores, min_pixel_area, scale)

    # calculate taxon results
    tax_res = taxon_results(class_scores, taxons)

    # create image
    labelled_image = label_image(image, boxes, tax_res, kwargs)

    # Create result file
    create_csv(boxes, class_scores, tax_res, ref_latitude, ref_longitude, gsd)
    
    return [labelled_image, "results.csv"]

index_to_class = json.load(open("index_to_class.json"))
taxons = {
    "common_name": [class_info["common_name"] for class_info in index_to_class.values()],
    "class": [class_info["class"] for class_info in index_to_class.values()],
    "order": [class_info["order"] for class_info in index_to_class.values()],
    "family": [class_info["family"] for class_info in index_to_class.values()],
    "genus": [class_info["genus"] for class_info in index_to_class.values()],
    "species": [class_info["species"] for class_info in index_to_class.values()]}
inputs = [
    gr.Checkbox(value = False, label = "Does the app have GPU access"),
    gr.Image(type="pil", label="Orthomosaic"),
    gr.Number(value = 0.005, label="Ground Sampling Distance"),
    gr.Number(value = -27.44423, label="Latitude of top left corner"),
    gr.Number(value = 153.1816, label="Longitude of top left corner"),
    gr.Number(value = 500, label="Minimum Pixel Area of Detections"),
    gr.Slider(minimum = 0, maximum = 1, value = 0.75, label = "Classification Score Threshold"),
    gr.Slider(minimum = 0, maximum = 1, value = 0.4, label = "Box Overlap Threshold"),
    gr.CheckboxGroup(
        choices = taxons["common_name"],
        value = taxons["common_name"], label = "Included Species")]

outputs = [
    gr.Image(type="pil", label = "Detected Birds"),
    gr.File(label="Download Results")]

# Add title to Gradio Interface
title = "AerialAI: AI to detect objects in drone imagery"

# Add title to Gradio Interface
description = "This application determines the determines the GPS locatoin of birds in aerial orthophotos and classifies their species returning a csv of the results"

# Add examples to Gradio Interface
examples = [
    [False, pathlib.Path('demo1.jpg').as_posix(), 0.005, -27.48388, 153.11551, 1000, 0.7, 0.2, taxons["common_name"]],
    [False, pathlib.Path('demo2.jpg').as_posix(), 0.005, -27.04474, 153.10526, 1000, 0.7, 0.2, taxons["common_name"]]]
# Generate Gradio interface
demo = gr.Interface(
    fn = main,
    inputs = inputs,
    outputs = outputs,
    title = title,
    description = description,
    examples = examples)
demo.queue().launch()