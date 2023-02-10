import gradio as gr
import os
import math
import csv
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
def create_detection_model(index_to_class, model_path, device, kwargs, target_gsd, class_filter):
    num_classes = len(index_to_class) + 1
    # Create backbone
    backbone = resnet_fpn_backbone(backbone_name = "resnet101", weights = "DEFAULT")
    # Create box head
    box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(backbone.out_channels * 4, num_classes)
    # Create new anchors
    min_bird_size = 25
    max_bird_size = 125
    step_size = int(((max_bird_size - min_bird_size)/4))
    anchor_sizes = list(range(min_bird_size, max_bird_size + step_size, step_size))
    anchor_sizes = tuple((size / (target_gsd * 100),) for size in anchor_sizes)
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
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
def prepare_image_for_detection(image, overlap, patch_width, patch_height, gsd, target_gsd):
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

# perform detection
def detect_birds(kwargs, image, model, device, index_to_class, overlap, patch_width, patch_height, reject, gsd, target_gsd):
    batch, pad_width, pad_height, n_crops_height, n_crops_width, scale = prepare_image_for_detection(image, overlap, patch_width, patch_height, gsd, target_gsd)
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
    return boxes, class_scores, scores, labels

# Calculate the GPS of each bird based on pixel co-ordinates
def calculate_gps(ref_latitude, ref_longitude, dx, dy):
    r_earth = 6371000.0
    latitude  = ref_latitude  + (-dy / r_earth) * (180 / math.pi)
    longitude = ref_longitude + (dx / r_earth) * (180 / math.pi) / math.cos(ref_latitude * math.pi/180)
    return(latitude, longitude)

# Create the results file
def create_csv(boxes, class_scores, labels, index_to_class, ref_latitude, ref_longitude, gsd):
    csv_path = "results.csv"
    with open(csv_path, 'a+', newline='') as csvfile:
        fieldnames = ["box", "x", "y", "latitude", "longitude", "bird", "species"] + list(index_to_class.values())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(boxes)):
            box = boxes[index].tolist()
            x = (box[0] + box[2])/2
            y = (box[1] + box[3])/2
            dx = x * gsd
            dy = y * gsd
            latitude, longitude = calculate_gps(ref_latitude, ref_longitude, dx, dy)
            row = {"box": box, "x": x, "y": y, "latitude": latitude, "longitude": longitude, "bird": round(sum(class_scores[index]), 2), "species": index_to_class[labels[index]]}
            for fieldname in index_to_class.values():
                row[fieldname] = class_scores[index][list(index_to_class.values()).index(fieldname)]
            writer.writerow(row)
    return

def main(image, gsd, reference_latitude, reference_longitude, included_species, box_score_thresh, box_nms_thresh, gpu):  
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
        overlap = 300
        patch_height = 800
        patch_width = 800
        reject = 10
    max_size = max(patch_height, patch_width)
    min_size = min(patch_height, patch_width)
    # define kwargs
    kwargs = {
        "rpn_pre_nms_top_n_test": 10000,
        "rpn_post_nms_top_n_test": 10000,
        "rpn_nms_thresh": 0.2,
        "rpn_score_thresh": 0.01,
        "box_score_thresh": box_score_thresh,
        "box_nms_thresh": box_nms_thresh,
        "box_detections_per_img": 1000,
        "min_size": min_size,
        "max_size": max_size}
    # create class filter
    class_filter = torch.zeros(len(index_to_class) + 1)
    class_filter[0] = 1
    for i in range(1, len(index_to_class) + 1):
        species = index_to_class[i]
        if species in included_species:
            class_filter[i] = 1
    # create detection model
    model_path = "./model_best_state_dict.pth"
    model = create_detection_model(index_to_class, model_path, device, kwargs, target_gsd, class_filter)

    # run detection model
    boxes, class_scores, scores, labels = detect_birds(kwargs, image, model, device, index_to_class, overlap, patch_width, patch_height, reject, gsd, target_gsd)

    # draw boxes on images
    labelled_scores = ["Bird: " + str(round(sum(class_scores[i]),2)) + "\n" + index_to_class[labels[i]] + ": " + str(round(scores[i], 2)) for i in range(len(scores))]
    tensor_image = torchvision.transforms.PILToTensor()(image)
    labelled_image = draw_bounding_boxes(tensor_image, boxes, labelled_scores, colors = "blue", width = 2)
    labelled_image = torchvision.transforms.ToPILImage()(labelled_image)

    # Create result file
    create_csv(boxes, class_scores, labels, index_to_class, reference_latitude, reference_longitude, gsd)
    
    return [labelled_image, "results.csv"]

index_to_class = {
    1: "Pacific Black Duck",
    2: "Silver Gull",
    3: "Pied Stilt",
    4: "Gull-billed Tern",
    5: "Bar-tailed Godwit",
    6: "Australian Wood Duck",
    7: "Masked Lapwing",
    8: "Royal Spoonbill",
    9: "Australian White Ibis"}

inputs = [
    gr.Image(type="pil", label="Orthomosaic"),
    gr.Number(value = 0.007, label="Ground Sampling Distance"),
    gr.Number(value = -27.44423, label="Latitude of top left corner"),
    gr.Number(value = 153.1816, label="Longitude of top left corner"),
    gr.CheckboxGroup(
        choices = list(index_to_class.values()),
        value = list(index_to_class.values()), label = "Included Species"),
    gr.Slider(minimum = 0, maximum = 1, value = 0.7, label = "Classification Score Threshold"),
    gr.Slider(minimum = 0, maximum = 1, value = 0.2, label = "Box Overlap Threshold"),
    gr.Checkbox(value = False, label = "Does the app have GPU access")]

outputs = [
    gr.Image(type="pil", label = "Detected Birds"),
    gr.File(label="Download Results")]

# Add title to Gradio Interface
title = "AerialAI: AI to detect objects in drone imagery"

# Add title to Gradio Interface
description = "This application determines the determines the GPS locatoin of birds in aerial orthophotos and classifies their species returning a csv of the results"

# Add examples to Gradio Interface
examples = [
    [pathlib.Path('demo1.jpg').as_posix(), 0.005, -27.48388, 153.11551, list(index_to_class.values()), 0.7, 0.2, False],
    [pathlib.Path('demo2.jpg').as_posix(), 0.005, -27.04474, 153.10526, list(index_to_class.values()), 0.7, 0.2, False],
    [pathlib.Path('demo3.jpg').as_posix(), 0.007, -27.44423, 153.18161, list(index_to_class.values()), 0.7, 0.2, False]]
# Generate Gradio interface
demo = gr.Interface(
    fn = main,
    inputs = inputs,
    outputs = outputs,
    title = title,
    description = description,
    examples = examples)
demo.queue().launch()