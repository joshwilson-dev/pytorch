# Title:        Inference
# Description:  Evaluated the performance of on object detection network using
#               COCO Evaluation Metrics
# Author:       Anonymous
# Date:         05/06/2024

#### Import packages
# Base packages
import os
import csv
import time
import json
import shutil
import pathlib
import colorsys
import shutil
import math

# External packages
from PIL import Image
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection import roi_heads
import gradio as gr

# Custom packages
import CS14_custom_roi_heads
import CS9_utils

# Redefining roi_heads to return scores for all classes and filter the classes
# based on a supplied filter
roi_heads.RoIHeads.postprocess_detections = CS14_custom_roi_heads.postprocess_detections
roi_heads.RoIHeads.forward = CS14_custom_roi_heads.forward

def create_model(class_filter, device):
    print("Creating model")
    backbone = resnet_fpn_backbone(backbone_name = backbone_name, weights = None)
    model = torchvision.models.detection.__dict__[model_name](backbone = backbone, num_classes = n_classes, **kwargs)
    checkpoint = torch.load(os.path.join(model_path, checkpoint_name), map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.roi_heads.class_filter = class_filter.to(device)
    torch.backends.cudnn.deterministic = True
    model.eval()
    model.to(device)
    return model

def load_dataset(images, gsd):
    print("Loading data")
    dataset = Custom_Dataset(images = images, gsd = gsd, panel_size=panel_size, overlap=overlap)
    data_loader = torch.utils.data.DataLoader(dataset, sampler=torch.utils.data.SequentialSampler(dataset))
    return data_loader

def reject_boxes(prediction, top, left, scaled_width, scaled_height):
    inds = []
    for i, box in enumerate(prediction['boxes']):
        if (
            (box[0] > margin or left == 0)
            and (box[1] > margin or top == 0)
            and (box[2] < panel_size - margin or left + panel_size > scaled_width - margin)
            and (box[3] < panel_size - margin or top + panel_size > scaled_height - margin)
        ):
            inds.append(i)
    prediction['boxes'] = prediction['boxes'][inds]
    prediction['labels'] = prediction['labels'][inds]
    prediction['scores'] = prediction['scores'][inds]
    prediction['class_scores'] = prediction['class_scores'][inds]
    return prediction

def adjust_boxes(prediction, top, left):
    for box in prediction['boxes']:
        box[0] += left
        box[1] += top
        box[2] += left
        box[3] += top
    return prediction

def rescale_boxes(prediction, scale):
    for box in prediction["boxes"]: box *= scale
    return prediction

def create_annotation(prediction, filename, width, height):
    shapes = []

    for box, label in zip(prediction['boxes'], prediction["labels"]):
        box = box.tolist()
        label = index_to_class[str(label.item())]
        label = label["name"] + " - " + label["age"]
        shapes.append({
            "label": label,
            "points": [[box[0], box[1]], [box[2], box[3]]],
            "group_id": 'null',
            "shape_type": "rectangle",
            "flags": {}})
    label_name = filename + '.json'
    label_path = os.path.join(output_dir, "results", label_name)
    annotation = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": filename + '.png',
        "imageData": 'null',
        "imageHeight": int(height),
        "imageWidth": int(width)}

    annotation_str = json.dumps(annotation, indent = 2).replace('"null"', 'null')
    with open(label_path, 'w') as annotation_file:
        annotation_file.write(annotation_str)
    return

def create_csv(prediction, filename):
    csv_path = os.path.join(output_dir, "results/results.csv")
    header = False if os.path.exists(csv_path) else True
    classes = [name + " - " + age for name, age in zip(categories["name"], categories["age"])]
    # Step through predictions and add to csv
    with open(csv_path, 'a+', newline='') as csvfile:
        fieldnames = ["filename", "left", "top", "right", "bottom"] + classes

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if header == True:
            writer.writeheader()
        for index in range(len(prediction["boxes"])):
            box = prediction["boxes"][index].tolist()
            row = {"filename": filename + ".png", "left": box[0], "top": box[1], "right": box[2], "bottom": box[3]}
            for i, fieldname in enumerate(classes):
                row[fieldname] = round(prediction["class_scores"][index][i].tolist(),2)
            writer.writerow(row)
    return

def generate_distinct_colors(n):
    hues = [i / n for i in range(n)]
    saturation = 0.8
    value = 0.8
    colors = []

    for hue in hues:
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        rgb_color = (int(r * 255), int(g * 255), int(b * 255))
        colors.append(rgb_color)

    return colors

def label_image(image, filename, prediction):
    # create string with labels and scores
    labels_with_scores = []
    for label, score in zip(prediction["labels"], prediction["scores"]):
        label = index_to_class[str(label.item())]
        score =  str(round(score.item(), 2))
        labels_with_scores.append(label["name"] + "-" + label["age"] + ": " + score)

    unique_labels = list(set(prediction["labels"]))
    distinct_colours = generate_distinct_colors(len(unique_labels))
    colours = []
    for label in prediction["labels"]:
        colours.append(distinct_colours[unique_labels.index(label)])
    image = (image * 255).to(torch.uint8)
    labelled_image = draw_bounding_boxes(image, prediction["boxes"], labels = labels_with_scores, colors = colours, width = 2)
    labelled_image = F.to_pil_image(labelled_image)
    labelled_image.save('./outputs/results/' + filename + '-labelled.png')
    print("Done Label Image")
    return labelled_image

class Custom_Dataset():
    def __init__(self, images, gsd, panel_size, overlap):
        self.img_names = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        self.panel_size = panel_size
        self.overlap = overlap
        gsds = [gsd * 0.001] * len(self.img_names)
        self.gsds = gsds

    def __getitem__(self, id, progress=gr.Progress()):
        filepath = self.img_names[id]
        filename = os.path.splitext(os.path.basename(filepath))[0]
        gsd = self.gsds[id]
        image = Image.open(os.path.join(filepath))
        image = image.convert("RGB")
        scale = target_gsd/gsd
        width, height = image.size
        scaled_width = int(width / scale)
        scaled_height = int(height / scale)
        scaled_image = image.resize((scaled_width, scaled_height))

        # Crop the image into a grid with overlap
        crops = []
        if self.panel_size < scaled_width:
            overlap_width = self.overlap
        else:
            overlap_width = 0
            scaled_width = self.panel_size

        if self.panel_size < scaled_height:
            overlap_height = self.overlap 
        else:
            overlap_height = 0
            scaled_height = self.panel_size
        n_crops = (
            math.ceil((scaled_width - overlap_width) / (self.panel_size - overlap_width)) *
            math.ceil((scaled_height - overlap_height) / (self.panel_size - overlap_height)))
        crop = 1
        top = 0
        while top + overlap_height < scaled_height:
            left = 0
            while left + overlap_width < scaled_width:
                # Crop and convert to tensor
                cropped_img = F.crop(scaled_image, top, left, self.panel_size, self.panel_size)

                cropped_img = F.to_tensor(cropped_img)[0]
                crops.append(cropped_img)
                left += self.panel_size - overlap_width
                progress((crop, n_crops), desc = "Pre-processing image {} of {}".format(n_image, n_images))
                crop += 1
            top += self.panel_size - overlap_height
        return F.to_tensor(image), crops, scaled_width, scaled_height, scale, filename, overlap_width, overlap_height, width, height

    def __len__(self):
        return len(self.img_names)

def main(images, gpu, gsd, det_score_thresh, class_score_thresh, box_nms_thresh, included_classes, progress=gr.Progress()):

    progress(0, desc="Preparing model...")

    # Delete previous results
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    os.mkdir(output_dir + "results")

    # Is the code running on a GPU?
    if gpu: device = torch.device("cuda")
    else: device = torch.device("cpu")

    # Update the thresholds and included classes
    kwargs["box_nms_thresh"] = box_nms_thresh
    if included_classes[0] == "All":
        included_classes = categories["name"]
    class_filter = torch.zeros(len(index_to_class) + 1)
    class_filter[0] = 1
    for i in range(len(categories["name"])):
        species = categories["name"][i]
        if species in included_classes:
            class_filter[i + 1] = 1   

    # Create the model
    model = create_model(class_filter, device)

    progress(0, desc="Loading images...")

    # Load the dataset
    data_loader = load_dataset(images, gsd)

    # Running model
    torch.set_num_threads(1)
    metric_logger = CS9_utils.MetricLogger(delimiter="  ")
    global n_images, n_image
    n_images = len(images)
    n_image = 1
    for image, crops, scaled_width, scaled_height, scale, filename, overlap_width, overlap_height, width, height in metric_logger.log_every(data_loader, 100, "Inference:"):
        image = image[0]
        filename = filename[0]

        if torch.cuda.is_available(): torch.cuda.synchronize()
        
        # Run prediction in batches
        model_time = time.time()
        prediction = []
        crop = 0
        for img_batch in torch.split(torch.stack(crops), batchsize):
            # print(torch.cuda.get_device_properties(0).total_memory)
            # print(torch.cuda.memory_reserved(0))
            # print(torch.cuda.memory_allocated(0))
            # print("\n")
            progress((crop, len(crops)), desc = "Processing image {} of {}".format(n_image, n_images))
            torch.cuda.empty_cache()
            batch_prediction = model(img_batch.to(device))
            batch_prediction = [{k: v.cpu().detach() for k, v in t.items()} for t in batch_prediction]
            prediction.extend(batch_prediction)
            crop += batchsize
        model_time = time.time() - model_time
    
        progress(0, desc = "Post-processing image {} of {}".format(n_image, n_images))

        # Adjust predictions due to grid splitting
        i = 0
        top = 0
        while top + overlap_height < scaled_height:
            left = 0
            while left + overlap_width < scaled_width:
                # Reject boxes near overlapping edges
                prediction[i] = reject_boxes(prediction[i], top, left, scaled_width, scaled_height)
                # Adjust box coordinates based on the crop position
                prediction[i] = adjust_boxes(prediction[i], top, left)
                left += int(panel_size - overlap_width)
                i += 1
            top += int(panel_size - overlap_height)

        # Combine crops into single prediction
        prediction = {
            'boxes': torch.cat([item['boxes'] for item in prediction]),
            'labels': torch.cat([item['labels'] for item in prediction]),
            'scores': torch.cat([item['scores'] for item in prediction]),
            'class_scores': torch.cat([item['class_scores'] for item in prediction])}

        # Non-maximum supression
        ids = torch.tensor([1]*len(prediction['labels']))
        inds = torchvision.ops.boxes.batched_nms(prediction['boxes'], prediction['scores'], ids, iou_threshold = kwargs["box_nms_thresh"])
        prediction['boxes'] = prediction['boxes'][inds]
        prediction['labels'] = prediction['labels'][inds]
        prediction['scores'] = prediction['scores'][inds]
        prediction['class_scores'] = prediction['class_scores'][inds]

        # Remove low scoring boxes for bird vs background
        inds = torch.where(torch.sum(prediction['class_scores'], 1) > det_score_thresh)[0]
        prediction['boxes'] = prediction['boxes'][inds]
        prediction['labels'] = prediction['labels'][inds]
        prediction['scores'] = prediction['scores'][inds]
        prediction['class_scores'] = prediction['class_scores'][inds]

        # Remove low scoring boxes for class score
        inds = torch.where(prediction['scores'] > class_score_thresh)[0]
        prediction['boxes'] = prediction['boxes'][inds]
        prediction['labels'] = prediction['labels'][inds]
        prediction['scores'] = prediction['scores'][inds]
        prediction['class_scores'] = prediction['class_scores'][inds]
    
        # Rescale crops to original scale
        rescale_boxes(prediction, scale)

        # Create label file
        create_annotation(prediction, filename, width, height)

        # Add to csv
        create_csv(prediction, filename)

        # Label image
        label_image(image, filename, prediction)
        
        metric_logger.update(model_time=model_time)
        n_image += 1
    # Zip outputs
    shutil.make_archive('./outputs/results', 'zip', './outputs/results')

    return ['./outputs/results.zip']

# Setup
Image.MAX_IMAGE_PIXELS = 1000000000000
output_dir = "outputs/"
kwargs = {
    "rpn_pre_nms_top_n_test": 250,
    "rpn_post_nms_top_n_test": 250,
    "rpn_nms_thresh": 0.5,
    "rpn_score_thresh": 0.01,
    "box_detections_per_img": 100}
img_dir = "images/"
model_path = ""
model_name = "FasterRCNN"
backbone_name = "resnet101"
checkpoint_name = "AS2_model.pth"
panel_size = 800
overlap = 200
margin = 5
target_gsd = 0.005
batchsize = 1

index_to_class = json.load(open(os.path.join(model_path, "AS1_index_to_class.json")))
n_classes = len(index_to_class) + 1

categories = {}
for key in index_to_class["1"].keys():
    categories[key] = [class_info[key] for class_info in index_to_class.values()]

included_classes = categories["name"]

inputs = [
    gr.File(file_count="directory", type = "filepath"),
    gr.Checkbox(value = False, label = "Is this running on a GPU? (If demo leave unchecked)"),
    gr.Number(value = 5, label="Ground sampling distance of images in millimeters per pixel."),
    gr.Slider(minimum = 0, maximum = 1, value = 0.75, label = "Detection score threshold"),
    gr.Slider(minimum = 0, maximum = 1, value = 0.0, label = "Classification score threshold"),
    gr.Slider(minimum = 0, maximum = 1, value = 0.2, label = "Box overlap threshold"),
    gr.Dropdown(
        choices = ["All"] + sorted(set(categories["name"])),
        value = ["All"],
        label = "Included classes",
        multiselect = True)]

outputs = [gr.File(label="Download Results", file_count="directory")]

# Add title to Gradio Interface
title = "AerialBirdID: A computer vision tool to detect birds in images captured from a drone"

# Add description to Gradio Interface
description = """
<div style="font-family: Arial, sans-serif; line-height: 1.6;">
    <p><strong>Description: </strong>This application determines the pixel location of birds in images captured from a drone and identifies their species and age.</p>
    <p><strong>DEMO: </strong>This a free demo of this application.</p>
    <ul>
        <li>It has limited memory and will only accept a few small images.</li>
        <li>It will process images slowly.</li>
        <li>To access the full version please contact the author.</li>
    </ul>
    <p><strong>Example:</strong></p>
    <ul>
        <li>Click the example at the bottom of the page.</li>
        <li>Click submit and wait for the images to be processed.</li>
        <li>Download the results.</li>
    </ul>
    <p><strong>Results:</strong></p>
    <ul>
        <li>A comma-separated value file detailing:
        <ul>
            <li>Filename: The name of the image file within which the bird was predicted.</li>
            <li>Coordinates: The coordinates of the box encompassing the predicted bird.</li>
            <li>Scores: The predicted score for each class included in the network.</li>
        </ul>
        </li>
        <li>A json file for each image in the Labelme annotation format (this can be viewed by putting it in the same folder as the original image file and opening that folder in the Labelme application).</li>
        <li>A jpg file for each image with the predictions overlayed.</li>
    </ul>
    <p><strong>Using the tool:</strong></p>
    <ol>
        <li>Select a folder containing your image files.</li>
        <li>Specify if the tool has access to a GPU (the demo version does not).</li>
        <li>Enter the ground sampling distance of the images in millimeters per pixel.</li>
        <li>Enter the threshold for classifying a bird vs background.</li>
        <li>Enter the threshold for classifying between species.</li>
        <li>Enter the threshold for removing overlapping boxes.</li>
        <li>Select the species you want the tool to consider during classification.</li>
        <li>Click submit and wait for the images to be processed.</li>
        <li>Download the results.</li>
    </ol>
</div>
"""

# Add examples to Gradio Interface
examples = [[[pathlib.Path('./demos/demo1.jpg').as_posix(), pathlib.Path('./demos/demo2.jpg').as_posix()], None, None, None, None, None, None]]

# Generate Gradio interface
demo = gr.Interface(
    fn = main,
    inputs = inputs,
    outputs = outputs,
    title = title,
    description = description,
    examples = examples,
    cache_examples = False)

demo.queue().launch()