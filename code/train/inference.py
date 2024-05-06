import os
import csv
import time
import json
import shutil

from PIL import Image
import torch
import torchvision
import torchvision.transforms.functional as F
import utils
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import roi_heads
import custom_roi_heads

# redefining roi_heads to return scores for all classes
# and filter the classes based on a supplied filter
roi_heads.RoIHeads.postprocess_detections = custom_roi_heads.postprocess_detections
roi_heads.RoIHeads.forward = custom_roi_heads.forward

def create_model():
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

def load_dataset():
    print("Loading data")
    dataset = Custom_Dataset(img_dir = img_dir, crop_size=crop_size, overlap=overlap)
    data_loader = torch.utils.data.DataLoader(dataset, sampler=torch.utils.data.SequentialSampler(dataset))
    return data_loader

def reject_boxes(prediction, top, left, width, height):
    inds = []
    for i, box in enumerate(prediction['boxes']):
        if (
            (box[0] > margin or left == 0)
            and (box[1] > margin or top == 0)
            and (box[2] < crop_size - margin or left + crop_size > width - margin)
            and (box[3] < crop_size - margin or top + crop_size > height - margin)
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

def create_annotation(prediction, image, filename):
    shapes = []
    image = F.to_pil_image(image)
    width, height = image.size
    filename = os.path.splitext(filename[0])[0]
    for box, label, score, class_scores in zip(prediction['boxes'], prediction["labels"], prediction["scores"], prediction["class_scores"]):
        box = box.tolist()
        label = index_to_class[str(label.item())]
        # label = json.dumps(label).replace('"', "'") + ": " + str(round(score.tolist(), 2))
        label = "bird: " + str(round(sum(class_scores).tolist(), 2)) + " - " + label["name"] + ": " + str(round(score.tolist(), 2))
        shapes.append({
            "label": label,
            "points": [[box[0], box[1]], [box[2], box[3]]],
            "group_id": 'null',
            "shape_type": "rectangle",
            "flags": {}})
    label_name = filename + '.json'
    label_path = os.path.join(output_dir, label_name)
    annotation = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": filename + '.jpg',
        "imageData": 'null',
        "imageHeight": height,
        "imageWidth": width}
    annotation_str = json.dumps(annotation, indent = 2).replace('"null"', 'null')
    with open(label_path, 'w') as annotation_file:
        annotation_file.write(annotation_str)
    return

def create_csv(prediction, filename):
    filename = os.path.splitext(filename[0])[0]
    csv_path = os.path.join(output_dir, "results.csv")
    header = False if os.path.exists(csv_path) else True
    classes = [name + " - " + age for name, age in zip(categories["name"], categories["age"])]
    # Step through predictions and add to csv
    with open(csv_path, 'a+', newline='') as csvfile:
        fieldnames = ["filename", "xcentre", "ycentre"] + classes

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if header == True:
            writer.writeheader()
        for index in range(len(prediction["boxes"])):
            box = prediction["boxes"][index].tolist()
            xcentre = int((box[0] + box[2])/2)
            ycentre = int((box[1] + box[3])/2)
            row = {"filename": filename + ".jpg", "xcentre": xcentre, "ycentre": ycentre}
            for i, fieldname in enumerate(classes):
                row[fieldname] = round(prediction["class_scores"][index][i].tolist(),2)
            writer.writerow(row)
    return

class Custom_Dataset():
    def __init__(self, img_dir, crop_size, overlap):
        self.root = img_dir
        self.img_names = [img for img in os.listdir(img_dir) if img.lower().endswith('.jpg')]
        self.crop_size = crop_size
        self.overlap = overlap
        # gsds = [0.005] * len(self.img_names)
        gsds = []
        for name in self.img_names:
            ann_name = os.path.splitext(name)[0]+'.json'
            ann = json.load(open(img_dir + ann_name))
            gsds.append(ann["gsd"])
        self.gsds = gsds

    def __getitem__(self, id):
        filename = self.img_names[id]
        gsd = self.gsds[id]
        image = Image.open(os.path.join(self.root, filename))
        image = image.convert("RGB")
        scale = target_gsd/gsd
        width, height = image.size
        scaled_width = int(width / scale)
        scaled_height = int(height / scale)
        scaled_image = image.resize((scaled_width, scaled_height))

        # Crop the image into a grid with overlap
        crops = []
        if self.crop_size < scaled_width:
            overlap_width = self.overlap
        else:
            overlap_width = 0
            scaled_width = self.crop_size

        if self.crop_size < scaled_height:
            overlap_height = self.overlap 
        else:
            overlap_height = 0
            scaled_height = self.crop_size
        
        top = 0
        while top + overlap_height < scaled_height:
            left = 0
            while left + overlap_width < scaled_width:
                # Crop and convert to tensor
                cropped_img = F.crop(scaled_image, top, left, self.crop_size, self.crop_size)
                cropped_img = F.to_tensor(cropped_img)[0]
                crops.append(cropped_img)
                left += self.crop_size - overlap_width
            top += self.crop_size - overlap_height
        return F.to_tensor(image)[0], crops, scaled_width, scaled_height, scale, filename, overlap_width, overlap_height

    def __len__(self):
        return len(self.img_names)

def main():
    # Load the test dataset
    data_loader = load_dataset()

    # Create the model
    model = create_model()

    # Running model
    torch.set_num_threads(1)
    metric_logger = utils.MetricLogger(delimiter="  ")
    for image, crops, width, height, scale, filename, overlap_width, overlap_height in metric_logger.log_every(data_loader, 100, "Inference:"):
        crops = [crop.to(device) for crop in crops]

        if torch.cuda.is_available(): torch.cuda.synchronize()
        
        # Run prediction in batches
        model_time = time.time()
        prediction = []
        i = 0
        for img_batch in torch.split(torch.stack(crops), batchsize):
            i += 1
            # print(torch.cuda.get_device_properties(0).total_memory)
            # print(torch.cuda.memory_reserved(0))
            # print(torch.cuda.memory_allocated(0))
            # print("\n")
            torch.cuda.empty_cache()
            batch_prediction = model(img_batch)
            batch_prediction = [{k: v.cpu().detach() for k, v in t.items()} for t in batch_prediction]
            prediction.extend(batch_prediction)
        model_time = time.time() - model_time

        # Adjust predictions due to grid splitting
        top = 0
        i = 0
        while top + overlap_height < height:
            left = 0
            while left + overlap_width < width:
                # print(i, width, height, overlap_height, overlap_width, top, left)
                # Reject boxes near overlapping edges
                prediction[i] = reject_boxes(prediction[i], top, left, width, height)
                # Adjust box coordinates based on the crop position
                prediction[i] = adjust_boxes(prediction[i], top, left)
                left += int(crop_size - overlap_width)
                i += 1
            top += int(crop_size - overlap_height)

        # Combine crops into single prediction
        prediction = {
            'boxes': torch.cat([item['boxes'] for item in prediction]),
            'labels': torch.cat([item['labels'] for item in prediction]),
            'scores': torch.cat([item['scores'] for item in prediction]),
            'class_scores': torch.cat([item['class_scores'] for item in prediction])}

        # Non-maximum supression
        ids = torch.tensor([1]*len(prediction['labels']))
        # ids = prediction['labels'] #TEST
        inds = torchvision.ops.boxes.batched_nms(prediction['boxes'], prediction['scores'], ids, iou_threshold = kwargs["box_nms_thresh"])
        prediction['boxes'] = prediction['boxes'][inds]
        prediction['labels'] = prediction['labels'][inds]
        prediction['scores'] = prediction['scores'][inds]
        prediction['class_scores'] = prediction['class_scores'][inds]

        # Remove low scoring boxes for bird vs background
        inds = torch.where(torch.sum(prediction['class_scores'], 1) > kwargs["bird_score_thresh"])[0]
        prediction['boxes'] = prediction['boxes'][inds]
        prediction['labels'] = prediction['labels'][inds]
        prediction['scores'] = prediction['scores'][inds]
        prediction['class_scores'] = prediction['class_scores'][inds]

        # Remove low scoring boxes for class score
        inds = torch.where(prediction['scores'] > kwargs["class_score_thresh"])[0]
        prediction['boxes'] = prediction['boxes'][inds]
        prediction['labels'] = prediction['labels'][inds]
        prediction['scores'] = prediction['scores'][inds]
        prediction['class_scores'] = prediction['class_scores'][inds]
    
        # Rescale crops to original scale
        rescale_boxes(prediction, scale)

        # Create label file
        create_annotation(prediction, image, filename)

        # Add to csv
        create_csv(prediction, filename)
        
        metric_logger.update(model_time=model_time)

# Setup
Image.MAX_IMAGE_PIXELS = 1000000000
output_dir = "C:/Users/uqjwil54/Documents/Projects/DBBD/outputs/"
# output_dir = "outputs/"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.mkdir(output_dir)

kwargs = {
    "rpn_pre_nms_top_n_test": 250,
    "rpn_post_nms_top_n_test": 250,
    "rpn_nms_thresh": 0.5,
    "rpn_score_thresh": 0.5,
    "box_detections_per_img": 100,
    "box_nms_thresh": 0.2,
    "bird_score_thresh": 0.75,
    "class_score_thresh": 0.0}

img_dir = "C:/Users/uqjwil54/Documents/Projects/DBBD/images/"
model_path = "C:/Users/uqjwil54/Documents/Projects/DBBD/balanced-2024_05_01/model-2024_05_06/"
# img_dir = "images/"
# model_path = "models/bird-2024_04_29/"
model_name = "FasterRCNN"
backbone_name = "resnet101"
checkpoint_name = "model.pth"
device = torch.device("cuda")
crop_size = 800
overlap = 200
margin = 5
target_gsd = 0.005
batchsize = 1


index_to_class = json.load(open(os.path.join(model_path, "index_to_class.json")))
n_classes = len(index_to_class) + 1

categories = {}
for key in index_to_class["1"].keys():
    categories[key] = [class_info[key] for class_info in index_to_class.values()]

included_classes = categories["name"]

class_filter = torch.zeros(len(index_to_class) + 1)
class_filter[0] = 1
for i in range(len(categories["name"])):
    species = categories["name"][i]
    if species in included_classes:
        class_filter[i + 1] = 1

main()