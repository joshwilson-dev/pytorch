import torch
import torchvision
import custom_detector
import PIL
import os
import importlib
importlib.reload(custom_detector)
import json
from torchvision.io import read_image
import utils
import math
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

patch_size = 1333
overlap = 150

def prepare_image(image_path, device):
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
    # check_patch = PIL.ImageDraw.Draw(image)  
    for height_index in range(n_crops_height):
        for width_index in range(n_crops_width):
            left = width_index * (patch_size - overlap)
            right = left + patch_size
            top = height_index * (patch_size - overlap)
            bottom = top + patch_size
            patch = image.crop((left, top, right, bottom))
            # check_patch.rectangle((left, top, right, bottom), outline ="red")
            patches.append(patch)
    # image.save("trialinput.png")
    batch = torch.empty(0, 3, patch_size, patch_size).to(device)
    for patch in patches:
        patch = torchvision.transforms.PILToTensor()(patch)
        patch = torchvision.transforms.ConvertImageDtype(torch.float)(patch)
        patch = patch.unsqueeze(0)
        patch = patch.to(device)
        batch = torch.cat((batch, patch), 0)
    return batch, pad_width, pad_height, n_crops_height, n_crops_width

def draw_boxes(image_path, model, device):
    batch, pad_width, pad_height, n_crops_height, n_crops_width = prepare_image(image_path, device)
    with torch.no_grad():
        prediction = model(batch)
    boxes = torch.empty(0, 4).to(device)
    scores = torch.empty(0).to(device)
    labels = torch.empty(0, dtype=torch.int64).to(device)
    # masks =
    for height_index in range(n_crops_height):
        for width_index in range(n_crops_width):
            patch_index = height_index * n_crops_width + width_index
            x_adj = (patch_size - overlap) * width_index - pad_width
            y_adj = (patch_size - overlap) * height_index - pad_height
            adjustment = torch.tensor([[x_adj, y_adj, x_adj, y_adj]]).to(device)
            adj_boxes = torch.add(adjustment, prediction[patch_index]["boxes"])
            boxes = torch.cat((boxes, adj_boxes), 0)
            scores = torch.cat((scores, prediction[patch_index]["scores"]), 0)
            labels = torch.cat((labels, prediction[patch_index]["labels"]), 0)
            # masks = 
    nms_indices = torchvision.ops.nms(boxes, scores, 0.1)
    boxes = boxes[nms_indices]
    scores = scores[nms_indices].tolist()
    labels = labels[nms_indices].tolist()
    image = read_image(image_path)
    # mask_threshold = 0.5
    # masks = prediction[0]['masks'] > mask_threshold
    string_scores = ['{0:.2f}'.format(score) for score in scores]
    named_labels = [index_to_class[str(i)] for i in labels]
    named_labels_with_scores = [named_labels[i] + ": " + string_scores[i] for i in range(len(scores))]
    ouput_image = torchvision.utils.draw_bounding_boxes(image, boxes = boxes, labels = named_labels_with_scores)
    # ouput_image = torchvision.utils.draw_segmentation_masks(ouput_image, masks.squeeze(1), alpha=0.5)
    ouput_image = torchvision.transforms.ToPILImage()(ouput_image)
    ouput_image.show()

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# device = torch.device("cpu")
device = torch.device("cuda")
# model = custom_detector.get_model()
kwargs = {
    "rpn_pre_nms_top_n_test": 1000,
    "rpn_post_nms_top_n_test": 1000,
    "rpn_nms_thresh": 0.7,
    "rpn_score_thresh": 0.0,
    "box_score_thresh": 0.9,
    "box_nms_thresh": 0.1,
    "box_detections_per_img": 1000}

# model = torchvision.models.detection.__dict__["maskrcnn_resnet50_fpn"](weights=None, weights_backbone=None, num_classes=3, **kwargs)
# model = torchvision.models.detection.__dict__["fasterrcnn_resnet50_fpn"](num_classes=2, **kwargs)
backbone = resnet_fpn_backbone(backbone_name = 'resnet101', weights = "ResNet101_Weights.DEFAULT")
box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(backbone.out_channels * 4, 2)
model = torchvision.models.detection.__dict__["FasterRCNN"](box_predictor = box_predictor, backbone = backbone, **kwargs)
model_root = "../../models/temp"
model_path = model_root + "/model_final_state_dict.pth"
index_path = model_root + "/index_to_class.txt"
model.load_state_dict(torch.load(model_path, map_location=device))
index_to_class = json.load(open(index_path))
model.eval()
model = model.to(device)
prediction_1 = draw_boxes("../../datasets/bird-detector-coco/original/0a43172c9392d9f3671a0cb620ef2e5d.JPG", model, device)
# for converting checkpoint
# model_path = "../../models/temp/model_249.pth"
# checkpoint = torch.load(model_path, map_location="cpu")
# model.load_state_dict(checkpoint["model"])
# utils.save_on_master(model.state_dict(), "../../models/temp/model_final_state_dict.pth")