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

def prepare_image(image_path):
    image = PIL.Image.open(image_path).convert('RGB')
    image = torchvision.transforms.PILToTensor()(image)
    image = torchvision.transforms.ConvertImageDtype(torch.float)(image)
    # image = torchvision.transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    return image

def draw_boxes(image_path, model):
    image = prepare_image(image_path)
    with torch.no_grad():
        prediction = model(image)
    image = read_image(image_path)
    mask_threshold = 0.5
    scores = prediction[0]['scores'].tolist()
    labels = prediction[0]["labels"].tolist()
    boxes = prediction[0]['boxes']
    masks = prediction[0]['masks'] > mask_threshold
    string_scores = ['{0:.2f}'.format(score) for score in scores]
    named_labels = [index_to_class[str(i)] for i in labels]
    named_labels_with_scores = [named_labels[i] + ": " + string_scores[i] for i in range(len(scores))]
    ouput_image = torchvision.utils.draw_bounding_boxes(image, boxes = boxes, labels = named_labels_with_scores)
    ouput_image = torchvision.utils.draw_segmentation_masks(ouput_image, masks.squeeze(1), alpha=0.5)
    ouput_image = torchvision.transforms.ToPILImage()(ouput_image)
    ouput_image.show()

os.chdir(os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cuda")
# device = torch.device("cpu")
# model = custom_detector.get_model()
kwargs = {
    'box_score_thresh': 0.9,
    'box_nms_thresh': 0.3}

model = torchvision.models.detection.__dict__["maskrcnn_resnet50_fpn"](weights=None, weights_backbone=None, num_classes=3, **kwargs)
model_root = "../../models/temp"
model_path = model_root + "/model_final_state_dict.pth"
index_path = model_root + "/index_to_class.txt"
model.load_state_dict(torch.load(model_path, map_location=device))
index_to_class = json.load(open(index_path))
model.eval()
model = model.to(device)
prediction_1 = draw_boxes("../../datasets/seed-detector-coco/val2017/43791b12fa969f4e9fb34f42b59ca3d2.JPG", model)
prediction_1

# for converting checkpoint
# model_path = "../../models/temp/model_249.pth"
# checkpoint = torch.load(model_path, map_location="cpu")
# model.load_state_dict(checkpoint["model"])
# utils.save_on_master(model.state_dict(), "../../models/temp/model_final_state_dict.pth")