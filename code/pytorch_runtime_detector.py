import torch
import torchvision
import PIL
import os
import json
from torchvision.io import read_image
import utils
import math
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="Pytorch to Labelme Auto-labeller", add_help=add_help)
    parser.add_argument("--backbone", default="resnet50", type=str, help="backbone architecture")
    parser.add_argument("--model", default="MaskRCNN", type=str, help="FasterRCNN or MaskRCNN?")
    parser.add_argument("--root", default="../models/temp/", type=str, help="Path to root")
    parser.add_argument("--device", default="cpu", type=str, help="cpu or cuda")
    parser.add_argument("--image", default="../datasets/seed-box/original/Pan - 21.JPG", type=str, help="Path to image")
    parser.add_argument("--convert", default="False", type=str, help="Do you want to convert a checkpoint to a state dict?")
    return parser

patch_size = 1333
overlap = 150

def create_model(device, index_to_class):
    num_classes = len(index_to_class) + 1
    kwargs = json.load(open(args.root + "/kwargs.txt"))
    backbone = resnet_fpn_backbone(backbone_name = args.backbone, weights = "DEFAULT")
    box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(backbone.out_channels * 4, num_classes)
    if args.model == "MaskRCNN":
        mask_predictor_in_channels = 256
        mask_dim_reduced = 256
        mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(mask_predictor_in_channels, mask_dim_reduced, num_classes)
        kwargs["mask_predictor"] = mask_predictor
    model = torchvision.models.detection.__dict__[args.model](box_predictor = box_predictor, backbone = backbone, **kwargs)
    return model

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
    return width, height, batch, pad_width, pad_height, n_crops_height, n_crops_width

def draw_boxes(index_to_class, image_path, model, device):
    width, height, batch, pad_width, pad_height, n_crops_height, n_crops_width = prepare_image(image_path, device)
    with torch.no_grad():
        prediction = model(batch)
    boxes = torch.empty(0, 4).to(device)
    scores = torch.empty(0).to(device)
    labels = torch.empty(0, dtype=torch.int64).to(device)
    masks = torch.empty(0, 1, height, width).to(device)
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
            if args.model == "MaskRCNN":
                # pad masks to full image
                padding_right = width - padding_left - patch_size
                padding_bottom = height - padding_top - patch_size
                padding = (int(math.ceil(padding_left)), int(math.floor(padding_right)), int(math.ceil(padding_top)), int(math.floor(padding_bottom)))
                padded_masks = torch.nn.functional.pad(prediction[patch_index]["masks"], padding, "constant", 0)
                masks = torch.cat((masks, padded_masks), 0)
    nms_indices = torchvision.ops.nms(boxes, scores, 0.1)
    boxes = boxes[nms_indices]
    scores = scores[nms_indices].tolist()
    labels = labels[nms_indices].tolist()
    image = read_image(image_path)
    string_scores = ['{0:.2f}'.format(score) for score in scores]
    named_labels = [index_to_class[str(i)] for i in labels]
    named_labels_with_scores = [named_labels[i] + ": " + string_scores[i] for i in range(len(scores))]
    ouput_image = torchvision.utils.draw_bounding_boxes(image, boxes = boxes, labels = named_labels_with_scores)
    if args.model == "MaskRCNN":
        mask_threshold = 0.5
        masks = masks[nms_indices] > mask_threshold
        ouput_image = torchvision.utils.draw_segmentation_masks(ouput_image, masks.squeeze(1), alpha=0.5)
    ouput_image = torchvision.transforms.ToPILImage()(ouput_image)
    ouput_image.show()

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device(args.device)
    index_to_class = json.load(open(args.root + "index_to_class.txt"))
    model = create_model(device, index_to_class)
    if args.convert == "True":
        state_dict_path = os.path.join(args.root, "model_59.pth")
        checkpoint = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        utils.save_on_master(model.state_dict(), args.root + "/model_final_state_dict.pth")
    else:
        model.load_state_dict(torch.load(args.root + "/model_final_state_dict.pth", map_location=device))
        model.eval()
        model = model.to(device)
        draw_boxes(index_to_class, args.image, model, device)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main()