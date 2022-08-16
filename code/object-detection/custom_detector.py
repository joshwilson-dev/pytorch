import torch
from PIL import Image
import numpy as np
import os
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers
import transforms as T
import json

class CustomDataloader(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms

        with open(os.path.join(self.root,"dataset.json")) as anns:
            data = json.load(anns)
        
        self.annotations = {"image_id": [], "filename": [], "bboxes": [], "labels": [], "areas": [], "iscrowd": []}
        for image in data["images"]:
            self.annotations["image_id"].append(image["id"])
            self.annotations["filename"].append(os.path.splitext(image["file_name"])[0])
            bboxes = []
            labels = []
            areas = []
            for annotation in data["annotations"]:
                if annotation["image_id"] == image["id"]:
                    bbox = annotation["bbox"]
                    xmin = bbox[0]
                    xmax = xmin + bbox[2]
                    ymin = bbox[1]
                    ymax = ymin + bbox[3]
                    bboxes.append([xmin, ymin, xmax, ymax])
                    labels.append(annotation["category_id"])
                    areas.append((xmax - xmin) * (ymax - ymin))
            self.annotations["bboxes"].append(bboxes)
            self.annotations["labels"].append(labels)
            self.annotations["areas"].append(areas)
            self.annotations["iscrowd"].append(len(bboxes))

    def __getitem__(self, idx):
        filename = self.annotations["filename"][idx]
        # load images and masks
        img_path = os.path.join(self.root, filename + ".JPG")
        mask_path = os.path.join(self.root, filename + '-mask.png')
        image = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # convert everything into a torch.Tensor
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([self.annotations["image_id"][idx]])
        iscrowd = torch.zeros((self.annotations["iscrowd"][idx],), dtype=torch.int64)
        boxes = torch.as_tensor(self.annotations["bboxes"][idx], dtype=torch.float32)
        labels = torch.tensor(self.annotations["labels"][idx])
        area = torch.tensor(self.annotations["areas"][idx])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.annotations["image_id"])

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

class get_model(torchvision.models.detection.mask_rcnn.MaskRCNN):
# class get_model(torchvision.models.detection.faster_rcnn.FasterRCNN):
    def __init__(self, pretrained_backbone=True, trainable_backbone_layers = 5, num_classes = 3):
        trainable_backbone_layers = _validate_trainable_layers(pretrained_backbone, trainable_backbone_layers, 5, 3)
        backbone = resnet_fpn_backbone('resnet101', pretrained_backbone, trainable_layers=trainable_backbone_layers)
        box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(1024, num_classes)
        mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(256, 256, num_classes)
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(anchor_sizes, aspect_ratios)
        kwargs = {
            'min_size': (0,),
            'backbone': backbone,
            'rpn_anchor_generator': rpn_anchor_generator,
            'box_predictor': box_predictor,
            'mask_predictor': mask_predictor,
            'rpn_pre_nms_top_n_test': 10000,
            'rpn_post_nms_top_n_test': 10000,
            'rpn_batch_size_per_image': 10000,
            'rpn_nms_thresh': 1.0,
            'rpn_score_thresh': 0.0,
            'box_score_thresh': 0.4,
            'box_nms_thresh': 0.1,
            'box_detections_per_img': 10000}
        super(get_model, self).__init__(**kwargs)

def get_dataset():
    dataset = CustomDataloader('../../datasets/seed-detector/mask/crops/', get_transform(train=True))
    dataset_test = CustomDataloader('../../datasets/seed-detector/mask/crops/', get_transform(train=False))
    return dataset, dataset_test