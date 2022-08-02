import torch
import torchvision
import torch.utils.data
import json
import PIL
import os
import transforms as T
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers

class CustomDataloader(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.transforms = transforms
        self.root = root
        with open(os.path.join(self.root,"dataset.json")) as anns:
            self.dictionary = json.load(anns)

        self.file_names = []
        for images in self.dictionary['images']:
            self.file_names.append(images['file_name'])

        self.iscrowds = []
        self.image_ids = []
        self.bboxes = []
        self.category_ids = []
        self.area = []
        for annotations in self.dictionary['annotations']:
            self.iscrowds.append(annotations['iscrowd'])
            self.image_ids.append(annotations['image_id'])
            self.bboxes.append(annotations['bbox'])
            # self.category_ids.append(annotations['category_id'])  # bird species
            self.category_ids.append(1) # birds
            self.area.append(annotations['area'])
        
    def __getitem__(self, index):
        start_index = self.image_ids.index(index + 1)
        end_index = self.image_ids.index(index + 2)
        
        boxes = []
        for i in range(start_index, end_index):
            xmin = self.bboxes[i][0]
            xmax = xmin + self.bboxes[i][2]
            ymin = self.bboxes[i][1]
            ymax = ymin + self.bboxes[i][3]
            boxes.append([xmin, ymin, xmax, ymax])

        iscrowd = torch.zeros((end_index - start_index,), dtype=torch.int64)
        image_id = torch.tensor([index + 1])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(self.category_ids[start_index:end_index])
        area = torch.tensor(self.area[start_index:end_index])

        target = {}
        target["iscrowd"] = iscrowd
        target["image_id"] = image_id
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area

        image_path = os.path.join(self.root, self.file_names[index])
        image = PIL.Image.open(image_path).convert('RGB')
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self):
        return len(self.file_names) - 1

class FRCNNObjectDetector(torchvision.models.detection.faster_rcnn.FasterRCNN):
    # def __init__(self, pretrained_backbone=True, trainable_backbone_layers = 5, num_classes = 3): # seed detector
    # def __init__(self, pretrained_backbone=True, trainable_backbone_layers = 5, num_classes = 18): # bird species detector
    def __init__(self, pretrained_backbone=True, trainable_backbone_layers = 5, num_classes = 2): # bird detector
        trainable_backbone_layers = _validate_trainable_layers(pretrained_backbone, trainable_backbone_layers, 5, 3)
        backbone = resnet_fpn_backbone('resnet101', pretrained_backbone, trainable_layers=trainable_backbone_layers)
        box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(1024, num_classes)
        # anchor_sizes = ((100,), (100,), (100,), (100,), (100,))
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(anchor_sizes, aspect_ratios)
        parameters = {
            'backbone': backbone,
            # 'num_classes': none, This is defined in the box_predictor

            'min_size': (0,), 
            # You need to update GeneralizedRCNNTransforms in torchvision/models/detection/transforms.py line 160 
            # to not resize if this is set to 0 for this to work:

            # JW addition, don't resize image if min_size = (0,)    
            # if self.min_size[-1] != 0:
                # image, target = _resize_image_and_masks(image, size, float(self.max_size), target, self.fixed_size)

            # 'max_size': 6000,
            # 'image_mean': None,
            # 'image_std': None,
            # 'rpn_anchor_generator': rpn_anchor_generator,
            # 'rpn_head': None,
            # 'rpn_pre_nms_top_n_train': 2000,
            'rpn_pre_nms_top_n_test': 2000,
            # 'rpn_post_nms_top_n_train': 2000,
            'rpn_post_nms_top_n_test': 2000,
            'rpn_nms_thresh': 1.0, # Test
            # 'rpn_fg_iou_thresh': 0.7,
            # 'rpn_bg_iou_thresh': 0.3,
            'rpn_batch_size_per_image': 2000,
            # 'rpn_positive_fraction': 0.5,
            'rpn_score_thresh': 0.0, # Test
            # 'box_roi_pool': roi_pooler,
            # 'box_head': None,
            'box_predictor': box_predictor,
            'box_score_thresh': 0.1, # Test
            'box_nms_thresh': 0.1, # Test
            'box_detections_per_img': 2000,
            # 'box_fg_iou_thresh': 0.5,
            # 'box_bg_iou_thresh': 0.3,
            'box_batch_size_per_image': 2000
            # 'box_positive_fraction': 0.25,
            # 'bbox_reg_weights': None
            }
        super(FRCNNObjectDetector, self).__init__(**parameters)

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomIoUCrop())
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_dataset():
    dataset = CustomDataloader('../../dataset/seed-detector/train', get_transform(train=True))
    dataset_test = CustomDataloader('../../dataset/seed-detector/train', get_transform(train=False))
    return dataset, dataset_test

