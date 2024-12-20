# Title:        Test performance
# Description:  Evaluated the performance of on object detection network using
#               COCO Evaluation Metrics
# Author:       Anonymous
# Date:         05/06/2024

#### Import packages
# Base packages
import os
import csv
import time

# External packages
import torch
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import pandas as pd

# Custom packages 
import CS9_utils
import CS11_transforms
import CS12_coco_utils
import CS13_evaluate

# Setup
data_root = "Detecting_and_identifying_birds_in_images_captured_using_a_drone"
data_path = os.path.join(data_root, "balanced")
output_dir = ""
dataset = "test"
img_folder = os.path.join(data_path, dataset)
ann_file = os.path.join(data_path, "annotations", dataset + ".json")
checkpoint_name = os.path.join(output_dir, "AS2_model.pth")
model_name = "FasterRCNN"
backbone_name = "resnet101"
n_classes = 97

min_dim = 0
max_dim = 5000
steps = 10
step_size = int((max_dim - min_dim)/steps)
dims = range(min_dim, max_dim, step_size)
areaRng = [[0, int(1e5**2)]] + \
    [[dim, dim + step_size] for dim in dims] + \
    [[max_dim, int(1e5**2)]]
areaRngLbl = ['all'] + [str(dim[1]) for dim in areaRng[1:]]
# areaRng = [[0, int(1e5**2)]]
# areaRngLbl = ['all']

kwargs = {
    "iou_types": ['bbox'],
    "useCats": 0,
    "maxDets": [1, 10, 100],
    "iouThrs": [0.5, 0.75],
    "areaRng": areaRng,
    "areaRngLbl": areaRngLbl
    }

def save_eval(results):
    result_dict = {
        'iou_type': [],
        'iouThr': [],
        'recThr': [],
        'catId': [],
        'area': [],
        'maxDet': [],
        'precision': [],
        'recall': [],
        'scores': []}
    for iou_type in results.iou_types:
        params = results.coco_eval[iou_type].eval["params"]
        precision = results.coco_eval[iou_type].eval["precision"]
        recall = results.coco_eval[iou_type].eval["recall"]
        scores = results.coco_eval[iou_type].eval["scores"]
        for i in range(len(params.iouThrs)):
            for r in range(len(params.recThrs)):
                for c in range(len(params.catIds)):
                    for a in range(len(params.areaRng)):
                        for m in range(len(params.maxDets)):
                            result_dict['iouThr'].append(params.iouThrs[i])
                            result_dict['recThr'].append(params.recThrs[r])
                            result_dict['catId'].append(params.catIds[c])
                            result_dict['area'].append(params.areaRng[a])
                            result_dict['maxDet'].append(params.maxDets[m])
                            result_dict['precision'].append(precision[
                                i, r, c, a, m])
                            result_dict['recall'].append(recall[i, c, a, m])
                            result_dict['scores'].append(scores[i, r, c, a, m])
                            result_dict['iou_type'].append(iou_type)
    with open(os.path.join(output_dir, "eval-1.csv"), "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(result_dict.keys())
        writer.writerows(zip(*result_dict.values()))
    return

def save_evalImgs(results, dataset_test):
    image_results = []
    
    for iou_type in results.iou_types:
        evalImgs = results.coco_eval[iou_type].evalImgs
        params = results.coco_eval[iou_type].eval["params"]
        evalImgs = [d for d in evalImgs if d is not None]

        for n in range(len(evalImgs)):
            entry = evalImgs[n]
            for d in range(len(entry['dtIds'])):
                for i in range(len(params.iouThrs)):
                    if type(entry['ious']) == list:
                        iou_max = 0
                        iou_match = 0
                    else:
                        iou_max = max(entry['ious'][d, :])
                        iou_match = entry['gtIds'][max(
                            enumerate(entry['ious'][d]), key=lambda x: x[1])[0]]

                    image_id_results = {
                        'image_name': dataset_test.coco.loadImgs(
                            int(entry['image_id']))[0]["file_name"],
                        'image_id': entry['image_id'],
                        'iou_type': iou_type,
                        'category_id': entry['category_id'],
                        'aRng': entry['aRng'],
                        'maxDet': entry['maxDet'],
                        'det_type': 'dt',
                        'Ids': entry['dtIds'][d],
                        'iouThrs': params.iouThrs[i],
                        'Matches': entry['dtMatches'][i][d],
                        'Scores': entry['dtScores'][d],
                        'Ignore': entry['dtIgnore'][i][d],
                        'iou_match': iou_match,
                        'iou_max': iou_max,
                    }
                    image_results.append(image_id_results)

            for g in range(len(entry['gtIds'])):
                for i in range(len(params.iouThrs)):
                    if type(entry['ious']) == list:
                        iou_max = 0
                        iou_match = ""
                    else:
                        iou_max = max(entry['ious'][:, g])
                        iou_match = ""

                    image_id_results = {
                        'image_name': dataset_test.coco.loadImgs(
                            int(entry['image_id']))[0]["file_name"],
                        'image_id': entry['image_id'],
                        'iou_type': iou_type,
                        'category_id': entry['category_id'],
                        'aRng': entry['aRng'],
                        'maxDet': entry['maxDet'],
                        'det_type': 'gt',
                        'Ids': entry['gtIds'][g],
                        'iouThrs': params.iouThrs[i],
                        'Matches': entry['gtMatches'][i][g],
                        'Scores': 1,
                        'Ignore': entry['gtIgnore'][g],
                        'iou_match': iou_match,
                        'iou_max': iou_max,
                    }
                    image_results.append(image_id_results)
    df = pd.DataFrame(image_results)
    df.to_csv(os.path.join(output_dir, "evalimgs-1.csv"))
    return

def main():
    # Setup
    device = torch.device("cuda")

    # Load the test and train datasets
    print("Loading data")
    
    test_transforms = CS11_transforms.Compose([
        CS12_coco_utils.ConvertCocoPolysToMask(),
        CS11_transforms.PILToTensor(),
        CS11_transforms.ToDtype(torch.float, scale=True)])
    
    dataset_test = CS12_coco_utils.CocoDetection(
        img_folder = img_folder,
        ann_file = ann_file,
        transforms = test_transforms)

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=test_sampler,
        collate_fn = CS9_utils.collate_fn)

    # Create the model
    print("Creating model")

    backbone = resnet_fpn_backbone(
        backbone_name = backbone_name, weights = None)
    model = torchvision.models.detection.__dict__[model_name](
        backbone = backbone, num_classes = n_classes)
    
    model.to(device)

    checkpoint = torch.load(checkpoint_name, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    torch.backends.cudnn.deterministic = True

    # Testing model
    results = CS13_evaluate.evaluate(
        model,
        data_loader_test,
        device,
        **kwargs)
    save_evalImgs(results, dataset_test)
    save_eval(results)

start = time.time()
main()
end = time.time()
print("testing time: ", end - start, "(s)")