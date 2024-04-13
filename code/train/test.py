import os
import csv

import torch
import torchvision
import utils
import transforms
import coco_utils
import evaluate
import pandas as pd
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

data_path = "data/balanced"
output_dir = "models/bird_2024_04_04"
dataset = "test"
img_folder = os.path.join(data_path, dataset)
ann_file = os.path.join(data_path, "annotations", dataset + ".json")
checkpoint_name = os.path.join(output_dir, "model.pth")
model_name = "FasterRCNN"
backbone_name = "resnet101"
n_classes = 94

min_dim = 0
max_dim = 525
steps = 21
step_size = int((max_dim - min_dim)/steps)
dims = range(min_dim, max_dim, step_size)
areaRng = [[0, int(1e5**2)]] + [[dim, dim + step_size] for dim in dims]
areaRngLbl = ['all'] + [str(dim[1]) for dim in areaRng]
areaRng = [[0, int(1e5**2)]]
areaRngLbl = ['all']
kwargs = {
    "iou_types": ['bbox'],
    "useCats": 0,
    "maxDets": [100],
    "iouThrs": [0.5],
    "areaRng": areaRng,
    "areaRngLbl": areaRngLbl
    }

def save_eval(results):
    result_dict = {'iou_type': [], 'iouThr': [], 'recThr': [], 'catId': [], 'area': [], 'maxDet': [], 'precision': [], 'recall': [], 'scores': []}
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
                            result_dict['precision'].append(precision[i, r, c, a, m])
                            result_dict['recall'].append(recall[i, c, a, m])
                            result_dict['scores'].append(scores[i, r, c, a, m])
                            result_dict['iou_type'].append(iou_type)
    with open(os.path.join(output_dir, "eval-0.csv"), "w", newline='') as outfile:
        writer = csv.writer(outfile)
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
                        iou_n = 0
                        iou_max = 0
                        iou_match = 0
                    else:
                        iou_n = sum(entry['ious'][d, :] > params.iouThrs)
                        iou_max = max(entry['ious'][d, :])
                        iou_match = entry['gtIds'][max(enumerate(entry['ious'][d]), key=lambda x: x[1])[0]]

                    image_id_results = {
                        'image_name': dataset_test.coco.loadImgs(int(entry['image_id']))[0]["file_name"],
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
                        'iou_n': iou_n,
                        'iou_match': iou_match,
                        'iou_max': iou_max,
                    }
                    image_results.append(image_id_results)

            for g in range(len(entry['gtIds'])):
                for i in range(len(params.iouThrs)):
                    if type(entry['ious']) == list:
                        iou_n = 0
                        iou_max = 0
                        iou_match = ""
                    else:
                        iou_n = sum(entry['ious'][:, g] > params.iouThrs)
                        iou_max = max(entry['ious'][:, g])
                        # iou_match = entry['dtIds'][max(enumerate(entry['ious'][g]), key=lambda x: x[1])[0]]
                        iou_match = ""

                    image_id_results = {
                        'image_name': dataset_test.coco.loadImgs(int(entry['image_id']))[0]["file_name"],
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
                        'iou_n': iou_n,
                        'iou_match': iou_match,
                        'iou_max': iou_max,
                    }
                    image_results.append(image_id_results)
    df = pd.DataFrame(image_results)
    df.to_csv(os.path.join(output_dir, "evalimgs-0.csv"))
    return

def main():
    # Setup
    device = torch.device("cuda")

    # Load the test and train datasets
    print("Loading data")
    
    test_transforms = transforms.Compose([
        coco_utils.ConvertCocoPolysToMask(),
        transforms.PILToTensor(),
        transforms.ToDtype(torch.float, scale=True)])
    
    dataset_test = coco_utils.CocoDetection(
        img_folder = img_folder,
        ann_file = ann_file,
        transforms = test_transforms)

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=test_sampler,
        collate_fn = utils.collate_fn)

    # Create the model
    print("Creating model")
    backbone = resnet_fpn_backbone(backbone_name = backbone_name, weights = None)
    model = torchvision.models.detection.__dict__[model_name](backbone = backbone, num_classes = n_classes)
    
    model.to(device)

    checkpoint = torch.load(checkpoint_name, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    torch.backends.cudnn.deterministic = True

    # Testing model
    results = evaluate.evaluate(
        model,
        data_loader_test,
        device,
        **kwargs)
    save_evalImgs(results, dataset_test)
    save_eval(results)

main()