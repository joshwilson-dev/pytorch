import time

import torch
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

@torch.inference_mode()
def evaluate(model, data_loader, device, **kwargs):
    n_threads = torch.get_num_threads()

    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)

    # Update iou_types
    if 'iou_types' in kwargs:
        iou_types = kwargs.get("iou_types")
    else:
        iou_types = ['bbox']

    # Create evaluator
    coco_evaluator = CocoEvaluator(coco, iou_types)

    # Update area ranges
    if 'areaRng' in kwargs:
        for iou_type in iou_types:
            coco_evaluator.coco_eval[iou_type].params.areaRng = kwargs.get("areaRng")
    
    if 'areaRngLbl' in kwargs:
        for iou_type in iou_types:
            coco_evaluator.coco_eval[iou_type].params.areaRngLbl = kwargs.get("areaRngLbl")

    # Update maxDets
    if 'maxDets' in kwargs:
        for iou_type in iou_types:
            coco_evaluator.coco_eval[iou_type].params.maxDets = kwargs.get("maxDets")
    
    # Update iouThr
    if 'iouThrs' in kwargs:
        for iou_type in iou_types:
            coco_evaluator.coco_eval[iou_type].params.iouThrs = kwargs.get("iouThrs")
    
    # Update use cats
    if 'useCats' in kwargs:
        for iou_type in iou_types:
            coco_evaluator.coco_eval[iou_type].params.useCats = kwargs.get("useCats")

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

