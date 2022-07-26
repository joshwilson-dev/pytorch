r"""PyTorch Detection Training.
To run in a multi-gpu environment, use the distributed launcher::
    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU
The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.
On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3
Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""
import datetime
import os
import time
import csv
import presets
import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
import utils
from coco_utils import get_coco, get_coco_kp
from engine import train_one_epoch, evaluate
from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from torchvision.transforms import InterpolationMode
from transforms import SimpleCopyPaste
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator

def copypaste_collate_fn(batch):
    copypaste = SimpleCopyPaste(blending=True, resize_interpolation=InterpolationMode.BILINEAR)
    return copypaste(*utils.collate_fn(batch))

def get_dataset(name, image_set, transform, data_path, num_classes):
    paths = {"coco": (data_path, get_coco, num_classes), "coco_kp": (data_path, get_coco_kp, 2)}
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train, args):
    if train:
        return presets.DetectionPresetTrain(data_augmentation=args.data_augmentation)
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()
        return lambda img, target: (trans(img), target)
    else:
        return presets.DetectionPresetEval()


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--data-path", default="/datasets01/COCO/022719/", type=str, help="dataset path")
    parser.add_argument("--dataset", default="coco", type=str, help="dataset name")
    parser.add_argument("--model", default="maskrcnn_resnet50_fpn", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=26, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument(
        "--lr",
        default=0.02,
        type=float,
        help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--lr-scheduler", default="multisteplr", type=str, help="name of lr scheduler (default: multisteplr)"
    )
    parser.add_argument(
        "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
    parser.add_argument(
        "--trainable-backbone-layers", default=None, type=int, help="number of trainable layers of backbone"
    )
    parser.add_argument(
        "--data-augmentation", default="hflip", type=str, help="data augmentation policy (default: hflip)"
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone", default=None, type=str, help="the backbone weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # Use CopyPaste augmentation training parameter
    parser.add_argument(
        "--use-copypaste",
        action="store_true",
        help="Use CopyPaste data augmentation. Works only with data-augmentation='lsj'.",
    )

    # Josh Wilson additions 01/07/2021
    parser.add_argument('--custommodel', default = 0, type = int, help = 'Should we use a custom model?')
    parser.add_argument('--numclasses', default = 91, type = int, help = "How many classes are there?")
    parser.add_argument('--patience', default = 5, type = int, help = "How many epochs without improvement before action?")
    parser.add_argument('--backbone', default = "resnet50", type = str, help = "Which backbone do you want to use?")
    parser.add_argument('--box_positive_fraction', default = 0.25, type = float, help = "Proportion of positive proposals in a mini-batch during training of the classification head")
    # Josh Wilson additions 01/07/2021

    return parser

def save_eval(results):
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [all, small, medium, large] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    #  score      - [TxRxKxAxM] score for every evaluation setting
    iouThrs = results.coco_eval["bbox"].eval["params"].iouThrs
    recThrs = results.coco_eval["bbox"].eval["params"].recThrs
    catIds = results.coco_eval["bbox"].eval["params"].catIds
    areaRng = results.coco_eval["bbox"].eval["params"].areaRng
    maxDets = results.coco_eval["bbox"].eval["params"].maxDets
    result_dict = {'iouThr': [], 'recThr': [], 'catId': [], 'area': [], 'maxDet': [], 'precision': [], 'scores': []}
    for iouThr_index in range(len(iouThrs)):
        for recThr_index in range(len(recThrs)):
            for catId_index in range(len(catIds)):
                for area_index in range(len(areaRng)):
                    for maxDet_index in range(len(maxDets)):
                        result_dict['iouThr'].append(iouThrs[iouThr_index])
                        result_dict['recThr'].append(recThrs[recThr_index])
                        result_dict['catId'].append(catIds[catId_index])
                        result_dict['area'].append(areaRng[area_index])
                        result_dict['maxDet'].append(maxDets[maxDet_index])
                        result_dict['precision'].append(results.coco_eval["bbox"].eval["precision"][iouThr_index, recThr_index, catId_index, area_index, maxDet_index])
                        result_dict['scores'].append(results.coco_eval["bbox"].eval["scores"][iouThr_index, recThr_index, catId_index, area_index, maxDet_index])
    with open(os.path.join(args.output_dir, "performance_metrics.csv"), "w", newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(result_dict.keys())
        writer.writerows(zip(*result_dict.values()))
    return

def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    # Data loading code
    print("Loading data")
    
    # Josh Wilson
    dataset, num_classes = get_dataset(args.dataset, "train", get_transform(True, args), args.data_path, args.numclasses)
    dataset_test, _ = get_dataset(args.dataset, "train", get_transform(False, args), args.data_path, args.numclasses)
    train_size = int(0.75 * len(dataset))
    test_size = len(dataset) - train_size
    dataset, _ = torch.utils.data.random_split(dataset, [train_size, test_size])
    _, dataset_test = torch.utils.data.random_split(dataset_test, [train_size, test_size])
    # _, dataset_test = torch.utils.data.random_split(dataset_test, [0, len(dataset_test)])
    # Josh Wilson
    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    train_collate_fn = utils.collate_fn
    if args.use_copypaste:
        if args.data_augmentation != "lsj":
            raise RuntimeError("SimpleCopyPaste algorithm currently only supports the 'lsj' data augmentation policies")

        train_collate_fn = copypaste_collate_fn

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers, collate_fn=train_collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    print("Creating model")

    # Josh Wilson additions 01/07/2021
    if args.custommodel == 1:
        # need to check if its fpn or not
        anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        kwargs = {
            "rpn_anchor_generator": rpn_anchor_generator,
            "box_positive_fraction": args.box_positive_fraction}
        backbone = resnet_fpn_backbone(backbone_name = args.backbone, weights=args.weights_backbone, trainable_layers=args.trainable_backbone_layers)
        box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(backbone.out_channels * 4, num_classes)
        model = torchvision.models.detection.__dict__[args.model](box_predictor = box_predictor, backbone = backbone, **kwargs)
    # Josh Wilson additions 01/07/2021
    else:
        kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers}
        if args.data_augmentation in ["multiscale", "lsj"]:
            kwargs["_skip_resize"] = True
        if "rcnn" in args.model:
            if args.rpn_score_thresh is not None:
                kwargs["rpn_score_thresh"] = args.rpn_score_thresh
        model = torchvision.models.detection.__dict__[args.model](
            weights=args.weights, weights_backbone=args.weights_backbone, num_classes = args.numclasses, **kwargs
        )
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.norm_weight_decay is None:
        parameters = [p for p in model.parameters() if p.requires_grad]
    else:
        param_groups = torchvision.ops._utils.split_normalization_params(model)
        wd_groups = [args.norm_weight_decay, args.weight_decay]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        torch.backends.cudnn.deterministic = True
        results = evaluate(model, data_loader_test, device=device)
        save_eval(results)

    # Josh Wilson
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir = args.output_dir)
    epochs_without_improvement = 0
    lr_steps = 0
    best_mAP = 0
    # Josh Wilson
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq, scaler)
        # lr_scheduler.step()
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            # utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))

        # Josh Wilson
        results = evaluate(model, data_loader_test, device=device)
        mAP = results.coco_eval["bbox"].stats[0]
        writer.add_scalar("mAP", mAP, epoch)
        if mAP > best_mAP:
            print("The model improved this epoch")
            # save best model and state dict
            print("Updating best model & results")
            utils.save_on_master(checkpoint,os.path.join(args.output_dir, 'model_best_checkpoint.pth'))
            utils.save_on_master(model_without_ddp.state_dict(),os.path.join(args.output_dir, 'model_best_state_dict.pth'))
            save_eval(results)
            best_mAP = mAP
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print("The model has not improved for {} epochs...".format(epochs_without_improvement))
        if epochs_without_improvement == args.patience:
            print("{} epochs without improvement...".format(args.patience))
            # load best model checkpoint
            checkpoint = torch.load(os.path.join(args.output_dir, 'model_best_checkpoint.pth'), map_location="cpu")
            model_without_ddp.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            # decrease the learning rate, unless at last lr, then stop
            if lr_steps < len(args.lr_steps) - 1:
                print("Decreasing learning rate...")
                lr_scheduler.step()
                lr_steps += 1
                epochs_without_improvement = 0
            else:
                writer.flush()
                break
        # Josh Wilson

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)