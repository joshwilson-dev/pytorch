import datetime
import os
import time

import torch
import torchvision
import utils
import engine
import transforms
import coco_utils
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.utils.tensorboard import SummaryWriter

data_path = "data/balanced/"
data_path = "C:/Users/uqjwil54/Desktop/balanced-2024_05_01"
model_name = "FasterRCNN"
weights = None
backbone_name = "resnet101"
trainable_backbone_layers = 5
weights_backbone="ResNet101_Weights.IMAGENET1K_V1"
n_classes = 97
batch_size = 3
lr_steps = 4
lr = 0.001
momentum = 0.9
weight_decay = 1e-4
gamma = 0.1
patience = 5
output_dir = "models/temp"
output_dir = "C:/Users/uqjwil54/Desktop/models/temp"

def main():
    # Setup
    if output_dir:
        utils.mkdir(output_dir)

    device = torch.device("cuda")

    # Load the test and train datasets
    print("Loading data")

    # Create transform pipeline   
    train_transforms = transforms.Compose([
        coco_utils.ConvertCocoPolysToMask(),
        transforms.RandomPhotometricDistort(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotationNinety(360),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor(),
        transforms.ToDtype(torch.float, scale=True)])
    
    test_transforms = transforms.Compose([
        coco_utils.ConvertCocoPolysToMask(),
        transforms.PILToTensor(),
        transforms.ToDtype(torch.float, scale=True)])
    
    # Get the dataset
    dataset = coco_utils.CocoDetection(
        img_folder = os.path.join(data_path, "train"),
        ann_file = os.path.join(data_path, "annotations/train.json"),
        transforms=train_transforms)
    
    dataset_test = coco_utils.CocoDetection(
        img_folder = os.path.join(data_path, "test"),
        ann_file = os.path.join(data_path, "annotations/test.json"),
        transforms = test_transforms)

    train_sampler = torch.utils.data.RandomSampler(dataset)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler = train_batch_sampler,
        collate_fn = utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=test_sampler,
        collate_fn = utils.collate_fn)

    # Create the model
    print("Creating model")
    kwargs = {}

    backbone = resnet_fpn_backbone(
        backbone_name = backbone_name,
        trainable_layers = trainable_backbone_layers,
        weights=weights_backbone)
    
    model = torchvision.models.detection.__dict__[model_name](
        backbone = backbone,
        num_classes = n_classes,
        **kwargs)
    
    model.to(device)
    model_without_ddp = model

    # Define optimizer
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        parameters,
        lr = lr,
        momentum = momentum,
        weight_decay = weight_decay)

    # Define learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=range(1, lr_steps + 1),
        gamma=gamma)
    
    writer = SummaryWriter(log_dir = output_dir)
    epochs_without_improvement = 0
    lr_step = 0
    best_F1 = 0
    epoch = 0

    # Train model
    print("Start training")
    start_time = time.time()
    while lr_step < lr_steps - 1:

        # Train for one epoch
        engine.train_one_epoch(model, optimizer, data_loader, device, epoch, 20)
        
        # Evaluate model
        results = engine.evaluate(model, data_loader_test, device=device)
        ap = results.coco_eval["bbox"].stats[0]
        ap5 = results.coco_eval["bbox"].stats[1]
        ap75 = results.coco_eval["bbox"].stats[2]
        ar = results.coco_eval["bbox"].stats[8]
        f1 = (2 * ap * ar) / (ap + ar)
        writer.add_scalar("AP:0.5-0.95", ap, epoch)
        writer.add_scalar("AP:0.5", ap5, epoch)
        writer.add_scalar("AP:0.75", ap75, epoch)
        writer.add_scalar("AR:0.5-0.95", ar, epoch)
        writer.add_scalar("F1", f1, epoch)

        # Update best model if F1 score is better
        if f1 > best_F1:
            print("The model improved this epoch.")
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict()}
            utils.save_on_master(checkpoint, os.path.join(output_dir, 'model.pth'))
            best_F1 = f1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print("The model has not improved for {} epochs...".format(epochs_without_improvement))

        # Step lr or stop training if epochs without improvement == patience
        if epochs_without_improvement == patience:
            print("{} epochs without improvement...".format(patience))

            # Load best model checkpoint
            checkpoint = torch.load(os.path.join(output_dir, 'model.pth'), map_location="cpu")
            model_without_ddp.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

            # Decrease the learning rate, unless at last lr, then stop
            print("Decreasing learning rate...")
            lr_scheduler.step()
            lr_step += 1
            epochs_without_improvement = 0
        epoch += 1
    writer.flush()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

main()