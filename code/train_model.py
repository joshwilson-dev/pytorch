import os

# Create the detection dataset
# os.system("python code/create_detection_dataset.py --datapath datasets/bird-detector/original --patchsize 1333")
# Create the classification datasets
os.system("python code/create_classification_datasets.py --datapath datasets/bird-detector")
# Create coco annotation
os.system("python code/labelme_coco.py --datapath datasets/bird-detector/train2017")
# Train bird detector
os.system("python code/train_detector.py --data-path datasets/bird-detector --model FasterRCNN --backbone resnet101 --weights-backbone ResNet101_Weights.DEFAULT --numclasses 2 --output-dir models/temp --lr 0.001 --lr-steps 1 2 3 --epochs 1000 --trainable-backbone-layers 5 --custommodel 1 --batch-size 1")
# Train classifiers
os.system("python code/train_classifiers.py")
# Determine performance metrics
# evaluate()
# perform detection
