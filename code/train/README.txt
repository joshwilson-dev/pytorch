## Title
Detecting and identifying birds in images captured using a drone

## Authors
Anonymous

## Date
05/06/2024

## Description
These files, along with the dataset available at (https://doi.org/10.5061/dryad.f4qrfj73z), contain the tables and scripts required to train, assess, and use a computer vision tool that detects and identifies birds in images captured using a drone. We provide these resources open-access in the hopes that they will (1) help devlop computer vision tools to detect and describe birds in drone imagery accross (2) reduce the costs associated with manually processing drone-based survey images (3) lay the foundations to conduct more extensive surveys and provide more comprehensive evidence to inform ecological research and conservation action.

## Instructions
1. Download 'Detecting_and_identifying_birds_in_images_captured_using_a_drone.zip' from https://doi.org/10.5061/dryad.f4qrfj73z.
2. Extract the files and place the extracted folder in the same folder as these supporting files.
3. Install the required R packages with renv::restore()
4. Create a python virtual environment (python3 -m venv venv)
5. Activate the enviornment (source venv/Scripts/activate)
6. Use pip to install the required packages (pip install -r requirements.txt).
7. Work through the scripts in order.

## File structure
*   Supporting Tables TS1-6.xlsx
    *   TS1-review: Relevant studies and authors contacted for contribution to dataset.
    *   TS2-dataset: A summary of the metrics of every image in the total, train, test, and validation datasets.
    *   TS3-cocoeval: COCO evaluation metrics summary.
    *   TS4-cocoevalimg: COCO evaluation metrics per image.
    *   TS5-training: Training summary.
    *   TS6-catId_to_class: Class information.

*   CS1_balanced_sampling.py: Code to split the dataset (https://doi.org/10.5061/dryad.f4qrfj73z) images into panels and sub-sample these panels to create the train, test, and validaiton datasets.
*   CS2_train.py: Code to train the network using the dataset created with CS1_balanced_sampling.py
*   CS3_dataset_overview.R: Code to calculate metrics summarising the dataset and generate corresponding figures.
*   CS4_test.py: Code to calculate the COCO evaluation metrics for the test and validation datasets.
*   CS5_tp_fp_fn.R: Code to calculate the true positives, false positives, and false negatives for each class on the validation and test datasets.
*   CS6_average_precision.R: Code to calculate the average precision for each class, detection, and classification.
*   CS7_error.R: Code to plot the most common false positive detections.
*   CS8_inference.py: Code to use the network to make predictions on new images.
*   CS9_utils.py: Code to support network training. Adapted from https://github.com/pytorch/vision/tree/main/references/detection.
*   CS10_engine.py: Code to support network training. Adapted from https://github.com/pytorch/vision/tree/main/references/detection.
*   CS11_transforms.py: Code to support network training. Adapted from https://github.com/pytorch/vision/tree/main/references/detection.
*   CS12_coco_ultils.py: Code to support network training. Adapted from https://github.com/pytorch/vision/tree/main/references/detection.
*   CS13_evaluate.py: Code to support network training. Adapted from https://github.com/pytorch/vision/tree/main/references/detection.
*   CS14_custom_roi_heads.py: Modified roi_heads.py from the torch package to support returning scores for all classes and filtering included classes.
*   CS15_coco_eval.py: Code to support network training. Adapted from https://github.com/pytorch/vision/tree/main/references/detection.

*   AS1_index_to_class.json: File linking the network classes to their index.
*   AS2_model.pth: The network file.

*   demos
    *   demo1.jpg: image for demonstrating CS8_inference.py
    *   demo2.jpg: image for demonstrating CS8_inference.py

*   renv
    *   activate.R: File used to install the required R packages.
    *   settings.json: File used to install the required R packages.
    *   .gitignore: File used to install the required R packages.
*   .Rprofile: File used to install the required R packages.
*   renv.lock: File used to install the required R packages.
*   requirements.txt: File used to install the required python packages.