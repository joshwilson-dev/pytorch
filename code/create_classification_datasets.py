################
#### Header ####
################

# Title: Crop Dataset
# Author: Josh Wilson
# Date: 29-07-2022
# Description: 
# This script takes an object detection dataset and crops the images
# and annotations to create an image classification dataset 

###############
#### Setup ####
###############
import json
import os
from PIL import Image
import hashlib
import piexif
import torchvision.transforms as T
import shutil

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Crop image & label dataset", add_help=add_help)

    parser.add_argument("--datapath", default="./datasets/bird-detector/", type=str, help="dataset path")
    return parser

def main(**kwargs):    
    os.chdir(kwargs["datapath"])
    birds_by_region = json.load(open("birds_by_region.json"))
    # load regional json
    regions = birds_by_region.keys()
    labels_dict = {}
    for region in regions:
        labels_dict[region] = {"images": [], "labels": []}
    labels_dict["Age"] = {"images": [], "labels": []}
    labels_dict["Global"] = {"images": [], "labels": []}
    for classifier in labels_dict.keys():
        if os.path.exists(classifier):
            shutil.rmtree(classifier)
        os.makedirs(classifier)
    if os.path.exists("Images"):
        shutil.rmtree("Images")
    os.makedirs("Images")
    # walk through image files and crop
    for file in os.listdir("train2017"):
        if file.endswith(".json"):
            print("Creating classifier data for", file)
            # read annotation file
            annotations = json.load(open(os.path.join("train2017/",file)))
            # get image
            image_file = os.path.splitext(file)[0] + '.JPG'
            image = Image.open(os.path.join("train2017/", image_file))
            # get exif data
            exif_dict = piexif.load(image.info['exif'])
            exif_bytes = piexif.dump(exif_dict)
            try:
                comments = json.loads("".join(map(chr, [i for i in exif_dict["0th"][piexif.ImageIFD.XPComment] if i != 0])))
                gsd = float(comments["gsd"])
            except:
                print("GSD NEEDED", file)
                break
            # if resolution is high enough for species detection
            if gsd < 0.075:
                for index1 in range(len(annotations["shapes"])):
                    # crop image to instance
                    points = annotations["shapes"][index1]["points"]
                    box_x1 = points[0][0]
                    box_x2 = points[1][0]
                    box_y1 = points[0][1]
                    box_y2 = points[1][1]
                    box_left = min(box_x1, box_x2)
                    box_right = max(box_x1, box_x2)
                    box_top = min(box_y1, box_y2)
                    box_bottom = max(box_y1, box_y2)
                    instance = image.crop((box_left, box_top, box_right, box_bottom))
                    
                    # pad crop to square
                    width, height = instance.size
                    max_dim = max(width, height)
                    if height > width:
                        pad = [int((height - width) / 2) + 20, 20]
                    else:
                        pad = [20, int((width - height)/2) + 20]
                    instance = T.Pad(padding=pad)(instance)
                    instance = T.CenterCrop(size=max_dim)(instance)

                    # resize crop to 224
                    instance = T.transforms.Resize(224)(instance)
                    
                    # determine image hash
                    md5hash = hashlib.md5(instance.tobytes()).hexdigest()
                    image_name = md5hash + ".JPG"

                    # get image exif
                    exif_dict = piexif.load(image.info["exif"])
                    exif_bytes = piexif.dump(exif_dict)
                    #TODO update exif image size

                    # save instance to images
                    instance.save(os.path.join("images", image_name), exif = exif_bytes)

                    # get species name and age from label
                    label = annotations["shapes"][index1]["label"].split("_")
                    species = label[-2] + " " + label[-1]
                    age = label[-5]

                    # check if species is unknown and only classify adult birds
                    if "unknown" not in species and age == "adult":
                        species_in_region = 0
                        # add to unknown region dataset
                        labels_dict["Global"]["images"].append(image_name)
                        labels_dict["Global"]["labels"].append(species)
                        # check which regions species occurs in
                        for region in regions:
                            if species in birds_by_region[region]:
                                # save label and image name into relveant label dict key
                                species_in_region = 1
                                labels_dict[region]["images"].append(image_name)
                                labels_dict[region]["labels"].append(species)
                        if species_in_region == 0:
                            print(species, " not in any region")

                    # check if age is unknown
                    if "unknown" not in age:
                        labels_dict["Age"]["images"].append(image_name)
                        labels_dict["Age"]["labels"].append(age)
        for classifier in labels_dict.keys():
            dataset_path = os.path.join(classifier, "dataset.json")
            instance_annotations = json.dumps(labels_dict[classifier], indent=2)
            with open(dataset_path, "w") as instance_annotation_file:
                instance_annotation_file.write(instance_annotations)

if __name__ == "__main__":
    kwargs = vars(get_args_parser().parse_args())
    main(**kwargs)