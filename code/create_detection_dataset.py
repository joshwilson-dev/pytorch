################
#### Header ####
################

# Title: Crop Dataset
# Author: Josh Wilson
# Date: 07-07-2022
# Description: 
# This script crops the dataset images and labels in the selected directory
# into the smallest equal size less than max crop size  

###############
#### Setup ####
###############
import json
import os
from PIL import Image
import math
import hashlib
import piexif
import os
import shutil
from random import sample
import labelme2coco
import copy

#################
#### Content ####
#################

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Crop image & label dataset", add_help=add_help)

    parser.add_argument("--datapath", default="datasets/seed-box/original", type=str, help="dataset path")
    parser.add_argument("--patchsize", default=800, type=int, help="Size of patches")
    return parser

# did the user select a dir or cancel?
def main(**kwargs):
    # change to directory
    os.chdir(kwargs["datapath"])
    # check if save directory exists and if no create one
    paths = ["../train2017", "../val2017"]
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    # walk through image files and crop
    for file in os.listdir():
        if file.endswith(".JPG"):
            # check if image is labelled
            annotation_name = os.path.splitext(file)[0] + '.json'
            if os.path.exists(annotation_name):
                print("Cropping", file)
                original_image = Image.open(file, mode="r")
                # get exif data
                exif_dict = piexif.load(original_image.info['exif'])
                exif_bytes = piexif.dump(exif_dict)
                width, height = original_image.size
                n_crops_width = math.ceil(width / kwargs["patchsize"])
                n_crops_height = math.ceil(height / kwargs["patchsize"])
                padded_width = n_crops_width * kwargs["patchsize"]
                padded_height = n_crops_height * kwargs["patchsize"]
                pad_width = (padded_width - width) / 2
                pad_height = (padded_height - height) / 2
                left = (padded_width/2) - (width/2)
                top = (padded_height/2) - (height/2)
                image = Image.new(original_image.mode, (padded_width, padded_height), "black")
                image.paste(original_image, (int(left), int(top)))
                for height_index in range(0, n_crops_height):
                    for width_index in range(0, n_crops_width):
                        left = width_index * kwargs["patchsize"]
                        right = left + kwargs["patchsize"]
                        top = height_index * kwargs["patchsize"]
                        bottom = top + kwargs["patchsize"]
                        with open(annotation_name, 'r') as annotation:
                            data = json.load(annotation)
                            data["imageHeight"] = kwargs["patchsize"]
                            data["imageWidth"] = kwargs["patchsize"]
                            restart = True
                            boxes_kept = 0
                            while restart == True and boxes_kept != len(data["shapes"]):
                                for shape_index in range(boxes_kept, len(data["shapes"])):
                                    box_x1 = min(x[0] for x in data["shapes"][shape_index]["points"]) + pad_width
                                    box_x2 = max(x[0] for x in data["shapes"][shape_index]["points"]) + pad_width
                                    box_y1 = min(x[1] for x in data["shapes"][shape_index]["points"]) + pad_height
                                    box_y2 = max(x[1] for x in data["shapes"][shape_index]["points"]) + pad_height

                                    box_left = min(box_x1, box_x2)
                                    box_right = max(box_x1, box_x2)
                                    box_top = min(box_y1, box_y2)
                                    box_bottom = max(box_y1, box_y2)
                                    box_width = box_right - box_left
                                    box_height = box_bottom - box_top
                                    box_area = box_width * box_height

                                    if box_left > right or box_right < left or \
                                        box_top > bottom or box_bottom < top:
                                        del data["shapes"][shape_index]
                                        restart = True
                                        break
                                    else:
                                        if box_left < left: box_left = left
                                        if box_right > right: box_right = right
                                        if box_top < top: box_top = top
                                        if box_bottom > bottom: box_bottom = bottom
                                        new_box_width = box_right - box_left
                                        new_box_height = box_bottom - box_top
                                        new_box_area = new_box_width * new_box_height
                                        if new_box_area < box_area:
                                            # blackout part of image if box is cut off
                                            topleft = (round(box_left), round(box_top))
                                            black_box = Image.new("RGB", (round(new_box_width), round(new_box_height)))
                                            image.paste(black_box, topleft)
                                            del data["shapes"][shape_index]
                                            restart = True
                                            break
                                        else:
                                            for index in range(len(data["shapes"][shape_index]["points"])):
                                                data["shapes"][shape_index]["points"][index][0] += - left + pad_width
                                                data["shapes"][shape_index]["points"][index][1] += - top + pad_height
                                            restart = False
                                            boxes_kept += 1
                            
                            if len(data["shapes"]) > 0:
                                image_crop = image.crop((left, top, right, bottom))
                                md5hash = hashlib.md5(image_crop.tobytes()).hexdigest()
                                image_crop.save("../train2017/" + md5hash + ".JPG", exif = exif_bytes)
                                annotation_output = "../train2017/" + md5hash + ".json"
                                data["imagePath"] = md5hash + ".JPG"
                                with open(annotation_output, 'w') as new_annotation:
                                    json.dump(data, new_annotation, indent=2)
    # # Create COCO annotations
    # os.chdir("../train2017")
    # # set path for coco json to be saved
    # if os.path.exists("../annotations"):
    #     shutil.rmtree("../annotations")
    # os.makedirs("../annotations")
    # save_json_path = "../annotations"

    # # convert labelme annotations to coco
    # labelme2coco.convert("./", save_json_path)

    # # adjust annotations
    # annotation_path = os.path.join("../annotations/", "dataset.json")
    # annotation_file = open(annotation_path)
    # annotation = json.load(annotation_file)
    # annotation_file.close()
    # os.remove(annotation_path)

    # # fix indexing error
    # for ann in annotation["annotations"]:
    #     for index in reversed(range(0, len(annotation["categories"]))):
    #         if ann["category_id"] == index:
    #             ann["category_id"] = index + 1
    # for ann in annotation["categories"]:
    #     for index in reversed(range(0, len(annotation["categories"]))):
    #         if ann["id"] == index:
    #             ann["id"] = index + 1
    
    # # create mask from bounding box
    # for index in range(len(annotation["annotations"])):
    #     box = annotation["annotations"][index]["bbox"]
    #     segmentation = [
    #         [
    #             box[0],
    #             box[1],
    #             box[0] + box[2],
    #             box[1],
    #             box[0] + box[2],
    #             box[1] + box[3],
    #             box[0],
    #             box[1] + box[3]
    #             ]
    #         ]
    #     annotation["annotations"][index]["segmentation"] = segmentation

    # # make all classes bird
    # for index in range(len(annotation["annotations"])):
    #     annotation["annotations"][index]["category_id"] = 1
    # annotation["categories"] = [{"id": 1, "name": "Bird","supercategory": "Bird"}]

    # # split annotation into test and train
    # test_annotation = copy.deepcopy(annotation)
    # train_annotation = copy.deepcopy(annotation)

    # for image_index in range(len(annotation["images"])):
    #     image_id = annotation["images"][image_index]["id"]
    #     if image_id < round(len(annotation["images"]) * 0.2):
    #         for index in reversed(range(len(train_annotation["annotations"]))):
    #             if train_annotation["annotations"][index]["image_id"] == image_id:
    #                 del train_annotation["annotations"][index]
    #         for index in reversed(range(len(train_annotation["images"]))):
    #             if train_annotation["images"][index]["id"] == image_id:
    #                 image_name = train_annotation["images"][index]["file_name"]
    #                 annotation_name = os.path.splitext(image_name)[0] + '.json'
    #                 shutil.move(annotation_name, os.path.join("../val2017", annotation_name))
    #                 shutil.move(image_name, os.path.join("../val2017", image_name))
    #                 del train_annotation["images"][index]
    #     else:
    #         for index in reversed(range(len(test_annotation["annotations"]))):
    #             if test_annotation["annotations"][index]["image_id"] == image_id:
    #                 del test_annotation["annotations"][index]
    #         for index in reversed(range(len(test_annotation["images"]))):
    #             if test_annotation["images"][index]["id"] == image_id:
    #                 del test_annotation["images"][index]
    
    # # rename annotation file
    # train_annotation_path = os.path.join("../annotations/", "instances_train2017.json")
    # test_annotation_path = os.path.join("../annotations/", "instances_val2017.json")

    # # save to file
    # train_annotation = json.dumps(train_annotation, indent=4)
    # test_annotation = json.dumps(test_annotation, indent=4)

    # with open(train_annotation_path, "w") as outfile:
    #     outfile.write(train_annotation)
    # with open(test_annotation_path, "w") as outfile:
    #     outfile.write(test_annotation)

if __name__ == "__main__":
    kwargs = vars(get_args_parser().parse_args())
    main(**kwargs)