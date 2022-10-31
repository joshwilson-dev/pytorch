# import package
import labelme2coco
import os
import json
import shutil

#################
#### Content ####
#################
def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Create COCO annotation from labelme", add_help=add_help)

    parser.add_argument("--datapath", default="datasets/seed-box/train2017", type=str, help="dataset path")
    return parser

def main(**kwargs):
    os.chdir(kwargs["datapath"])

    # set path for coco json to be saved
    if os.path.exists("../annotations"):
        shutil.rmtree("../annotations")
    os.makedirs("../annotations")
    save_json_path = "../annotations"

    # convert labelme annotations to coco
    labelme2coco.convert("./", save_json_path)

    # adjust annotations
    annotation_path = os.path.join("../annotations/", "dataset.json")
    annotation_file = open(annotation_path)
    annotation = json.load(annotation_file)
    annotation_file.close()
    os.remove(annotation_path)

    # rename annotation file
    annotation_path = os.path.join("../annotations/", "instances_train2017.json")

    # fix indexing error
    for ann in annotation["annotations"]:
        for index in reversed(range(0, len(annotation["categories"]))):
            if ann["category_id"] == index:
                ann["category_id"] = index + 1
    for ann in annotation["categories"]:
        for index in reversed(range(0, len(annotation["categories"]))):
            if ann["id"] == index:
                ann["id"] = index + 1

    # add pixels to bounding boxes
    # for index in range(len(annotation["annotations"])):
    #     box = annotation["annotations"][index]["bbox"]
    #     box[0] += -2
    #     box[1] += -2
    #     box[2] += 4
    #     box[3] += 4
    #     area = box[2] * box[3]
    #     annotation["annotations"][index]["bbox"] = box
    #     annotation["annotations"][index]["area"] = area

    # create mask from bounding box
    for index in range(len(annotation["annotations"])):
        box = annotation["annotations"][index]["bbox"]
        segmentation = [
            [
                box[0],
                box[1],
                box[0] + box[2],
                box[1],
                box[0] + box[2],
                box[1] + box[3],
                box[0],
                box[1] + box[3]
                ]
            ]
        annotation["annotations"][index]["segmentation"] = segmentation
    
    # make all classes bird
    # for index in range(len(annotation["annotations"])):
    #     annotation["annotations"][index]["category_id"] = 1
    # annotation["categories"] = [{"id": 1, "name": "Bird","supercategory": "Bird"}]

    # save to file
    annotation = json.dumps(annotation, indent=4)

    with open(annotation_path, "w") as outfile:
        outfile.write(annotation)

if __name__ == "__main__":
    kwargs = vars(get_args_parser().parse_args())
    main(**kwargs)