import torch
import torchvision
import custom_dataloader
import PIL
import numpy as np
from matplotlib import pyplot as plt
import os
import importlib

idx_to_class = {
    1: "fertilised",
    2: "unfertilised"
}

idx_to_colour = {
    1: "orange",
    2: "blue"
}

def prepare_image(image_path):
    image = PIL.Image.open(image_path).convert('RGB')
    image = torchvision.transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    return image

def show(imgs):
    if not isinstance(imgs,list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize = (40,40))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.show()

def draw_boxes(image_path, model):
    image = prepare_image(image_path)
    with torch.no_grad():
        prediction = model(image)
    image = (image*255).type(torch.uint8)[0]
    image_boxes = prediction[0]["boxes"]
    labels = prediction[0]["labels"].tolist()
    scores = prediction[0]["scores"].tolist()
    string_scores = ['{0:.2f}'.format(score) for score in scores]
    named_labels = [idx_to_class[i] for i in labels]
    colours = [idx_to_colour[i] for i in labels]
    named_labels_with_scores = [named_labels[i] + ": " + string_scores[i] for i in range(len(scores))]
    visualise = torchvision.utils.draw_bounding_boxes(image = image, boxes = image_boxes, labels = named_labels_with_scores, font_size = 15, width=5, colors = colours)
    show(visualise)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cuda")
# device = torch.device("cpu")
importlib.reload(custom_dataloader)
model = custom_dataloader.FRCNNObjectDetector()
model_path = "../model/temp/model_final_state_dict.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model = model.to(device)
prediction_1 = draw_boxes("../dataset/test_full/OFF_011_C.JPG", model)
# prediction_1 = draw_boxes("../dataset/test/0a259b562b59aaeb192d61734edaae1f.JPG", model)
prediction_1