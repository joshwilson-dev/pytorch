import torch
import numpy as np
import custom_dataloader
import PIL
import torchvision
import os
from matplotlib import pyplot as plt

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

def roitoimg(roi):
    ndarr = roi.permute(1, 2, 0).cpu().numpy()
    img_to_draw = PIL.Image.fromarray(ndarr)
    return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)

def get_selected_rois(model, image, verbose=True):
    outputs = []
    model.eval()

    hook = model.backbone.register_forward_hook(
        lambda self, input, output: outputs.append(output))
    res = model(image)
    hook.remove()

    selected_rois = model.roi_heads.box_roi_pool(
        outputs[0], [r['boxes'] for r in res], [i.shape[-2:] for i in image])

    if verbose:
        print(selected_rois)
        print(selected_rois.shape)
    return selected_rois

os.chdir(os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cuda")
model = custom_dataloader.FRCNNObjectDetector()
model_path = "./models/bird-detector/model_final_state_dict.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
image = prepare_image("../dataset/bird-detector/test/small.jpg")
rois = get_selected_rois(model, image, verbose=True)

for i in range(0, 1):
    roi = rois[i][0]
    print(roi)
    roi = (roi*255).type(torch.uint8)[0]
    roi_img = roitoimg(roi)
    show(roi_img)
