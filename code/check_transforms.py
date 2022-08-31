from PIL import Image

import torch
import torchvision.transforms as T
torch.manual_seed(0)
orig_img = Image.open("../datasets/trial/crop_1.jpg")
orig_img.show()
width, height = orig_img.size
if height > width: pad = [int((height - width) / 2), 0]
else: pad = [0, int((width - height)/2)]
print(pad)
transformed_img = T.Pad(padding=pad)(orig_img)
transformed_img.show()