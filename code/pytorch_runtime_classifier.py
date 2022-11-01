from torchvision import datasets, models, transforms
import torch.nn as nn
import torch
import PIL
import torch.nn.functional as F
import json

def prepare_image(image_path):
    image = PIL.Image.open(image_path).convert('RGB')
    width, height = image.size
    # pad crop to square and resize
    max_dim = max(width, height)
    if height > width:
        pad = [int((height - width)/2) + 20, 20]
    else:
        pad = [20, int((width - height)/2) + 20]
    image = transforms.Pad(padding=pad)(image)
    image = transforms.CenterCrop(size=max_dim)(image)
    image = transforms.Resize(224)(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    return image

root = "models/temp/Global/"

idx_to_class = json.load(open(root + "index_to_class.json"))
number_classes = len(idx_to_class)
model = models.resnet101()
features = model.fc.in_features
model.fc = nn.Linear(features, number_classes)
device = torch.device("cuda")
model = model.to(device)
model.load_state_dict(torch.load(root + "model_best_state_dict.pth"))
model.eval()
prediction = model(prepare_image("datasets/test/masked lapwing - 1.jpg"))[0]
prediction = F.softmax(prediction, dim=0).tolist()
print(prediction)
species = idx_to_class[str(prediction.index(max(prediction)))]
print(species)