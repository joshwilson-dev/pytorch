from torchvision import datasets, models, transforms
import torch.nn as nn
import torch
import PIL
import torch.nn.functional as F

def prepare_image(image_path):
    image = PIL.Image.open(image_path).convert('RGB')
    image = transforms.ToTensor()(image)
    image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
    image = transforms.Resize(256)(image)
    image = transforms.CenterCrop(224)(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    return image

idx_to_order = {1: 'anseriformes', 2: 'gruiformes', 3: 'pelecaniformes', 4: 'charadriiformes', 5: 'suliformes', 6: 'passeriformes'}

model = models.resnet50()
features = model.fc.in_features
model.fc = nn.Linear(features, 6)
device = torch.device("cuda")
model = model.to(device)
model.load_state_dict(torch.load("../../models/bird-detector/object/aves/model_final_state_dict.pth"))
model.eval()
prediction = model(prepare_image("../../datasets/bird-detector/test/test-4.JPG"))[0]
prediction = F.softmax(prediction, dim=0).tolist()
print(prediction)
# order = idx_to_order[prediction.index(max(prediction))]