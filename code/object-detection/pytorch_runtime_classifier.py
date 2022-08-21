from torchvision import datasets, models, transforms
import torch.nn as nn
import torch
import PIL
import torch.nn.functional as F

def prepare_image(image_path):
    image = PIL.Image.open(image_path).convert('RGB')
    image = transforms.Resize(256)(image)
    image = transforms.CenterCrop(224)(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    return image
# idx_to_class = {0: 'adult', 1: 'chick'}
idx_to_class = {0: 'anseriformes', 1: 'charadriiformes', 2: 'ciconiiformes', 3: 'gruiformes', 4: 'passeriformes', 5: 'pelecaniformes', 6: 'phoenicopteriformes', 7: 'podicipediformes', 8: 'sphenisciformes', 9: 'suliformes'}

model = models.resnet50()
features = model.fc.in_features
model.fc = nn.Linear(features, len(idx_to_class))
device = torch.device("cuda")
model = model.to(device)
model.load_state_dict(torch.load("../../models/bird-classifier/order/model_final_state_dict.pth"))
model.eval()
prediction = model(prepare_image("../../datasets/bird-classifier/age/train/adult/0a3bf5cc77cbb722ee7b79b44e33b405.JPG"))[0]
prediction = F.softmax(prediction, dim=0).tolist()
print(prediction)
order = idx_to_class[prediction.index(max(prediction))]
print(order)