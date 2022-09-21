from torchvision import datasets, models, transforms
import torch.nn as nn
import torch
import PIL
import torch.nn.functional as F

def prepare_image(image_path):
    image = PIL.Image.open(image_path).convert('RGB')
    image = transforms.Resize(224)(image)
    image = transforms.CenterCrop(224)(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    return image

idx_to_order = {0: "anas superciliosa", 1: "anhinga rufa", 2: "aythya australis", 3: "bubulcus ibis", 4: "cairina moschata", 5: "calidris tenuirostris", 6: "chenonetta jubata", 7: "chroicocephalus novaehollandiae", 8: "corvus orru", 9: "cracticus nigrogularis", 10: "cygnus atratus", 11: "gallinula tenebrosa", 12: "gelochelidon nilotica", 13: "grallina cyanoleuca", 14: "haematopus longirostris", 15: "himantopus leucocephalus", 16: "larus argentatus", 17: "larus fuscus", 18: "limosa lapponica", 19: "numenius phaeopus", 20: "onychoprion aleuticus", 21: "porphyrio melanotus", 22: "pygoscelis antarcticus", 23: "pygoscelis papua", 24: "sterna hirundo", 25: "threskiornis molucca", 26: "threskiornis spinicollis", 27: "vanellus miles"}

model = models.resnet101()
features = model.fc.in_features
model.fc = nn.Linear(features, 28)
device = torch.device("cuda")
model = model.to(device)
model.load_state_dict(torch.load("../models/temp/Oceania/model_best_state_dict.pth"))
model.eval()
# prediction = model(prepare_image("../datasets/bird-detector/gsd/fine/classifiers/Images/842587bdc5cc9b4a68bbfc281b4e122b.JPG"))[0]
prediction = model(prepare_image("../datasets/trial/trial3.JPG"))[0]
prediction = F.softmax(prediction, dim=0).tolist()
print(prediction)
species = idx_to_order[prediction.index(max(prediction))]
print(species)