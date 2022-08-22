# Notes
# https://github.com/onnx/onnx/issues/654

import PIL
import torchvision
import custom_dataloader
import torch
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

device = torch.device("cpu")
model = custom_dataloader.FRCNNObjectDetector()
model_path = "../model/seed-detector/model_final_state_dict.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model = model.to(device)

print("Exporting model as onnx")
test_image = PIL.Image.open("../dataset/test/0a259b562b59aaeb192d61734edaae1f.JPG").convert('RGB')
test_image = torchvision.transforms.ToTensor()(test_image)
test_image = test_image.unsqueeze(0)
test_image = test_image.to(device)
test_image = test_image.to("cpu")

onnx_path = "../model/seed-detector/model_final_state_dict.onnx"
export_params=True
opset_version=11
do_constant_folding=True
input_names=['input']
output_names=['output']
dynamic_axes= {'input':{0:'batch_size', 2:'width', 3:'height'}, 'output':{0:'batch_size', 2:'width', 3:'height'}}
torch.onnx.export(model, test_image, onnx_path, export_params=export_params, opset_version=opset_version, do_constant_folding=do_constant_folding, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)