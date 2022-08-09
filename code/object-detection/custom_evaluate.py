import torch
import custom_dataloader
import importlib
importlib.reload(custom_dataloader)
import utils

from engine import evaluate
device = torch.device("cuda")
model = custom_dataloader.FRCNNObjectDetector()
model_path = "../model/temp/model_final_state_dict.pth"
model.load_state_dict(torch.load(model_path, map_location=device))

_, dataset_original = custom_dataloader.get_dataset()
train_size = int(0.85 * len(dataset_original))
test_size = len(dataset_original) - train_size
_, dataset_test = torch.utils.data.random_split(dataset_original, [train_size, test_size])
torch.backends.cudnn.deterministic = True
test_sampler = torch.utils.data.SequentialSampler(dataset_test)
workers = 4

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    sampler=test_sampler,
    num_workers=workers,
    collate_fn=utils.collate_fn)

evaluate(model, data_loader_test, device=device)