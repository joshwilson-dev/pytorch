from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
from torchvision.transforms import functional as F, transforms as T
from torchvision.ops import masks_to_boxes

class RandomVerticalFlip(T.RandomVerticalFlip):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.vflip(image)
            if target is not None:
                _, height, _ = F.get_dimensions(image)
                target["boxes"][:, [1, 3]] = height - target["boxes"][:, [3, 1]]
                if "masks" in target:
                    target["masks"] = target["masks"].flip(1)
        return image, target

class RandomRotationNinety(T.RandomRotation):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        fill = self.fill
        channels, _, _ = F.get_dimensions(image)
        if isinstance(image, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]
        angles = list(range(0, 360, 90))
        angle = angles[torch.randint(len(angles), (1,))]
        image = F.rotate(image, angle)
        if target is not None:
            target["masks"] = F.rotate(target["masks"], angle, self.interpolation, self.expand, self.center, fill)
            target["boxes"] = masks_to_boxes(target["masks"])
        return image, target