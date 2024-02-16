import cv2
import numpy as np
import torch
from torch import Tensor


def save_image(path: str, image: Tensor, unnorm: bool = True) -> Tensor:
    if unnorm:
        image = image / 2 + 0.5
    image = image.clip(0, 1)
    image = torch.movedim(image, 0, -1)
    image = (image.cpu().numpy() * 255 + 0.5).astype(np.uint8)
    cv2.imwrite(path, image[..., ::-1])
