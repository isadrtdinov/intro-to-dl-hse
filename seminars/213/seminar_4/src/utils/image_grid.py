import torch
from torch import Tensor


def image_grid(*images, num_images = 1, clip=(-1, 1)) -> Tensor:
    new_images = []
    for image in images:
        if image.shape[1] == 2:
            new_images += [torch.cat([image, torch.zeros_like(image[:, :1])], dim=1)]
        elif image.shape[1] == 1:
            new_images += [image.repeat(1, 3, 1, 1)]
        else:
            new_images += [image]
    images = new_images
    horizontal_grid = torch.cat(images, dim=-1)
    images = torch.unbind(horizontal_grid[:num_images], dim=0)
    vertical_grid = torch.cat(images, dim=-2)
    out = torch.clip(vertical_grid, min=clip[0], max=clip[1])
    return out
