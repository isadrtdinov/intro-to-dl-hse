import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image

from .transforms import default_transform


class OxfordIIITPetDataset(Dataset):
    """Oxford IIIT Pet dataset. Read images and apply transforms.

    Args:
        data_dir (str): path to data folder
        transform (torchvision.transforms.transform): image transform
        random_aug (bool): enable random augmentations

    """

    def __init__(
            self,
            data_dir,
            input_transform = None,
            target_transform = None,
            random_aug = False,
            test_mode = True
    ):
        self.data_dir = data_dir[:-1] if data_dir[-1] == "/" else data_dir
        self.images_dir = data_dir + "/images"
        self.mask_dir = data_dir + "/trimaps"
        self.input_transform = default_transform() if input_transform is None else input_transform
        self.target_transform = default_transform() if target_transform is None else target_transform
        self.random_aug = random_aug

        self.ids = [name for name in os.listdir(self.images_dir) if
                    name.lower().endswith('.png') or
                    name.lower().endswith('.jpg') or
                    name.lower().endswith('.jpeg') or
                    name.lower().endswith('.gif') or
                    name.lower().endswith('.bmp')]
        # Get cats only
        self.ids = [name for name in self.ids if
                    name[0].upper() == name[0]]
        self.ids.sort()
        train_ids, test_ids = train_test_split(self.ids, test_size=0.1, random_state=42)
        if test_mode:
            self.ids = test_ids
        else:
            self.ids = train_ids

    def __getitem__(self, i):
        # Load image
        # print(self.images_dir + "/" + self.ids[i])
        # image = cv2.imread(self.images_dir + "/" + self.ids[i], cv2.IMREAD_UNCHANGED)
        # image = image[..., ::-1]
        image = np.array(Image.open(self.images_dir + "/" + self.ids[i]).convert('RGB'))
        h, w = image.shape[:2]

        mask = cv2.imread(self.mask_dir + "/" + self.ids[i][:-3] + "png", cv2.IMREAD_GRAYSCALE)
        foreground = (mask == 1) | (mask == 3)  # 1 - foreground, 2 - background, 3 - not sure
        horizontal = np.nonzero(np.any(foreground, axis=0))[0]
        vertical = np.nonzero(np.any(foreground, axis=1))[0]
        if len(horizontal) == 0:
            left, right = 0, image.shape[1]
        else:
            left, right = np.min(horizontal), np.max(horizontal) + 1
        if len(vertical) == 0:
            top, bottom = 0, image.shape[0]
        else:
            top, bottom = np.min(vertical), np.max(vertical) + 1

        image = image[top:bottom, left:right]
        mask = mask[top:bottom, left:right].astype(np.float32)

        image *= np.expand_dims((mask != 2), 2)  # 2 - background
        image += np.expand_dims((mask == 2) * 255, 2).astype(np.uint8)

        # Random flips
        if self.random_aug:
            if np.random.rand() < 0.5:
                image = image[:, ::-1]
                mask = mask[:, ::-1]
            target_size = np.random.randint(max(image.shape[0], image.shape[1]), max(h, w) + 1)
            x = np.random.randint(0, target_size - image.shape[1] + 1)
            y = np.random.randint(0, target_size - image.shape[0] + 1)
            padding = [(y, target_size - image.shape[0] - y), (x, target_size - image.shape[1] - x)]
            image = np.pad(image, padding + [(0, 0)], constant_values=255)
            mask = np.pad(mask, padding, constant_values=2)  # 2 - background

        # Apply transforms
        image = self.target_transform(image.copy())
        mask = self.input_transform(mask.copy())

        return mask, image

    def __len__(self):
        return len(self.ids)
