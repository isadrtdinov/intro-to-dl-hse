import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image

from .transforms import default_transform


class OxfordIIITPetEdgesDataset(Dataset):
    """Oxford IIIT Pet dataset. Read images and apply transforms.

    Args:
        data_dir (str): path to data folder
        transform (torchvision.transforms.transform): image transform
        random_aug (bool): enable random augmentations

    """

    def __init__(
            self,
            data_dir,
            resize=None,
            input_transform = None,
            target_transform = None,
            random_aug = False,
            test_mode = True,
            thresholds = (100, 200)
    ):
        self.data_dir = data_dir[:-1] if data_dir[-1] == "/" else data_dir
        self.resize = resize
        self.images_dir = data_dir + "/images"
        self.mask_dir = data_dir + "/trimaps"
        self.input_transform = default_transform() if input_transform is None else input_transform
        self.target_transform = default_transform() if target_transform is None else target_transform
        self.random_aug = random_aug
        self.test_mode = test_mode
        self.thresholds = thresholds

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
        # image = cv2.imread(self.images_dir + "/" + self.ids[i], cv2.IMREAD_UNCHANGED)
        # image = image[..., ::-1]
        image = np.array(Image.open(self.images_dir + "/" + self.ids[i]).convert('RGB'))
        h, w = image.shape[:2]

        # Load trimap
        mask = cv2.imread(self.mask_dir + "/" + self.ids[i][:-3] + "png", cv2.IMREAD_GRAYSCALE)

        # Crop everything to content
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
        mask = mask[top:bottom, left:right]

        # Random augs
        if self.random_aug:
            if np.random.rand() < 0.5:
                image = image[:, ::-1]
                mask = mask[:, ::-1]
            target_size = np.random.randint(max(image.shape[0], image.shape[1]), max(h, w) + 1)
        else:
            target_size = max(image.shape[0], image.shape[1])

        # Pad to square
        x = np.random.randint(0, target_size - image.shape[1] + 1)
        y = np.random.randint(0, target_size - image.shape[0] + 1)
        padding = [(y, target_size - image.shape[0] - y), (x, target_size - image.shape[1] - x), (0, 0)]
        image = np.pad(image, padding, constant_values=255)
        mask = np.pad(mask, padding[:-1], constant_values=2)  # 2 - background

        # Resize
        if self.resize is not None:
            image = cv2.resize(image, self.resize)
            mask = cv2.resize(mask, self.resize, interpolation=cv2.INTER_NEAREST)

        # Extract edges
        expanded_mask = (mask == 1).astype(np.uint8) * 255  # 1 - foreground
        kernel_size = 15 * self.resize[0] // target_size
        if kernel_size % 2 == 0:
            kernel_size += 1
        expanded_mask = (cv2.GaussianBlur(expanded_mask, (kernel_size, kernel_size), 0) > 0)
        input = cv2.Canny(image,
                          threshold1=self.thresholds[0],
                          threshold2=self.thresholds[1])
        input *= expanded_mask
        # input = np.stack((np.zeros_like(expanded_mask), input / 255, ~expanded_mask), 2)
        input = np.stack((input.astype(np.float32) / 255, ~expanded_mask), 2)

        # Apply mask to image
        image *= np.expand_dims(expanded_mask, 2)
        image += np.expand_dims((~expanded_mask) * 255, 2).astype(np.uint8)

        # Apply transforms
        image = self.target_transform(image.copy())
        input = self.input_transform(input.copy())

        return input, image

    def __len__(self):
        return len(self.ids)
