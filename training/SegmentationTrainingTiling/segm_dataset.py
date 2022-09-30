from typing import Callable, Optional, Tuple

import cv2
import torch
from tqdm import tqdm
import os
import numpy as np
from torch.utils.data import Dataset
from coco_dataset import create_square_crop_by_detection


class MaskDataset(Dataset):
    def __init__(
            self,
            images_path: str,
            masks_path: str,
            transform: Optional[Callable] = None,
            shape: tuple = (700, 700),
            augmentations: Optional[Callable] = None,
            scale: int = 2
    ) -> None:
        super().__init__()

        images_files = os.listdir(images_path)
        masks_files = os.listdir(masks_path)
        masks_basenames = [os.path.splitext(mf)[0] for mf in masks_files]

        self.data = []

        for img_file in images_files:
            img_basename = os.path.splitext(img_file)[0]
            if img_basename in masks_basenames:
                mask_file_idx = masks_basenames.index(img_basename)

                self.data.append(
                    {
                        'image': os.path.join(images_path, img_file),
                        'mask': os.path.join(masks_path, masks_files[mask_file_idx])
                    }
                )

        self.transform = transform
        self.shape = shape

        self.name = 'MasksDataset'
        self.transforms = transform
        self.augmentations = augmentations
        self.scale = scale

    def __len__(self) -> int:
        return len(self.data) * self.scale

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        sample = self.data[index % len(self.data)]

        assert os.path.exists(sample['image']), sample['image']
        assert os.path.exists(sample['mask']), sample['mask']

        image = cv2.imread(sample['image'], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)

        # image = create_square_crop_by_detection(
        #     image,
        #     [0, 0, *image.shape[:2][::-1]],
        #     zero_pad=True
        # )
        #
        # mask = create_square_crop_by_detection(
        #     mask,
        #     [0, 0, *mask.shape[:2][::-1]],
        #     zero_pad=True
        # )

        if self.augmentations is not None:
            aug_res = self.augmentations(image=image, mask=mask)
            image = aug_res['image']
            mask = aug_res['mask']

        # image = cv2.resize(
        #     image,
        #     self.shape,
        #     interpolation=cv2.INTER_AREA
        # )
        # mask = cv2.resize(mask, self.shape, interpolation=cv2.INTER_NEAREST)

        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = mask != 255

        if self.transforms is not None:
            sample = self.transform(image=image, mask=mask)
            image = sample['image']
            mask = sample['mask']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1) / 255.0

        # _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        # old_masks = np.zeros((1 + 1, self.shape[0], self.shape[1]), dtype=np.uint8)
        # old_masks[-1] = np.ones((self.shape[0], self.shape[1]), dtype=np.uint8)
        # old_masks[0][mask == 255] = 1
        # old_masks[1][mask == 255] = 0

        # mask = torch.LongTensor(mask != 255)
        # old_masks = torch.FloatTensor(old_masks)
        # old_masks = torch.FloatTensor(old_masks)

        return image, mask.to(torch.long)  # , old_masks
