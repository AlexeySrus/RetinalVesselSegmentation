import torch
from cv2 import phase
import torch.utils.data as data
import albumentations as A
import os
import numpy as np
import cv2


def create_square_crop_by_detection(
        frame: np.ndarray,
        box: list,
        return_shifts: bool = False,
        zero_pad: bool = True):
    """
    Rebuild detection box to square shape
    Args:
        frame: rgb image in np.uint8 format
        box: list with follow structure: [x1, y1, x2, y2]
        return_shifts: if set True then function return tuple of image crop
           and (x, y) tuple of shift coordinates
        zero_pad: pad result image by zeros values

    Returns:
        Image crop by box with square shape or tuple of crop and shifted coords
    """
    w = box[2] - box[0]
    h = box[3] - box[1]
    cx = box[0] + w // 2
    cy = box[1] + h // 2
    radius = max(w, h) // 2
    exist_box = []
    pads = []

    # y top
    if cy - radius >= 0:
        exist_box.append(cy - radius)
        pads.append(0)
    else:
        exist_box.append(0)
        pads.append(-(cy - radius))

    # y bottom
    if cy + radius >= frame.shape[0]:
        exist_box.append(frame.shape[0] - 1)
        pads.append(cy + radius - frame.shape[0] + 1)
    else:
        exist_box.append(cy + radius)
        pads.append(0)
    # x left
    if cx - radius >= 0:
        exist_box.append(cx - radius)
        pads.append(0)

    else:
        exist_box.append(0)
        pads.append(-(cx - radius))

    # x right
    if cx + radius >= frame.shape[1]:
        exist_box.append(frame.shape[1] - 1)
        pads.append(cx + radius - frame.shape[1] + 1)
    else:
        exist_box.append(cx + radius)
        pads.append(0)

    exist_crop = frame[
                 exist_box[0]:exist_box[1],
                 exist_box[2]:exist_box[3]
                 ]

    if len(frame.shape) > 2:
        croped = np.pad(
            exist_crop,
            (
                (pads[0], pads[1]),
                (pads[2], pads[3]),
                (0, 0)
            ),
            'edge' if not zero_pad else 'constant'
        )
    else:
        croped = np.pad(
            exist_crop,
            (
                (pads[0], pads[1]),
                (pads[2], pads[3])
            ),
            'edge' if not zero_pad else 'constant'
        )

    if not return_shifts:
        return croped

    shift_x = exist_box[2] - pads[2]
    shift_y = exist_box[0] - pads[0]

    return croped, (shift_x, shift_y)


def get_imglist(data_dir, fold, phase):
    imgs = np.load(os.path.join(data_dir, 'img.npy'))
    mask = np.load(os.path.join(data_dir, 'mask.npy'))
    filenames = np.load(os.path.join(data_dir, 'filename.npy'))
    testnum = int(len(filenames) * 0.2)
    teststart = fold * testnum
    testend = (fold + 1) * testnum

    if phase == 'test':
        filenames = filenames[teststart:testend]
        imgs = imgs[teststart:testend]
        masks = mask[teststart:testend]
    else:
        filenames = np.concatenate([filenames[:teststart], filenames[testend:]], axis=0)
        imgs = np.concatenate([imgs[:teststart], imgs[testend:]], axis=0)
        masks = np.concatenate([mask[:teststart], mask[testend:]], axis=0)
        valnum = int(len(filenames) * 0.2)
        if phase == 'train':
            imgs, masks, filenames = imgs[valnum:], masks[valnum:], filenames[valnum:]
        elif phase == 'val':
            imgs, masks, filenames = imgs[:valnum], masks[:valnum], filenames[:valnum]
        else:
            raise ValueError('phase should be train or val or test')
    return imgs, masks, filenames


class DataFolder(data.Dataset):
    def __init__(self, root_dir, phase, fold, gan_aug=False, data_transform=None):
        """
        :param root_dir: 
        :param data_transform: data transformations
        :param phase: train, val, test
        :param fold: fold number, 0, 1, 2, 3, 4
        :param gan_aug: whether to use gan augmentation
        """
        super(DataFolder, self).__init__()
        self.data_transform = data_transform
        self.gan_aug = gan_aug
        self.phase = phase
        self.fold = fold 
        self.root_dir = root_dir
        self.imgs, self.masks, self.filenames = get_imglist(os.path.join(self.root_dir, 'NumpyData'), self.fold, self.phase)
        self.imgs_aug, self.masks_aug, _ = get_imglist(os.path.join(self.root_dir, 'aug'), self.fold, self.phase)      

    def __len__(self):
        return self.filenames.shape[0]

    def __getitem__(self, idx):
        img, mask, name = self.imgs[idx], self.masks[idx], self.filenames[idx]
        if self.gan_aug and np.random.rand() < 0.2:
            img = self.imgs_aug[idx]
            mask = self.masks_aug[idx]
        
        if self.data_transform is not None:
            transformed = self.data_transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        return {'image': img, 'label': mask, 'name': name}


class EyeMaskDataset(data.Dataset):
    def __init__(self, root_dir, phase, fold, gan_aug=False, data_transform=None):
        super().__init__()

        images_path = os.path.join(root_dir, phase, 'images/')
        masks_path = os.path.join(root_dir, phase, 'masks/')

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

        self.data_transform = data_transform
        self.gan_aug = gan_aug
        self.phase = phase
        self.fold = fold
        self.root_dir = root_dir
        size: int = 512
        self.crop_trans = A.RandomCrop(width=size, height=size, always_apply=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        sample = self.data[index]
        img = cv2.imread(sample['image'], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)

        # img = create_square_crop_by_detection(
        #     img,
        #     [0, 0, *img.shape[:2][::-1]],
        #     zero_pad=True
        # )
        #
        # mask = create_square_crop_by_detection(
        #     mask,
        #     [0, 0, *mask.shape[:2][::-1]],
        #     zero_pad=True
        # )
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = mask // 255

        name = os.path.splitext(os.path.basename(sample['image']))[0]

        tsample = self.crop_trans(image=img, mask=mask)
        img = tsample['image']
        mask = tsample['mask']

        if self.data_transform is not None:
            transformed = self.data_transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        return {'image': img, 'label': mask, 'name': name}


class TestEyeMaskDataset(data.Dataset):
    def __init__(self, root_dir, phase, fold, gan_aug=False, data_transform=None):
        super().__init__()

        images_path = root_dir

        images_files = os.listdir(images_path)

        self.data = []

        for img_file in images_files:
            self.data.append(
                {
                    'image': os.path.join(images_path, img_file)
                }
            )

        self.data_transform = data_transform
        self.gan_aug = gan_aug
        self.phase = phase
        self.fold = fold
        self.root_dir = root_dir

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        sample = self.data[index]
        img = cv2.imread(sample['image'], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        original_size = list(img.shape[:2][::-1])

        # img, shifts = create_square_crop_by_detection(
        #     img,
        #     [0, 0, *img.shape[:2][::-1]],
        #     zero_pad=True,
        #     return_shifts=True
        # )
        shifts = (0, 0)

        name = os.path.splitext(os.path.basename(sample['image']))[0]

        if self.data_transform is not None:
            transformed = self.data_transform(image=img)
            img = transformed['image']
        return {
            'image': img,
            'shifts': torch.LongTensor(list(shifts)),
            'size': torch.LongTensor(original_size),
            'name': name
        }
