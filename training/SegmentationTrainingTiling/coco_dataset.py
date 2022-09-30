from typing import Callable, Optional, Tuple

import cv2
import torch
from tqdm import tqdm
import os
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset


COCO_LABEL_MAP = { 1:  1,  2:  2,  3:  3,  4:  4,  5:  5,  6:  6,  7:  7,  8:  8,
                   9:  9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}


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


class COCOImageData(Dataset):
    def __init__(
            self,
            annotations_path: str,
            root_path: str,
            transform: Optional[Callable] = None,
            has_gt: bool = True,
            shape: tuple = (700, 700),
            to_save_masks: bool = False
    ) -> None:
        super().__init__()
        self.coco = COCO(annotations_path)

        self.ids = list(self.coco.imgToAnns.keys())
        if len(self.ids) == 0 or not has_gt:
            self.ids = list(self.coco.imgs.keys())

        self.transform = transform
        self.root = root_path
        self.shape = shape

        self.name = 'COCO'
        self.has_gt = has_gt
        self.transforms = transform
        self.to_save_masks = to_save_masks

    def get_weights(self) -> torch.FloatTensor:
        counts = torch.zeros(81)

        print('Compute dataset weights...')
        for idx in tqdm(range(self.__len__())):
            img_id = self.ids[idx]

            target = []
            if self.has_gt:
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                target = [x for x in self.coco.loadAnns(ann_ids) if x['image_id'] == img_id]

            crowd = [x for x in target if ('iscrowd' in x and x['iscrowd'])]
            target = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]

            for x in crowd:
                x['category_id'] = -1
            target += crowd

            target = np.array(target)

            for t in target:
                label = t['category_id']
                if label == -1:
                    continue

                label_num = COCO_LABEL_MAP[label] - 1
                counts[label_num] += 1

        counts[80] = 0.9999

        return 1.0 - counts / counts.sum()

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        img_id = self.ids[index]

        target = []
        if self.has_gt:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)

            # Target has {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}
            target = [x for x in self.coco.loadAnns(ann_ids) if
                      x['image_id'] == img_id]

        # Separate out crowd annotations. These are annotations that signify a large crowd of
        # objects of said class, where there is no annotation for each individual object. Both
        # during testing and training, consider these crowds as neutral.
        crowd = [x for x in target if ('iscrowd' in x and x['iscrowd'])]
        target = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]
        num_crowds = len(crowd)

        for x in crowd:
            x['category_id'] = -1

        # This is so we ensure that all crowd annotations are at the end of the array
        target += crowd

        target = np.array(target)

        # The split here is to have compatibility with both COCO2014 and 2017 annotations.
        # In 2014, images have the pattern COCO_{train/val}2014_%012d.jpg, while in 2017 it's %012d.jpg.
        # Our script downloads the images as %012d.jpg so convert accordingly.
        file_name = self.coco.loadImgs(img_id)[0]['file_name']

        if file_name.startswith('COCO'):
            file_name = file_name.split('_')[-1]

        path = os.path.join(self.root, file_name)
        assert os.path.exists(path), 'Image path does not exist: {}'.format(path)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape

        masks = []
        if len(target) > 0:
            # Pool all the masks for this image into one [num_objects,height,width] matrix
            masks = [self.coco.annToMask(obj).reshape(-1) for obj in target]
            masks = np.vstack(masks)
            masks = masks.reshape(-1, height, width)

        labels = [t['category_id'] for t in target]

        # print(labels)

        if target.shape[0] == 0:
            print(
                'Warning: Augmentation output an example with no ground truth. Resampling...')
            return self.pull_item(np.random.randint(0, len(self.ids)))

        image = create_square_crop_by_detection(
            img,
            [0, 0, *img.shape[:2][::-1]],
            zero_pad=True
        )

        image = cv2.resize(
            image,
            self.shape,
            interpolation=cv2.INTER_AREA
        )

        masks = [(m * 255).astype(np.uint8) for m in masks]

        for k in range(len(masks)):
            masks[k] = create_square_crop_by_detection(
                masks[k],
                [0, 0, *masks[k].shape[:2][::-1]],
                zero_pad=True
            )

            masks[k] = cv2.resize(
                masks[k],
                self.shape,
                interpolation=cv2.INTER_NEAREST
            )

        res_masks = np.zeros((self.shape[0], self.shape[1]), dtype=np.int32)
        res_masks += 80     # Set default values as background

        old_masks = np.zeros((80 + 1, self.shape[0], self.shape[1]), dtype=np.uint8)
        old_masks[-1] = np.ones((self.shape[0], self.shape[1]), dtype=np.uint8)     # Set background (for softmax activation)

        sort_mask = list(range(len(labels)))
        sort_mask.sort(key=lambda midx: masks[midx].sum(), reverse=True)

        for i in sort_mask:
            idx = labels[i]

            if idx == -1:
                continue
            label_num = COCO_LABEL_MAP[idx] - 1
            res_masks[masks[i] > 0] = label_num
            old_masks[label_num][masks[i] > 0] = masks[i][masks[i] > 0] / 255.0
            old_masks[-1][masks[i] > 0] = 0     # Disable background for labelled regions (for softmax activation)

        masks_tensor = torch.LongTensor(res_masks)
        old_masks_tensor = torch.FloatTensor(old_masks)

        if self.to_save_masks:
            return res_masks, path

        if self.transforms is not None:
            sample = self.transform(image=image)
            image = sample["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1) / 255.0

        return image, masks_tensor, old_masks_tensor


class COCOImageDataWithMasks(COCOImageData):
    def __init__(
            self,
            annotations_path: str,
            root_path: str,
            masks_path: str,
            transform: Optional[Callable] = None,
            has_gt: bool = True,
            shape: tuple = (700, 700)
    ) -> None:
        super().__init__(annotations_path, root_path, transform, has_gt, shape)

        self.masks_path = masks_path

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        img_id = self.ids[index]

        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        if file_name.startswith('COCO'):
            file_name = file_name.split('_')[-1]

        path = os.path.join(self.root, file_name)
        assert os.path.exists(path), 'Image path does not exist: {}'.format(path)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape

        mask_path = os.path.join(
            self.masks_path,
            os.path.splitext(file_name)[0] + '.npz'
        )

        assert os.path.exists(mask_path), 'Mask path does not exist: {}'.format(path)

        res_masks = np.load(mask_path)['x']

        assert res_masks.shape == tuple(self.shape), \
            'Miss matched shapes: {} and {}'.format(res_masks.shape, self.shape)

        image = create_square_crop_by_detection(
            img,
            [0, 0, *img.shape[:2][::-1]],
            zero_pad=True
        )

        image = cv2.resize(
            image,
            self.shape,
            interpolation=cv2.INTER_AREA
        )

        old_masks = np.zeros((80 + 1, self.shape[0], self.shape[1]), dtype=np.uint8)
        old_masks[-1] = np.ones((self.shape[0], self.shape[1]), dtype=np.uint8)     # Set background (for softmax activation)

        for cls_num in range(80):
            old_masks[cls_num][res_masks == cls_num] = 255

        masks_tensor = torch.LongTensor(res_masks)
        old_masks_tensor = torch.FloatTensor(old_masks)

        if self.transforms is not None:
            sample = self.transform(image=image)
            image = sample["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1) / 255.0

        return image, masks_tensor, old_masks_tensor


if __name__ == '__main__':
    masks_colors = [
        (255, 76, 148),
        (76, 255, 183),
        (76, 148, 255),
        (255, 183, 76)
    ]

    vis_masks = False

    dataset = COCOImageData(
        annotations_path='/home/alexey/programming/upwork/background_removal/third_patry/yolact/data/coco/annotations/instances_train2017.json',
        root_path='/home/alexey/programming/upwork/background_removal/third_patry/yolact/data/coco/images/'
    )

    iwname = 'Image'
    imname = 'Mask'
    cv2.namedWindow(iwname, cv2.WINDOW_NORMAL)
    cv2.namedWindow(imname, cv2.WINDOW_NORMAL)

    for sample_index in range(len(dataset)):
        sample = dataset[sample_index]
        if vis_masks:
            masks_names = ['Mask {}'.format(i) for i in range(len(sample[1][1]))]
            for m in masks_names:
                cv2.namedWindow(m, cv2.WINDOW_NORMAL)

        image = (
                sample[0] * 255.0
        ).permute(1, 2, 0).to('cpu').numpy().astype(np.uint8)

        masks = [
            (sample[2][i] * 255.0).to('cpu').numpy().astype(np.uint8)
            for i in range(len(sample[2]))
        ]

        res_mask = np.zeros((*masks[0].shape[:2], 3), dtype=np.uint8)

        for i, mask in enumerate(masks[:-1]):
            class_index = i # int(sample[1][0][i][-1])
            colorfull_mask = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
            colorfull_mask += np.array(
                masks_colors[class_index % len(masks_colors)],
                dtype=np.uint8
            )

            res_mask[mask > 0] = colorfull_mask[mask > 0]

            image[mask > 0] = (
                    image[mask > 0].astype(np.float16) * 0.2 + colorfull_mask[
                [mask > 0]].astype(np.float16) * 0.8
            ).astype(np.uint8)

        cv2.imshow(iwname, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        cv2.imshow(imname, cv2.cvtColor(res_mask, cv2.COLOR_BGR2RGB))
        if vis_masks:
            for i in range(len(masks)):
                cv2.imshow(masks_names[i], masks[i])

        k = cv2.waitKey(0)
        if vis_masks:
            for m in masks_names:
                cv2.destroyWindow(m)

        if k == 27:
            break

    cv2.destroyWindow(iwname)
    cv2.destroyWindow(imname)