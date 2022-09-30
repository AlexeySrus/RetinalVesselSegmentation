import torch.utils.data as data

import cv2
import albumentations as A
from PIL import Image
import numpy as np

import os
import os.path


def create_square_crop_by_detection(
        frame: np.ndarray,
        box: list,
        return_shifts: bool = False,
        zero_pad: bool = False):
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


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, extensions):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                item = (path, 0)
                images.append(item)

    return images


class DatasetFolder(data.Dataset):
    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        # classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


image_transforms = A.Compose([
    # A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=35, border_mode=cv2.BORDER_CONSTANT, value=0),
    # A.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT, shift_limit=0, value=0),
    # A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, value=0),
    # A.ElasticTransform(alpha=5, border_mode=cv2.BORDER_CONSTANT, alpha_affine=0, value=0),
    # A.Perspective(p=0.5),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.ImageCompression(quality_lower=75, quality_upper=100, p=0.8),
])


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def pad_loader(path: str) -> Image:
    assert os.path.isfile(path), 'Can\'t find file: {}'.format(path)

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    timg = image_transforms(image=img)['image']

    if np.random.randint(0, 10) > 7:
        min_dim_size = min(timg.shape[:2])

        x_shift = np.random.randint(0, timg.shape[1] - min_dim_size + 1)
        y_shift = np.random.randint(0, timg.shape[0] - min_dim_size + 1)

        cropped_img = timg[y_shift:y_shift+min_dim_size, x_shift:x_shift+min_dim_size]
    else:
        cropped_img = create_square_crop_by_detection(
            timg,
            [0, 0, *timg.shape[:2][::-1]],
            zero_pad=True
        )

    return Image.fromarray(cropped_img)


def default_loader(path):
    return pad_loader(path)


class ImageFolder(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples
