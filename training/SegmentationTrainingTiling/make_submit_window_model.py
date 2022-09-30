import torch
import torch.nn.functional as F

from enum import Enum


class TensorRotate(Enum):
    """Rotate enumerates class"""
    NONE = lambda x: x
    ROTATE_90_CLOCKWISE = lambda x: x.transpose(1, 2).flip(2)
    ROTATE_180 = lambda x: x.flip(1, 2)
    ROTATE_90_COUNTERCLOCKWISE = lambda x: x.transpose(1, 2).flip(1)
    HORIZONTAL_FLIP = lambda x: x.flip(2)
    VERTICAL_FLIP = lambda x: x.flip(1)
    RGB_SHIFT = lambda x: x.flip(0)


def rotate_tensor(img: torch.Tensor, rot_value: TensorRotate) -> torch.Tensor:
    """Rotate image tensor
    Args:
        img: tensor in CHW format
        rot_value: element of TensorRotate class, possible values
            TensorRotate.NONE,
            TensorRotate.ROTATE_90_CLOCKWISE,
            TensorRotate.ROTATE_180,
            TensorRotate.ROTATE_90_COUNTERCLOCKWISE,
    Returns:
        Rotated image in same of input format
    """
    return rot_value(img)


def tta_inference(_t: torch.Tensor, _model: torch.nn.Module) -> torch.Tensor:
    transforms = [
        TensorRotate.NONE,
        TensorRotate.ROTATE_90_CLOCKWISE,
        TensorRotate.ROTATE_180,
        TensorRotate.ROTATE_90_COUNTERCLOCKWISE,
        TensorRotate.HORIZONTAL_FLIP,
        TensorRotate.VERTICAL_FLIP,
        # TensorRotate.RGB_SHIFT
    ]

    inv_transforms = [
        TensorRotate.NONE,
        TensorRotate.ROTATE_90_COUNTERCLOCKWISE,
        TensorRotate.ROTATE_180,
        TensorRotate.ROTATE_90_CLOCKWISE,
        TensorRotate.HORIZONTAL_FLIP,
        TensorRotate.VERTICAL_FLIP,
        # TensorRotate.RGB_SHIFT
    ]

    tta_batch = torch.stack(
        [rotate_tensor(_t, tr) for tr in transforms]
    )

    out = _model(tta_batch).detach()

    avg_out = torch.stack(
        [
            rotate_tensor(out[tri], tr)
            for tri, tr in enumerate(inv_transforms)
        ]
    )

    return avg_out.mean(dim=0)


WINDOW_SIZE = 256
INPUT_SIZE = None


def inference_tiling_intersected(
        img: torch.Tensor, single_inference: torch.nn.Module, tile_size=WINDOW_SIZE, input_size=INPUT_SIZE
) -> torch.Tensor:
    """
    Process the image with splitting on tiles.
    `singel_inferece` will be applied to each tile. Its expected the input
    image is torch.Tensor [C, H, W] shape of float type.
    """
    res_mask = torch.zeros(img.size(1), img.size(2), dtype=torch.float32, device=img.device)
    counter_mask = torch.zeros(img.size(1), img.size(2), dtype=torch.long, device=img.device)

    stride = tile_size // 2

    x0_vec = []
    y0_vec = []

    target_x = 0
    while target_x + tile_size < img.size(2):
        x0_vec.append(target_x)
        target_x += stride
    x0_vec.append(img.size(2) - tile_size - 1)

    target_y = 0
    while target_y + tile_size < img.size(1):
        y0_vec.append(target_y)
        target_y += stride
    y0_vec.append(img.size(1) - tile_size - 1)

    for y0 in y0_vec:
        for x0 in x0_vec:
            img_crop = img[:, y0:y0 + tile_size, x0:x0 + tile_size]# .unsqueeze(0)
            # res = single_inference(img_crop).detach()[0][1]     # Because model returns [BS, 2, H, W]
            # resized_crop = F.interpolate(img_crop.unsqueeze(0), (input_size, input_size)).squeeze(0)
            res = tta_inference(img_crop, single_inference)
            # res = single_inference(img_crop.unsqueeze(0)).squeeze(0)
            # resized_res = F.interpolate(res.unsqueeze(0), (tile_size, tile_size)).squeeze(0)
            resized_res = res
            resized_res = resized_res[-1]
            res_mask[y0:y0 + tile_size, x0:x0 + tile_size] += resized_res
            counter_mask[y0:y0 + tile_size, x0:x0 + tile_size] += 1

    return res_mask / counter_mask


import torch
import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm


input_folder = '/media/alexey/HDDDataDIsk/datasets/RetinalVesselSegmentation/CompetitionDataset/test/'
target_folder = '/media/alexey/HDDDataDIsk/datasets/RetinalVesselSegmentation/CompetitionDataset/submit/'

os.makedirs(target_folder, exist_ok=True)

input_names = [os.path.splitext(p)[0] for p in os.listdir(input_folder)]


mean = [0.18879161, 0.16486329, 0.14568384]
std = [0.22082106, 0.21477145, 0.19575508]

# mean = np.array([111.65964692, 101.50364134,  90.17587756]) / 255
# std = np.array([43.16146952, 46.91914326, 46.32995654]) / 255

transforms = A.Compose(
    [
        A.Normalize(mean=mean, std=std, always_apply=True),
        ToTensorV2(transpose_mask=False)
    ]
)

device = 'cuda'
# model_path = '../third_patry/SA_Uet-pytorch/weights/new_data_pt_best_model_traced_256.pt'
model_path = '/media/alexey/SSDDataDisk/experiments/RetinalVesselSegmentation/CompetitionDatasetTiling/MANet_efficientnetv2_s_256/last_state_trace.pt'
model = torch.jit.load(model_path).to(device)
_ = model.eval()

for basename in tqdm(input_names):
    save_path = os.path.join(target_folder, basename + '.png')
    image_path = os.path.join(input_folder, basename + '.png')
    npy_path = os.path.join(target_folder, basename + '.npy')

    if os.path.exists(npy_path):
        continue

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_img_shape = img.shape[:2]

    sample = transforms(image=img)

    img_tensor = sample["image"].to(device)
    with torch.no_grad():
        # out = inference_tiling_intersected(img_tensor, model).to('cpu')
        out = inference_tiling_intersected(img_tensor, lambda _x: torch.softmax(model(_x), dim=1)[:, 0:1]).to('cpu')

    np.save(
        npy_path,
        out.numpy()
    )

    pred_mask = out > 0.4
    pred_mask = (pred_mask * 255).numpy().astype(np.uint8)

    # pred_mask = cv2.resize(
    #     pred_mask,
    #     (max(input_img_shape), max(input_img_shape)),
    #     interpolation=cv2.INTER_NEAREST
    # )

    # sx = abs(sx)
    # sy = abs(sy)
    #
    # mask = pred_mask[sy:sy + input_img_shape[0], sx:sx + input_img_shape[1]]
    mask = pred_mask
    cv2.imwrite(save_path, mask)
