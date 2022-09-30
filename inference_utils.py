import torch
from PIL import Image
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from enum import Enum


class TensorRotate(Enum):
    """Rotate enumerates class"""
    NONE = lambda x: x
    ROTATE_90_CLOCKWISE = lambda x: x.transpose(1, 2).flip(2)
    ROTATE_180 = lambda x: x.flip(1, 2)
    ROTATE_90_COUNTERCLOCKWISE = lambda x: x.transpose(1, 2).flip(1)
    HORIZONTAL_FLIP = lambda x: x.flip(2)
    VERTICAL_FLIP = lambda x: x.flip(1)


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
    ]

    inv_transforms = [
        TensorRotate.NONE,
        TensorRotate.ROTATE_90_COUNTERCLOCKWISE,
        TensorRotate.ROTATE_180,
        TensorRotate.ROTATE_90_CLOCKWISE,
        TensorRotate.HORIZONTAL_FLIP,
        TensorRotate.VERTICAL_FLIP,
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


def np2Tensor(*args, rgb_range=255, single_test=False):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        if single_test:
            np_transpose = np.expand_dims(np_transpose, 0)
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]


def postprocess_mask(_mask, morph_r=3, th_k=2):
    if morph_r is not None:
        init_mask = cv2.morphologyEx(
            _mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_r, morph_r))
        )
    else:
        init_mask = _mask.copy()

    contours, hierarchy = cv2.findContours(init_mask, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    cnt_areas = np.array([cv2.contourArea(cnt) for cnt in contours])

    if th_k is not None:
        th = cnt_areas.mean() / th_k
    else:
        th = 50

    selected_contours = [
        cnt
        for cnt in contours
        if cv2.contourArea(cnt) > th
    ]

    new_mask = np.zeros_like(init_mask)

    new_mask = cv2.drawContours(new_mask, selected_contours, -1, 255, -1)
    return np.bitwise_and(init_mask, new_mask)


def inference_tiling_intersected(
        img: torch.Tensor, single_inference: callable, tile_size=256, stride_k=2
) -> torch.Tensor:
    """
    Process the image with splitting on tiles.
    `singel_inferece` will be applied to each tile. Its expected the input
    image is torch.Tensor [C, H, W] shape of float type.
    """
    res_mask = torch.zeros(img.size(1), img.size(2), dtype=torch.float32, device=img.device)
    counter_mask = torch.zeros(img.size(1), img.size(2), dtype=torch.long, device=img.device)

    stride = tile_size // stride_k

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
            img_crop = img[:, y0:y0 + tile_size, x0:x0 + tile_size]
            # res = single_inference(img_crop.unsqueeze(0)).squeeze(0)
            res = tta_inference(img_crop, single_inference)
            res_mask[y0:y0 + tile_size, x0:x0 + tile_size] += res[0]
            counter_mask[y0:y0 + tile_size, x0:x0 + tile_size] += 1

    return res_mask / counter_mask


class SegmentationInference(object):
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self.device = device

        mean = [0.18879161, 0.16486329, 0.14568384]
        std = [0.22082106, 0.21477145, 0.19575508]

        self.transform = A.Compose(
            [
                A.Normalize(mean=mean, std=std, always_apply=True),
                ToTensorV2(transpose_mask=False)
            ]
        )

    def __call__(self, input_data: Image) -> Image:
        np_image = np.array(input_data.convert('RGB'))

        sample = self.transform(image=np_image)['image']

        img_tensor = sample.to(self.device)

        with torch.no_grad():
            out = inference_tiling_intersected(
                img_tensor,
                lambda _x: torch.softmax(self.model(_x), dim=1)[:, 0:1]
            ).to('cpu')

        pred_mask = out > 0.4
        pred_mask = (pred_mask * 255).numpy().astype(np.uint8)
        pred_mask = postprocess_mask(pred_mask)

        output = Image.fromarray(pred_mask)

        return output


class SGLInference(object):
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.sgl_model = torch.jit.load(model_path, map_location=device)
        self.sgl_model.eval()
        self.device = device

    def __call__(self, input_data: Image) -> Image:
        np_image = np.array(input_data.convert('RGB'))

        img_tensor = np2Tensor(np_image)[0]

        imgt = torch.zeros(1, 3, 2000, 2000)
        imgt[0, :, :img_tensor.size(1), :img_tensor.size(2)] = img_tensor
        imgt = imgt.to(self.device)

        with torch.no_grad():
            enchance, est_o = self.sgl_model(
                imgt,
                torch.LongTensor([1]).to(self.device)
            )
            est_o = est_o.cpu()[0, 0, :img_tensor.size(1), :img_tensor.size(2)]

        eomask = (est_o * 255.).numpy().astype(np.uint8)
        eomask = ((eomask > 100) * 255).astype(np.uint8)

        pred_mask = postprocess_mask(eomask, None, None)

        output = Image.fromarray(pred_mask)

        output.save('AAAAMASKL.png')

        return output
