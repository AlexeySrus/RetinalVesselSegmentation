import torch
import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm

from coco_dataset import create_square_crop_by_detection


sample_folder = '/home/alexey/programming/squanch_work/RetinalVesselSegmentation/competition_solution/sample_submit/'
input_folder = '/media/alexey/HDDDataDIsk/datasets/RetinalVesselSegmentation/CompetitionDataset/test/'
target_folder = '/media/alexey/HDDDataDIsk/datasets/RetinalVesselSegmentation/CompetitionDataset/submit/'

os.makedirs(target_folder, exist_ok=True)

sample_names = [os.path.splitext(p)[0] for p in os.listdir(sample_folder)]
input_names = [os.path.splitext(p)[0] for p in os.listdir(input_folder)]

U = set(sample_names).union(input_names)
I = set(sample_names).intersection(input_names)


for basename in set(sample_names).difference(I):
    save_path = os.path.join(target_folder, basename + '.png')
    image_path = os.path.join(sample_folder, basename + '.png')

    img = cv2.imread(image_path)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.imwrite(save_path, mask)


mean = np.array([111.65964692, 101.50364134,  90.17587756]) / 255
std = np.array([43.16146952, 46.91914326, 46.32995654]) / 255

transforms = A.Compose(
    [
        A.Normalize(mean=mean.tolist(), std=std.tolist(), always_apply=True),
        ToTensorV2(transpose_mask=False)
    ]
)

device = 'cuda'
model_path = '/media/alexey/SSDDataDisk/experiments/RetinalVesselSegmentation/CompetitionDataset/unet_mit_b1/last_state_trace.pt'
model = torch.jit.load(model_path).to(device)
_ = model.eval()

for basename in tqdm(I):
    save_path = os.path.join(target_folder, basename + '.png')
    image_path = os.path.join(input_folder, basename + '.png')

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_img_shape = img.shape[:2]

    img, (sx, sy) = create_square_crop_by_detection(
        img,
        [0, 0, *img.shape[:2][::-1]],
        return_shifts=True,
        zero_pad=True
    )

    img = cv2.resize(
        img,
        [1024, 1024],
        interpolation=cv2.INTER_AREA
    )

    sample = transforms(image=img)
    img_tensor = sample["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_tensor)

    pred_mask = 1 - out.argmax(dim=1)[0].to('cpu')
    pred_mask = (pred_mask * 255).numpy().astype(np.uint8)

    pred_mask = cv2.resize(
        pred_mask,
        (max(input_img_shape), max(input_img_shape)),
        interpolation=cv2.INTER_NEAREST
    )

    sx = abs(sx)
    sy = abs(sy)

    mask = pred_mask[sy:sy + input_img_shape[0], sx:sx + input_img_shape[1]]
    cv2.imwrite(save_path, mask)
