import cv2
import os
import numpy as np
from tqdm import tqdm

target_folder = '/media/alexey/HDDDataDIsk/datasets/RetinalVesselSegmentation/CompetitionDataset/submit/'

input_names = [os.path.splitext(p)[0] for p in os.listdir(target_folder) if p.endswith('.npy')]

for basename in tqdm(input_names):
    save_path = os.path.join(target_folder, basename + '.png')
    npy_path = os.path.join(target_folder, basename + '.npy')

    out = np.load(npy_path)

    pred_mask = out > 0.4
    pred_mask = (pred_mask * 255).astype(np.uint8)

    mask = pred_mask

    init_mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    contours, hierarchy = cv2.findContours(init_mask, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    cnt_areas = np.array([cv2.contourArea(cnt) for cnt in contours])

    th = cnt_areas.mean()

    selected_contours = [
        cnt
        for cnt in contours
        if cv2.contourArea(cnt) > th
    ]

    new_mask = np.zeros_like(init_mask)

    new_mask = cv2.drawContours(new_mask, selected_contours, -1, 255, -1)
    mask = np.bitwise_and(init_mask, new_mask)

    cv2.imwrite(save_path, mask)
