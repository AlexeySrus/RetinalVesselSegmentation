from argparse import ArgumentParser, Namespace
import cv2
import os
import numpy as np
from tqdm import tqdm


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Make submit')
    parser.add_argument('-o', '--output', type=str, help='Path to submit folder')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    target_folder = args.output

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

        th = cnt_areas.mean() / 4

        selected_contours = [
            cnt
            for cnt in contours
            if cv2.contourArea(cnt) > th
        ]

        new_mask = np.zeros_like(init_mask)

        new_mask = cv2.drawContours(new_mask, selected_contours, -1, 255, -1)
        mask = np.bitwise_and(init_mask, new_mask)

        cv2.imwrite(save_path, mask)
