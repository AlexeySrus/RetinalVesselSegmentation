from argparse import ArgumentParser, Namespace
import cv2
import numpy as np
import os
import json
from tqdm import tqdm
from shutil import copyfile


def parse_polygon(coordinates, image_size):
    mask = np.zeros(image_size, dtype=np.float32)

    if len(coordinates) == 1:
        points = [np.int32(coordinates)]
        cv2.fillPoly(mask, points, 1)
    else:
        points = [np.int32([coordinates[0]])]
        cv2.fillPoly(mask, points, 1)

        for polygon in coordinates[1:]:
            points = [np.int32([polygon])]
            cv2.fillPoly(mask, points, 0)

    return mask


def parse_mask(shape: dict, image_size: tuple) -> np.ndarray:
    """
    Метод для парсинга фигур из geojson файла
    """
    mask = np.zeros(image_size, dtype=np.float32)
    coordinates = shape['coordinates']
    if shape['type'] == 'MultiPolygon':
        for polygon in coordinates:
            mask += parse_polygon(polygon, image_size)
    else:
        mask += parse_polygon(coordinates, image_size)

    return mask


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description='Convert dataset with geojson to image+mask format'
    )
    parser.add_argument(
        '-i', '--input', type=str,
        help='Path to input folder'
    )
    parser.add_argument(
        '-o', '--output', type=str,
        help='Path to output folder with images/ and masks/ folders'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    images_folder = os.path.join(args.output, 'images/')
    masks_folder = os.path.join(args.output, 'masks/')

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)

    for img_name in tqdm(os.listdir(args.input)):
        if not img_name.endswith('.png'):
            continue

        bname = os.path.splitext(img_name)[0]

        img_path = os.path.join(args.input, img_name)
        dst_image_path = os.path.join(images_folder, img_name)
        dst_mask_path = os.path.join(masks_folder, img_name)
        anns_path = os.path.join(args.input, bname + '.geojson')

        if not os.path.isfile(anns_path):
            print('Skip file: {}'.format(bname))
            continue

        img = cv2.cvtColor(
            cv2.imread(
                img_path,
                cv2.IMREAD_COLOR
            ),
            cv2.COLOR_BGR2RGB
        )

        with open(anns_path, 'r', encoding='cp1251') as f:
            geodata = json.load(f)

        res_mask = np.zeros_like(img[..., 0])
        image_size = res_mask.shape

        if type(geodata) == dict and geodata['type'] == 'FeatureCollection':
            features = geodata['features']
        elif type(geodata) == list:
            features = geodata
        else:
            features = [geodata]

        for shape in features:
            mask = parse_mask(shape['geometry'], image_size)
            res_mask = np.maximum(res_mask, mask)

        res_mask = res_mask * 255
        _, res_mask = cv2.threshold(res_mask, 127, 255, cv2.THRESH_BINARY)

        assert cv2.imwrite(dst_mask_path, res_mask), dst_mask_path
        copyfile(img_path, dst_image_path)
