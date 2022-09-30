from typing import List

import os
import glob
import random
import pickle

from data import common

import cv2
import numpy as np
import imageio
import torch
import torch.utils.data as data

class DBData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.train_images_path = '/media/alexey/HDDDataDIsk/datasets/RetinalVesselSegmentation/CompetitionDataset/segmentation_representation/train/images/'
        self.train_masks_path = '/media/alexey/HDDDataDIsk/datasets/RetinalVesselSegmentation/CompetitionDataset/segmentation_representation/train/masks/'
        self.test_images_path = '/media/alexey/HDDDataDIsk/datasets/RetinalVesselSegmentation/CompetitionDataset/segmentation_representation/val/images/'
        self.test_masks_path = '/media/alexey/HDDDataDIsk/datasets/RetinalVesselSegmentation/CompetitionDataset/segmentation_representation/val/masks/'

        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.input_large = (args.model == 'VDSR')
        self.scale = args.scale
        self.idx_scale = 0
        
        self._set_filesystem(args.dir_data)
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        list_hr: List[str] = os.listdir(self.train_images_path)
        if not train:
            list_hr: List[str] = os.listdir(self.test_images_path)

        if args.ext.find('img') >= 0 or benchmark:
            self.images_hr, self.images_lr = list_hr, list_lr
        elif args.ext.find('sep') >= 0:
            os.makedirs(
                self.dir_hr.replace(self.apath, path_bin),
                exist_ok=True
            )
            os.makedirs(
                self.dir_ve.replace(self.apath, path_bin),
                exist_ok=True
            )
            os.makedirs(
                self.dir_ma.replace(self.apath, path_bin),
                exist_ok=True
            )
            self.images_hr = []
            self.images_ve = []
            for h in list_hr:
                self.images_hr.append(
                    os.path.join(self.train_images_path if train else self.test_images_path, h)
                )
                self.images_ve.append(
                    os.path.join(self.train_masks_path if train else self.test_masks_path, h)
                )

        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)

    # Below functions as used to prepare images
    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        return names_hr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_ve = os.path.join(self.apath, 'VE')
        self.dir_ma = os.path.join(self.apath, 'mask')
        self.ext = ('.tif', '.gif')

    def _check_and_load(self, ext, img, f, t, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(cv2.imread(img), _f)

    def __getitem__(self, idx):
        hr, ve, ma, filename = self._load_file(idx)
        pair = self.get_patch(hr, ve, ma)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
        return pair_t, filename

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_ve = self.images_ve[idx]
        # f_ma = self.images_ma[idx]
        filename, _ = os.path.splitext(os.path.basename(f_hr))

        hr = cv2.cvtColor(cv2.imread(f_hr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        ve = cv2.imread(f_ve, cv2.IMREAD_GRAYSCALE)
        ma = np.ones_like(ve) * 255

        return hr, ve, ma, filename

    def get_patch(self, hr, ve, ma):
        scale = self.scale[self.idx_scale]
        if self.train and not self.args.no_augment:
            hr,ve, ma = common.augment(hr,ve, ma)
        data_pack = common.get_patch(
            hr,ve, ma,
            self.args.patch_size,
            True, 
            self.train
        )
        
        return data_pack

    def set_scale(self, idx_scale):
        if not self.input_large:
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)

