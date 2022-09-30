import cv2

import torch
import torch.nn.functional as F
import os
import imageio
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import albumentations as A

from NetworkTrainer.networks.unet import UNet
from NetworkTrainer.networks.resunet import ResUNet, ResUNet_ds
from NetworkTrainer.networks.denseunet import DenseUNet
from NetworkTrainer.networks.vit_seg_modeling import \
    VisionTransformer as ViT_seg
from NetworkTrainer.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from NetworkTrainer.dataloaders.dataload import DataFolder, TestEyeMaskDataset
from NetworkTrainer.utils.util import AverageMeterArray
from NetworkTrainer.utils.accuracy import compute_metrics
from NetworkTrainer.utils.post_process import *


def inference_tiling_intersected(
        img: torch.Tensor, single_inference: callable, tile_size=256, stride_k=2
) -> torch.Tensor:
    """
    Process the image with splitting on tiles.
    `singel_inferece` will be applied to each tile. Its expected the input
    image is torch.Tensor [C, H, W] shape of float type.
    """
    res_mask = torch.zeros(2, img.size(1), img.size(2), dtype=torch.float32, device=img.device)
    counter_mask = torch.zeros(2, img.size(1), img.size(2), dtype=torch.long, device=img.device)

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
            res = single_inference(img_crop.unsqueeze(0)).squeeze(0)
            res_mask[:, y0:y0 + tile_size, x0:x0 + tile_size] += res
            counter_mask[:, y0:y0 + tile_size, x0:x0 + tile_size] += 1

    return res_mask / counter_mask


def postprocess_mask(_mask, morph_r = 3, th_k=2.0):
    init_mask = cv2.morphologyEx(
        _mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_r, morph_r))
    )

    contours, hierarchy = cv2.findContours(init_mask, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    cnt_areas = np.array([cv2.contourArea(cnt) for cnt in contours])

    th = cnt_areas.mean() / th_k

    selected_contours = [
        cnt
        for cnt in contours
        if cv2.contourArea(cnt) > th
    ]

    new_mask = np.zeros_like(init_mask)

    new_mask = cv2.drawContours(new_mask, selected_contours, -1, 255, -1)
    return np.bitwise_and(init_mask, new_mask)


class NetworkInference:
    def __init__(self, opt):
        self.opt = opt

    def set_GPU_device(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
            str(x) for x in self.opt.test['gpus'])

    def set_network(self):
        if 'res' in self.opt.model['name']:
            self.net = ResUNet(net=self.opt.model['name'], seg_classes=2,
                               colour_classes=3,
                               pretrained=self.opt.model['pretrained'])
            if self.opt.train['deeps']:
                self.net = ResUNet_ds(net=self.opt.model['name'], seg_classes=2,
                                      colour_classes=3,
                                      pretrained=self.opt.model['pretrained'])
        elif 'dense' in self.opt.model['name']:
            self.net = DenseUNet(net=self.opt.model['name'], seg_classes=2)
        elif 'trans' in self.opt.model['name']:
            config_vit = CONFIGS_ViT_seg[self.opt.model['name']]
            config_vit.n_classes = 2
            config_vit.n_skip = 4
            if self.opt.model['name'].find('R50') != -1:
                config_vit.patches.grid = (
                int(self.opt.model['input_size'][0] / 16),
                int(self.opt.model['input_size'][1] / 16))
            self.net = ViT_seg(config_vit,
                               img_size=self.opt.model['input_size'][0],
                               num_classes=config_vit.n_classes).cuda()
        else:
            self.net = UNet(3, 2, 2)
        self.net = torch.nn.DataParallel(self.net)
        self.net = self.net.cuda()

        # ----- load trained model ----- #
        print(f"=> loading trained model in {self.opt.test['model_path']}")
        checkpoint = torch.load(self.opt.test['model_path'])
        state_dict = self.net.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if
                           k in state_dict}
        state_dict.update(pretrained_dict)
        self.net.load_state_dict(state_dict)

        # self.net.load_state_dict(checkpoint['state_dict'])
        print("=> loaded model at epoch {}".format(checkpoint['epoch']))
        self.net = self.net.module
        self.net.eval()

    def set_dataloader(self):
        test_set = TestEyeMaskDataset(root_dir=self.opt.root_dir, phase=None,
                                  data_transform=A.Compose(
                                      self.opt.transform['test']),
                                  fold=self.opt.fold)
        self.test_loader = DataLoader(test_set,
                                      batch_size=self.opt.test['batch_size'],
                                      shuffle=False, drop_last=False)

    def set_save_dir(self):
        if self.opt.test['save_flag']:
            if not os.path.exists(
                    os.path.join(self.opt.test['save_dir'], 'submit')):
                os.mkdir(os.path.join(self.opt.test['save_dir'], 'submit'))

    def post_process(self, pred):
        if self.opt.post['abl']:
            pred = abl(pred, for_which_classes=[1])
        if self.opt.post['rsa']:
            # pred = rsa(pred, for_which_classes=[1], minimum_valid_object_size={1: 120})
            pred = rsa(pred, for_which_classes=[1],
                       minimum_valid_object_size={1: 10})
        return pred

    def run(self):
        for i, data in enumerate(tqdm(self.test_loader)):
            input, shifts, sizes, name = data['image'].cuda(), data['shifts'], data['size'], data['name']
            tta = TTA_2d(flip=self.opt.test['flip'],
                         rotate=self.opt.test['rotate'])
            input_list = tta.img_list(input)
            y_list = []
            for x in input_list:
                x = torch.from_numpy(x.copy()).cuda()
                with torch.no_grad():
                    if not self.opt.train['deeps']:
                        # y = self.net(x)

                        y = inference_tiling_intersected(x.squeeze(0), self.net, stride_k=2).unsqueeze(0)
                    else:
                        # y = self.net(x)[0]
                        y = inference_tiling_intersected(x, lambda _x: self.net(_x)[0])

                y = torch.nn.Softmax(dim=1)(y)[:, 1]
                y = y.cpu().detach().numpy()
                y_list.append(y)
            y_list = tta.img_list_inverse(y_list)
            output = np.mean(y_list, axis=0)
            # pred = (output > 0.5).astype(np.uint8)

            for j in range(output.shape[0]):
                sample_pred = output[j]

                sample_shifts = shifts[j].to('cpu').numpy()
                sx = abs(int(sample_shifts[0]))
                sy = abs(int(sample_shifts[1]))
                original_size = sizes[j].to('cpu').numpy().tolist()    # w, h

                max_size = max(original_size)

                # sample_pred = F.interpolate(torch.FloatTensor([[sample_pred]]), (max_size, max_size), mode='bicubic')[0][0].numpy()
                sample_pred = (sample_pred > 0.5).astype(np.uint8)
                sample_pred = self.post_process(sample_pred) * 255
                # sample_pred = postprocess_mask(sample_pred.astype(np.uint8), th_k=1.0)

                # sample_pred = sample_pred[sy:sy + original_size[1], sx:sx + original_size[0]]

                if self.opt.test['save_flag']:
                    imageio.imwrite(
                        os.path.join(self.opt.test['save_dir'], 'submit',
                                     f'{name[j]}.png'),
                        sample_pred)
