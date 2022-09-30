import os
import torch
import random
import numpy as np
from functools import reduce
from visdom import Visdom
from torchvision.transforms import ToPILImage, ToTensor
from shutil import copyfile
import torch.nn.functional as F
import cv2


def add_prefix(path, pref):
    """
    Add prefix to file in path
    Args:
        path: path to file
        pref: prefixvectors2line
    Returns:
        path to file with named with prefix
    """
    splitted_path = list(os.path.split(path))
    splitted_path[-1] = pref + splitted_path[-1]
    return reduce(lambda x, y: x + '/' + y, splitted_path)


class AbstractCallback(object):
    def per_batch(self, args):
        raise RuntimeError("Don\'t implement batch callback method")

    def per_epoch(self, args):
        raise RuntimeError("Don\'t implement epoch callback method")

    def early_stopping(self, args):
        raise RuntimeError("Don\'t implement early stopping callback method")


class ModelLogging(AbstractCallback):
    def __init__(self, path, save_step=1, update_file_step=100,
                 columns=None, continue_train=False):
        """
        Callback constructor
        Args:
            path: path to csv file which will contained data
            save_step: parameter of after how many epochs save
            update_file_step: parameter of after how many records
            file will update
            columns: list of columns which will writen in file
            (default: all)
            continue_train: continue model training
            (file was created)
        """

        self.path = path
        self.tmp_path = add_prefix(self.path, 'tmp_')
        self.step = save_step
        self.update_step = update_file_step
        self.columns = columns

        self.logfile = open(
            self.tmp_path,
            'a' if continue_train else 'w'
        )

        self.make_title = True

    def save(self):
        self.logfile.close()
        copyfile(self.tmp_path, self.path)
        self.logfile = open(self.tmp_path, 'a')

    def __del__(self):
        self.save()
        os.remove(self.tmp_path)

    def per_batch(self, args):
        pass

    def per_epoch(self, args):
        if self.make_title:
            self.logfile.write(', '.join(
                args.keys()
                if self.columns is None
                else
                self.columns
            ) + '\n')
            self.make_title = False

        if args['n'] % self.step == 0:
            if self.columns is None:
                ws = ', '.join([str(v) for v in args.values()])
            else:
                ws = ', '.join([str(args[item]) for item in self.columns])

            self.logfile.write(ws + '\n')

        if args['n'] % self.update_step == 0:
            self.save()

    def early_stopping(self, args):
        pass


class SaveModelPerEpoch(AbstractCallback):
    def __init__(self, path, save_step=1):
        self.path = path
        self.step=save_step

        if not os.path.isdir(path):
            os.makedirs(path)

    def per_batch(self, args):
        pass

    def per_epoch(self, args):
        if args['n'] % self.step == 0:
            args['model'].save(
                os.path.join(self.path, 'model-{}.trh'.format(args['n']))
            )

    def early_stopping(self, args):
        args['model'].save(
            os.path.join(self.path, 'early_model-{}.trh'.format(args['n']))
        )


class SaveOptimizerPerEpoch(AbstractCallback):
    def __init__(self, path, save_step=1):
        self.path = path
        self.step=save_step

        if not os.path.isdir(path):
            os.makedirs(path)

    def per_batch(self, args):
        pass

    def per_epoch(self, args):
        if args['n'] % self.step == 0:
            torch.save(args['optimize_state'], (
                os.path.join(
                    self.path,
                    'optimize_state-{}.trh'.format(args['n'])
                )
            ))

    def early_stopping(self, args):
        torch.save(args['optimize_state'], (
            os.path.join(
                self.path,
                'early_optimize_state-{}.trh'.format(args['n'])
            )
        ))


class VisPlot(AbstractCallback):
    def __init__(self, title, server='https://localhost', port=8080,
                 logname=None):
        self.viz = Visdom(server=server, port=port, log_to_filename=logname)
        self.windows = {}
        self.title = title

    def register_scatterplot(self, name, xlabel, ylabel, legend=None):
        options = dict(title=self.title, markersize=5,
                        xlabel=xlabel, ylabel=ylabel) if legend is None \
                       else dict(title=self.title, markersize=5,
                        xlabel=xlabel, ylabel=ylabel,
                        legend=legend)

        self.windows[name] = [None, options]

    def update_scatterplot(self, name, x, y1, y2=None, window_size=100):
        """
        Update plot
        Args:
            name: name of updating plot
            x: x values for plotting
            y1: y values for plotting
            y2: plot can contains two graphs
            window_size: window size for plot smoothing (by mean in window)
        Returns:
        """
        if y2 is None:
            self.windows[name][0] = self.viz.line(
                np.array([y1], dtype=np.float32),
                np.array([x], dtype=np.float32),
                win=self.windows[name][0],
                opts=self.windows[name][1],
                update='append' if self.windows[name][0] is not None else None
            )
        else:
            self.windows[name][0] = self.viz.line(
                np.array([[y1, y2]], dtype=np.float32),
                np.array([x], dtype=np.float32),
                win=self.windows[name][0],
                opts=self.windows[name][1],
                update='append' if self.windows[name][0] is not None else None
            )

    def per_batch(self, args, keyward='per_batch'):
        for win in self.windows.keys():
            if keyward in win:
                if 'train' in win:
                    self.update_scatterplot(
                        win,
                        args['n'],
                        args['loss']
                    )

    def per_epoch(self, args, keyward='per_epoch'):
        for win in self.windows.keys():
            if keyward in win:
                if 'train' in win and 'validation' in win and 'acc' not in win:
                    self.update_scatterplot(
                        win,
                        args['n'],
                        args['loss'],
                        args['val loss']
                    )

                if 'train' in win and 'validation' in win and 'acc' in win:
                    self.update_scatterplot(
                        win,
                        args['n'],
                        # args['acc'],
                        args['val acc']
                    )

    def early_stopping(self, args):
        pass


class VisImage(AbstractCallback):
    masks_colors = torch.FloatTensor([
        (3, 169, 244),
        (244, 67, 54),
        (233, 30, 99),
        (156, 39, 176),
        (103, 58, 183),
        (63, 81, 181),
        (33, 150, 243),
        (0, 188, 212),
        (0, 150, 136),
        (76, 175, 80),
        (139, 195, 74),
        (205, 220, 57),
        (255, 235, 59),
        (255, 193, 7),
        (255, 152, 0),
        (255, 87, 34),
        (121, 85, 72),
        (158, 158, 158),
        (96, 125, 139)
    ]).unsqueeze(1).permute(0, 2, 1) / 255
    min_th = 0.5

    def __init__(self, title, server='https://localhost', port=8080,
                 vis_step=1, scale=10, use_mdn=False, coefs=30, samples=2):
        self.viz = Visdom(server=server, port=port)
        self.title = title + 'Image'
        self.windows = {1: None}
        self.n = 0
        self.step = vis_step
        self.scale = scale

        self.to_image = ToPILImage()
        self.to_tensor = ToTensor()

        self.use_mdn = use_mdn
        self.coefs = coefs
        self.samples = samples

        random.seed()

    def per_batch(self, args, label=1):
        if self.n % self.step == 0:
            i = random.randint(0, args['img'].size(0) - 1)

            # mean = torch.FloatTensor([111.65964692, 101.50364134,  90.17587756]).unsqueeze(1).unsqueeze(1) / 255
            # std = torch.FloatTensor([43.16146952, 46.91914326, 46.32995654]).unsqueeze(1).unsqueeze(1) / 255
            #
            # mean = 0
            # std = 1

            # mean = torch.FloatTensor([187.37888289, 60.40710529, 30.97342565]).unsqueeze(1).unsqueeze(1) / 255
            # std = torch.FloatTensor([34.97379752, 20.30510642, 11.72539536]).unsqueeze(1).unsqueeze(1) / 255

            mean = torch.FloatTensor([0.18879161, 0.16486329, 0.14568384]).unsqueeze(1).unsqueeze(1)
            std = torch.FloatTensor([0.22082106, 0.21477145, 0.19575508]).unsqueeze(1).unsqueeze(1)

            for win in self.windows.keys():
                if win == label:
                    img = args['img'][i].to(
                        'cpu'
                    )
                    img = (img * std + mean)
                    # print(img.min(), img.max())
                    original_mask = args['mask_true'][i].to(
                        'cpu'
                    )
                    predicted_masks_indexes = args['mask_pred'][i].detach().to(
                        'cpu'
                    ).argmax(dim=0)

                    # predicted_masks = torch.zeros_like(original_masks)
                    # for ch in range(1):
                    #     predicted_masks[ch][predicted_masks_indexes == ch] = 1.0

                    # sort_mask = list(range(original_masks.size(0)))
                    # sort_mask.sort(key=lambda midx: original_masks[midx].sum(), reverse=True)

                    original_masks_with_image = torch.clone(img)
                    predicted_masks_with_image = torch.clone(img)
                    original_masks_only = torch.ones_like(img) / 2
                    predicted_masks_only = torch.ones_like(img) / 2

                    for ch in range(1):
                        original_masks_with_image[:,
                            original_mask == ch] = \
                                    original_masks_with_image[:,
                                original_mask == ch] * 0.5 + \
                            self.masks_colors[ch % len(self.masks_colors)] * 0.5

                        predicted_masks_with_image[:,
                                predicted_masks_indexes == ch] = \
                            predicted_masks_with_image[:, predicted_masks_indexes == ch] * 0.5 + \
                            self.masks_colors[ch % len(self.masks_colors)] * 0.5

                        original_masks_only[:, original_mask == ch] = self.masks_colors[ch % len(self.masks_colors)]
                        predicted_masks_only[:,predicted_masks_indexes == ch] = self.masks_colors[ch % len(self.masks_colors)]

                    img = torch.cat(
                        (
                            original_masks_with_image,
                            predicted_masks_with_image,
                            original_masks_only,
                            predicted_masks_only
                        ),
                        dim=2
                    )

                    img = torch.clamp(img, 0, 1)

                    self.windows[win] = self.viz.image(
                        F.interpolate(
                            img.unsqueeze(0),
                            scale_factor=(self.scale, self.scale)
                        ).squeeze(0),
                        win=self.windows[win],
                        opts=dict(title=self.title)
                    )

        self.n += 1
        if self.n >= 1000000000:
            self.n = 0

    def per_epoch(self, args):
        pass

    def early_stopping(self, args):
        pass

    def add_window(self, label):
        self.windows[label] = None
