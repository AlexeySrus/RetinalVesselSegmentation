import cv2
import torch
import yaml
import argparse
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm
from apex import amp
import os
import numpy as np
from functools import reduce

from segm_dataset import MaskDataset
from callbacks import VisPlot, VisImage
from losses import GeneralizedSoftDiceLoss, CE_DiceLoss, binary_f1_score, SoftIoULoss, FocalLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Segmentation train')
    parser.add_argument('--config', required=False, type=str,
                          default='train_config.yml',
                          help='Path to configuration yml file.'
                        )
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    return parser.parse_args()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def lr_f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_f)


if __name__ == '__main__':
    args = parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    batch_size = config['train']['batch_size']
    n_jobs = config['train']['num_workers']
    epochs = config['train']['epochs']
    start_epoch = 1
    use_apex = config['train']['use_apex']
    prev_f1_path = os.path.join(config['train']['save']['model'], 'best_f1_score.txt')
    gradient_accumulation_steps = config['train']['gradient_accumulation_steps']

    plot_visualizer = VisPlot(
        title='Training curves',
        server=config['visualization']['visdom_server'],
        port=config['visualization']['visdom_port']
    )

    plot_visualizer.register_scatterplot(
        name='train validation loss per_epoch',
        xlabel='Epoch',
        ylabel='LOSS',
        legend=['train', 'val']
    )

    plot_visualizer.register_scatterplot(
        name='train validation acc per_epoch',
        xlabel='Epoch',
        ylabel='F1 Score',
        legend=['val']
    )

    image_visualizer = VisImage(
        'Image visualisation',
        config['visualization']['visdom_server'],
        config['visualization']['visdom_port'],
        config['visualization']['image']['every'],
        scale=config['visualization']['image']['scale']
    )

    model = smp.MAnet(
        encoder_name=config['model']['net'],
        in_channels=config['model']['input_channels'],
        classes=config['model']['model_classes'],
        encoder_weights='imagenet',
        activation=None
    )

    latest_path = os.path.join(
        config['train']['save']['model'],
        'last_state.pt'
    )

    model = model.to(device)

    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=config['train']['lr'],
    #     weight_decay=config['train']['weight_decay']
    # )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['train']['lr'],
        momentum=0.9,
        weight_decay=config['train']['weight_decay']
    )

    ckp = None
    if os.path.exists(latest_path):
        print('Load from: {}'.format(latest_path))
        ckp = torch.load(latest_path)
        model.load_state_dict(ckp['model'])
        # optimizer.load_state_dict(ckp['optimizer'])
        # start_epoch = ckp['epoch']

        optimizer.param_groups[0]['lr'] = config['train']['lr']

    # mean = np.array([187.37888289, 60.40710529, 30.97342565]) / 255
    # std = np.array([34.97379752, 20.30510642, 11.72539536]) / 255

    # mean = np.array([111.65964692, 101.50364134,  90.17587756]) / 255
    # std = np.array([43.16146952, 46.91914326, 46.32995654]) / 255

    mean = [0.18879161, 0.16486329, 0.14568384]
    std = [0.22082106, 0.21477145, 0.19575508]

    train_transforms = A.Compose(
        [
            A.RandomCrop(*tuple(config['dataset']['shape']), always_apply=True),
            A.Normalize(mean=mean, std=std, always_apply=True),
            ToTensorV2(transpose_mask=False)
        ]
    )
    # train_transforms = None
    val_transforms = train_transforms

    train_augmentations = None
    if config['dataset']['use_augmentations']:
        train_augmentations = A.Compose(
            [
                A.RandomCrop(int(config['dataset']['shape'][0] * 2), int(config['dataset']['shape'][0] * 2), always_apply=True),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.ISONoise(p=0.1),
                # A.GaussianBlur(p=0.5),
                A.Rotate(limit=15, p=0.2, border_mode=cv2.BORDER_CONSTANT,
                         mask_value=1),
                A.Affine(scale=(0.9, 1.2), p=0.5),
                A.Perspective(mask_pad_val=1, p=0.2),
                A.ElasticTransform(p=0.2, mask_value=1,
                                   border_mode=cv2.BORDER_CONSTANT),
                A.GridDistortion(p=0.2, mask_value=1,
                                 border_mode=cv2.BORDER_CONSTANT),
                A.OpticalDistortion(p=0.2, mask_value=1,
                                    border_mode=cv2.BORDER_CONSTANT),
                # A.CLAHE(p=0.5),
                # A.ColorJitter(p=0.5),
                # A.RandomBrightnessContrast(p=0.5),
                # A.RandomGamma(p=0.2),
                # A.ImageCompression(quality_lower=35, p=0.2)
            ]
        )

    train_dataset = MaskDataset(
        images_path=config['dataset']['train_images_path'],
        masks_path=config['dataset']['train_masks_path'],
        transform=train_transforms,
        shape=tuple(config['dataset']['shape']),
        augmentations=train_augmentations,
        # mean=[187.37888289, 60.40710529, 30.97342565],
        # std=[34.97379752, 20.30510642, 11.72539536]
    )

    validation_dataset = MaskDataset(
        images_path=config['dataset']['test_images_path'],
        masks_path=config['dataset']['test_masks_path'],
        transform=val_transforms,
        shape=tuple(config['dataset']['shape']),
        # mean=[187.37888289, 60.40710529, 30.97342565],
        # std=[34.97379752, 20.30510642, 11.72539536]
    )

    print('Train dataset size: {}'.format(len(train_dataset)))
    print('Validation dataset size: {}'.format(len(validation_dataset)))

    train_data = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=n_jobs,
        shuffle=True,
        drop_last=True,
        sampler=None
    )

    validation_data = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=n_jobs,
        drop_last=False,
        shuffle=False
    )

    lr_scheduler = None
    # lr_scheduler = create_lr_scheduler(optimizer, len(train_data), epochs, warmup=True)
    if lr_scheduler is not None and ckp is not None:
        if 'scheduler' in ckp.keys():
            lr_scheduler.load_state_dict(ckp['scheduler'])

    if use_apex:
        opt_level = 'O1'
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

        if ckp is not None:
            try:
                amp.load_state_dict(ckp['amp'])
            except Exception as e:
                print('Can\'t load APEX parameters, because: {}'.format(e))

    os.makedirs(config['train']['save']['model'], exist_ok=True)
    # loss_functions = []

    # loss_function = torch.nn.CrossEntropyLoss(weight=dataset_weights)
    loss_function = CE_DiceLoss()
    # loss_function = SoftIoULoss(n_classes=2)
    # loss_function = smp.losses.tversky.TverskyLoss('multiclass')

    # loss_functions.append(smp.losses.tversky.TverskyLoss('multiclass'))
    # loss_functions.append(CE_DiceLoss())
    # loss_functions.append(SoftIoULoss(n_classes=2))
    # loss_functions.append(FocalLoss())
    # count_of_losses = len(loss_functions)
    # loss_function = lambda _pred, _gt: sum([lf(_pred, _gt) / count_of_losses for lf in loss_functions])

    if os.path.exists(prev_f1_path):
        with open(prev_f1_path, 'r') as f:
            best_f1_score = float(f.read())
        print('Best f1: {}'.format(best_f1_score))
    else:
        best_f1_score = 0

    batches_count = len(train_data)

    for epoch in range(start_epoch, epochs + 1):
        model.train()

        avg_train_loss = 0
        with tqdm(total=len(train_data)) as pbar:
            for idx, train_batch_sample in enumerate(train_data):
                _img, _y_true = train_batch_sample

                img = _img.to(device)
                y_true = _y_true.to(device)

                y_pred = model(img)

                loss = loss_function(y_pred, y_true)
                loss = loss / gradient_accumulation_steps
                loss.backward()

                if (gradient_accumulation_steps == 1) or ((idx + 1) % gradient_accumulation_steps == 0) or (idx + 1 == batches_count):
                    if use_apex:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            optimizer.step()
                    else:
                        optimizer.step()

                    optimizer.zero_grad()

                    if lr_scheduler is not None:
                        lr_scheduler.step()

                avg_train_loss += loss.item()

                pbar.postfix = \
                    'Epoch: {}/{}, loss: {:.5f}, lr: {:.7f}'.format(
                        epoch,
                        epochs,
                        loss.item() / batch_size,
                        get_lr(optimizer)
                    )

                image_visualizer.per_batch(
                    {
                        'img': img,
                        'mask_true': y_true,
                        'mask_pred': y_pred
                    }
                )

                pbar.update(1)

        avg_train_loss = avg_train_loss / len(train_data)

        model.eval()
        avg_val_loss = 0
        avg_val_f1 = 0

        y_pred_stack = []
        y_gt_stack = []

        for idx, batch_sample in enumerate(tqdm(validation_data)):
            with torch.no_grad():
                _img, _y_true = train_batch_sample

                img = _img.to(device)
                y_true = _y_true.to(device)

                y_pred = model(img)

                loss = loss_function(
                    y_pred,
                    y_true
                )

                y_pred_stack.append(torch.concat([_t for _t in y_pred.to('cpu')], dim=1))
                y_gt_stack.append(torch.concat([_t for _t in y_true.to('cpu')], dim=0))

                # avg_val_f1 += binary_f1_score(y_pred, y_true).item()
                # avg_val_f1 += smp.metrics.get_stats(y_pred, y_true, 'multilabel', num_classes=2)
                avg_val_loss += loss.item()

        avg_loss = avg_val_loss / len(validation_data)
        # avg_val_f1 = avg_val_f1 / len(validation_data)
        avg_val_f1 = binary_f1_score(
            torch.concat(y_pred_stack, dim=1).unsqueeze(0),
            torch.concat(y_gt_stack, dim=0).unsqueeze(0)
        ).item()

        plot_visualizer.per_epoch(
            {
                'n': epoch,
                'val loss': avg_val_loss,
                'loss': avg_train_loss,
                'val acc': avg_val_f1
            }
        )

        # Save latest

        save_checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }

        if lr_scheduler is not None:
            save_checkpoint['scheduler'] = lr_scheduler.state_dict(),

        if use_apex:
            save_checkpoint['amp'] = amp.state_dict()

        traced = torch.jit.trace(
            model,
            torch.rand(1, 3, *config['dataset']['shape'][::-1],
                       requires_grad=True).to(device)
        )

        traced.save(
            os.path.join(
                config['train']['save']['model'],
                'last_state_trace.pt'
            )
        )

        torch.save(
            save_checkpoint,
            latest_path
        )

        # Save best
        if avg_val_f1 - best_f1_score > -1E-5:
            best_f1_score = avg_val_f1
            with open(prev_f1_path, 'w') as f:
                f.write('{}'.format(best_f1_score))

            torch.save(
                save_checkpoint,
                os.path.join(
                    config['train']['save']['model'],
                    'best_state.pt'
                )
            )

            traced.save(
                os.path.join(
                    config['train']['save']['model'],
                    'best_state_trace.pt'
                )
            )

            torch.save(
                save_checkpoint,
                os.path.join(
                    config['train']['save']['model'],
                    'best_state.pt'
                )
            )

