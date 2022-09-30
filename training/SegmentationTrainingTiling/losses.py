import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def l2(y_pred, y_true):
    return torch.sqrt(((y_pred - y_true) ** 2).sum())


def acc(y_pred, y_true):
    return (y_pred.argmax(dim=1) == y_true.argmax(dim=1)).sum().type(
        torch.FloatTensor
    ) / y_true.size(0)


def iou_acc(y_pred, y_true, threshold=0.5):
    y_pred_byte = torch.sigmoid(y_pred)[:, :-1] > threshold
    y_true_byte = y_true[:, :-1] > threshold

    union = torch.add(y_pred_byte, y_true_byte).sum()
    interception = torch.mul(y_pred_byte, y_true_byte).sum()

    if union == 0:
        if interception == 0:
            return 1
        else:
            return 0

    return interception / union


class GeneralizedSoftDiceLoss(torch.nn.Module):

    def __init__(self,
                 p=1,
                 smooth=1,
                 reduction='mean',
                 weight: torch.FloatTensor = None,
                 ignore_lb=80):
        super(GeneralizedSoftDiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth
        self.reduction = reduction
        self.weight = None if weight is None else weight
        self.ignore_lb = ignore_lb

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        logits = logits.float()
        ignore = label.data.cpu() == self.ignore_lb
        label = label.clone()
        label[ignore] = 0
        lb_one_hot = torch.zeros_like(logits).scatter_(1, label.unsqueeze(1), 1)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        lb_one_hot[[a, torch.arange(lb_one_hot.size(1)).long(), *b]] = 0
        lb_one_hot = lb_one_hot.detach()

        # compute loss
        probs = torch.sigmoid(logits)
        numer = torch.sum((probs*lb_one_hot), dim=(2, 3))
        denom = torch.sum(probs.pow(self.p)+lb_one_hot.pow(self.p), dim=(2, 3))
        if not self.weight is None:
            numer = numer * self.weight.view(1, -1)
            denom = denom * self.weight.view(1, -1)
        numer = torch.sum(numer, dim=1)
        denom = torch.sum(denom, dim=1)
        loss = 1 - (2*numer+self.smooth)/(denom+self.smooth)

        if self.reduction == 'mean':
            loss = loss.mean()
        return loss


class BatchSoftDiceLoss(torch.nn.Module):

    def __init__(self,
                 p=1,
                 smooth=1,
                 weight=None,
                 ignore_lb=255):
        super(BatchSoftDiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth
        self.weight = None if weight is None else torch.tensor(weight)
        self.ignore_lb = ignore_lb

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        logits = logits.float()
        ignore = label.data.cpu() == self.ignore_lb
        label = label.clone()
        label[ignore] = 0
        lb_one_hot = torch.zeros_like(logits).scatter_(1, label.unsqueeze(1), 1)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        lb_one_hot[[a, torch.arange(lb_one_hot.size(1)).long(), *b]] = 0
        lb_one_hot = lb_one_hot.detach()

        # compute loss
        probs = torch.sigmoid(logits)
        numer = torch.sum((probs*lb_one_hot), dim=(2, 3))
        denom = torch.sum(probs.pow(self.p)+lb_one_hot.pow(self.p), dim=(2, 3))
        if not self.weight is None:
            numer = numer * self.weight.view(1, -1)
            denom = denom * self.weight.view(1, -1)
        numer = torch.sum(numer)
        denom = torch.sum(denom)
        loss = 1 - (2*numer+self.smooth)/(denom+self.smooth)
        return loss


class SoftIoULoss(torch.nn.Module):
    def __init__(self, n_classes):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes

    @staticmethod
    def to_one_hot(tensor, n_classes):
        n, h, w = tensor.size()
        one_hot = torch.zeros(
            n, n_classes, h, w, device=tensor.device
        ).scatter_(1, tensor.view(n, 1, h, w), 1)
        return one_hot

    def forward(self, logit, target):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(logit)

        pred = F.softmax(logit, dim=1)
        target_onehot = self.to_one_hot(target, self.n_classes)

        # Numerator Product
        inter = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot - (pred * target_onehot)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        loss = inter / (union + 1e-16)

        # Return average loss over classes and batch
        return 1 - loss.mean()


def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2],
                                labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target


def get_weights(target):
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts
    # cls_w = class_weight.compute_class_weight('balanced', classes, t_np)

    weights = np.ones(7)
    weights[classes] = cls_w
    return torch.from_numpy(weights).float().cuda()


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index,
                                      reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255,
                 size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False,
                                           ignore_index=ignore_index,
                                           weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()


class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255,
                 weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight,
                                                 reduction=reduction,
                                                 ignore_index=ignore_index)

    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss


def binary_f1_score(output: torch.Tensor, target: torch.Tensor):
    """
    Calculate F1 score
    Args:
        output: model prediction    (BxCxHxW)
        target: ground-truth        (BxHxW)

    Returns:
        Average score value for the batch
    """
    output_classes = 1 - output.argmax(dim=1)
    target_classes = 1 - target

    # print(output_classes.shape)
    # print(target_classes.shape)

    intersection = torch.sum(output_classes * target_classes)
    accuracy = intersection / (output_classes.sum() + 1E-5)
    recall = intersection / (target_classes.sum() + 1E-5)

    f1_res = 2 * accuracy * recall / (accuracy + recall + 1E-5)

    return f1_res / output.size(0)


if __name__ == '__main__':
    t1 = torch.LongTensor([
        [[0, 1, 1], [0, 0, 0]]
    ])

    t2 = torch.LongTensor([
        [[0, 0, 0], [1, 1, 1]]
    ])

    print(t1.shape)

    print(binary_f1_score(t1, t2))