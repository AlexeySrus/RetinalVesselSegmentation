import sys
sys.path.append('../../')
import time
from unittest import result
import torch
import torch.nn as nn
import torch.nn.functional as F
from NetworkTrainer.networks.resnet import resnet18, resnet34, resnet101, resnet50, resnet152

class dilated_conv(nn.Module):
    """ same as original conv if dilation equals to 1 """
    def __init__(self, in_channel, out_channel, kernel_size=3, dropout_rate=0.0, activation=F.relu, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=dilation, dilation=dilation)
        self.norm = nn.BatchNorm2d(out_channel)
        self.activation = activation
        if dropout_rate > 0:
            self.drop = nn.Dropout2d(p=dropout_rate)
        else:
            self.drop = lambda x: x  # no-op

    def forward(self, x):
        # CAB: conv -> activation -> batch normal
        x = self.norm(self.activation(self.conv(x)))
        x = self.drop(x)
        return x


class ConvDownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate=0.0, dilation=1):
        super().__init__()
        self.conv1 = dilated_conv(in_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)
        self.conv2 = dilated_conv(out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.pool(x), x


class ConvUpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate=0.0, dilation=1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, 2, stride=2)
        self.conv1 = dilated_conv(in_channel // 2 + out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)
        self.conv2 = dilated_conv(out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)

    def forward(self, x, x_skip):
        x = self.up(x)
        H_diff = x.shape[2] - x_skip.shape[2]
        W_diff = x.shape[3] - x_skip.shape[3]
        x_skip = F.pad(x_skip, (0, W_diff, 0, H_diff), mode='reflect')
        x = torch.cat([x, x_skip], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# Transfer Learning ResNet as Encoder part of UNet
class ResUNet(nn.Module):
    def __init__(self, net='res50', seg_classes = 2, colour_classes = 3, fixed_feature=False, pretrained=False):
        super().__init__()
        # load weight of pre-trained resnet
        self.l = [64, 64, 128, 256, 512]
        if 'res101' in net:
            self.resnet = resnet101(pretrained=pretrained, arch=net)
            self.l = [64, 256, 512, 1024, 2048]
        elif 'res50' in net:
            self.resnet = resnet50(pretrained=pretrained, arch=net)
            self.l = [64, 256, 512, 1024, 2048]
        elif 'res18' in net:
            self.resnet = resnet18(pretrained=pretrained)
            self.l = [64, 64, 128, 256, 512]
        elif 'res34' in net:
            self.resnet = resnet34(pretrained=pretrained)
            self.l = [64, 64, 128, 256, 512]
        elif 'res152' in net:
            self.resnet = resnet152(pretrained=pretrained)
            self.l = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError('Unknown network architecture: {}'.format(net))
        # self.resnet1 = Resnet34(pretrained=False)
        if fixed_feature:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # up conv
        self.u5 = ConvUpBlock(self.l[4], self.l[3], dropout_rate=0.1)
        self.u6 = ConvUpBlock(self.l[3], self.l[2], dropout_rate=0.1)
        self.u7 = ConvUpBlock(self.l[2], self.l[1], dropout_rate=0.1)
        self.u8 = ConvUpBlock(self.l[1], self.l[0], dropout_rate=0.1)
        # final conv
        self.seg = nn.ConvTranspose2d(self.l[0], seg_classes, 2, stride=2)

    def forward(self, x):
        # refer https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = s1 = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        # print(x.shape)
        x = s2 = self.resnet.layer1(x)
        # print(x.shape)
        x = s3 = self.resnet.layer2(x)
        # print(x.shape)
        x = s4 = self.resnet.layer3(x)
        # print(x.shape)
        x = self.resnet.layer4(x)
        # print(x.shape)
        x1 = self.u5(x, s4)
        x1 = self.u6(x1, s3)
        x1 = self.u7(x1, s2)
        x1 = self.u8(x1, s1)
        out = self.seg(x1)
        return out


class ResUNet_ds(ResUNet):
    def __init__(self, net='res50', seg_classes = 2):
        super().__init__()
        # load weight of pre-trained resnet
        self.seg1 = nn.Conv2d(self.l[0], seg_classes, 1, stride=1)
        self.seg2 = nn.Conv2d(self.l[1], seg_classes, 1, stride=1)
        self.seg3 = nn.Conv2d(self.l[2], seg_classes, 1, stride=1)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = s1 = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = s2 = self.resnet.layer1(x)
        x = s3 = self.resnet.layer2(x)
        x = s4 = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x4 = self.u5(x, s4)
        x3 = self.u6(x4, s3)
        x2 = self.u7(x3, s2)
        x1 = self.u8(x2, s1)
        out = self.seg(x1)
        out1 = self.seg1(x1)
        out2 = self.seg2(x2)
        out3 = self.seg3(x3)
        return [out, out1, out2, out3]


if __name__=='__main__':
    x = torch.randn((2, 3, 256, 256))
    net = ResUNet()
    pred = net(x)
    print(pred.shape)