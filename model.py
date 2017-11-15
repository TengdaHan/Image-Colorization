import os
import os.path as osp
import sys
import numpy as np
import matplotlib.pyplot as plt
import collections
import time

import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.models import vgg16


# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class FCN32(nn.Module):
    ''''''
    def __init__(self, n_class=2):
        super(FCN32, self).__init__()

        self.vgg = vgg16(pretrained=True)
        self.features = self.vgg.features

        # conv1 to conv5
        self.first = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.rest = nn.Sequential(self.features[1],
                                  self.features[2],
                                  self.features[3],
                                  self.features[4],
                                  self.features[5],
                                  self.features[6],
                                  self.features[7],
                                  self.features[8],
                                  self.features[9],
                                  self.features[10],
                                  self.features[11],
                                  self.features[12],
                                  self.features[13],
                                  self.features[14],
                                  self.features[15],
                                  self.features[16],
                                  self.features[17],
                                  self.features[18],
                                  self.features[19],
                                  self.features[20],
                                  self.features[21],
                                  self.features[22],
                                  self.features[23],
                                  self.features[24],
                                  self.features[25],
                                  self.features[26],
                                  self.features[27],
                                  self.features[28],
                                  self.features[29],
                                  self.features[30])

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 3, stride=1, padding=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 3, stride=1, padding=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.upscore = nn.ConvTranspose2d(n_class, n_class, 64, stride=32,
                                          bias=False)

        self._initialize_weights()


    def forward(self, x):
        # x is the input image, one channel
        first = self.first(x)
        h = self.rest(first)

        # fc6
        h = self.fc6(h)
        h = self.relu6(h)
        h = self.drop6(h)

        # fc7
        h = self.fc7(h)
        h = self.relu7(h)
        h = self.drop7(h)

        # upsampling
        h = self.score_fr(h)
        h = self.upscore(h)
        h = h[:, :, 16:16 + x.size()[2], 16:16 + x.size()[3]].contiguous()

        return h

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if name == 'first':
                m.weight.data = torch.mean(self.features[0].weight.data, 1, keepdim=True)
                m.bias.data = self.features[0].bias.data
            if name == 'fc6' or name == 'fc7' or name == 'score_fr':
                m.weight.data.zero_()
            if name == 'upscore':
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)


class Simple(nn.Module):
    ''''''
    def __init__(self, n_class=2):
        super(Simple, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv4 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv6 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.relu7 = nn.ReLU(inplace=True)

        self.score_fr = nn.Conv2d(512, n_class, 1)
        self.upscore = nn.ConvTranspose2d(n_class, n_class, 64, stride=32,
                                          bias=False)

        self._initialize_weights()

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.relu1(h)
        h = self.pool1(h)

        h = self.conv2(h)
        h = self.relu2(h)
        h = self.pool2(h)

        h = self.conv3(h)
        h = self.relu3(h)
        h = self.pool3(h)

        h = self.conv4(h)
        h = self.relu4(h)
        h = self.pool4(h)

        h = self.conv5(h)
        h = self.relu5(h)
        h = self.pool5(h)

        h = self.conv6(h)
        h = self.relu6(h)

        h = self.conv7(h)
        h = self.relu7(h)

        h = self.score_fr(h)
        h = self.upscore(h)
        h = F.tanh(h)
        h = h[:, :, 16:16 + x.size()[2], 16:16 + x.size()[3]].contiguous()

        return h

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.1)

class Simple_Classify(nn.Module):
    ''''''
    def __init__(self, n_class=313):
        super(Simple_Classify, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        # self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        # self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        # self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # self.conv4 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        # self.relu4 = nn.ReLU(inplace=True)
        # # self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        #
        # self.conv5 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        # self.relu5 = nn.ReLU(inplace=True)
        # # self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        #
        # self.conv6 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        # self.relu6 = nn.ReLU(inplace=True)
        #
        # self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        # self.relu7 = nn.ReLU(inplace=True)

        self.score_fr = nn.Conv2d(64, n_class, 1)
        # self.upscore = nn.ConvTranspose2d(n_class, n_class, 64, stride=32,
        #                                   bias=False)

        self._initialize_weights()

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.relu1(h)
        # h = self.pool1(h)

        h = self.conv2(h)
        h = self.relu2(h)
        # h = self.pool2(h)

        h = self.conv3(h)
        h = self.relu3(h)
        # h = self.pool3(h)

        # h = self.conv4(h)
        # h = self.relu4(h)
        # h = self.pool4(h)

        # h = self.conv5(h)
        # h = self.relu5(h)
        # h = self.pool5(h)

        # h = self.conv6(h)
        # h = self.relu6(h)

        # h = self.conv7(h)
        # h = self.relu7(h)

        h = self.score_fr(h)
        # h = self.upscore(h)
        # h = h[:, :, 16:16 + x.size()[2], 16:16 + x.size()[3]].contiguous()

        return h

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.1)

if __name__ == '__main__':
    model = FCN32()
