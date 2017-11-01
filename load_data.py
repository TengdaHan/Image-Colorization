import os
import os.path as osp
import sys
import numpy as np
import pickle
from PIL import Image, ImageCms
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import collections
import time

import torch
import torchvision
from torchvision import transforms
from torch.utils import data
# from transform import HorizontalFlip, VerticalFlip
import scipy.io as io
import glob
import csv
from transform import ReLabel, ToLabel, ToSP, Scale


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class lfw_Dataset(data.Dataset):
    def __init__(self, root,
        shuffle=False,
        small=False,
        mode='test',
        transform=None,
        target_transform=None,
        loader=pil_loader):

        self.root = root
        self.loader = loader
        self.image_transform = transform
        self.imgpath = glob.glob(root + 'lfw_funneled/*/*')

        # read split
        self.train_people = set()
        with open(self.root + 'peopleDevTrain.txt', 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                self.train_people.add(row[0])
        assert self.train_people.__len__() == 4038

        self.test_people = set()
        with open(self.root + 'peopleDevTest.txt', 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                self.test_people.add(row[0])
        assert self.test_people.__len__() == 1711

        self.path = []
        if mode == 'train':
            for item in self.imgpath:
                if item.split('/')[-2] in self.train_people:
                    self.path.append(item)
        elif mode == 'test':
            for item in self.imgpath:
                if item.split('/')[-2] in self.test_people:
                    self.path.append(item)

    def __getitem__(self, index):
        mypath = self.path[index]
        img = self.loader(mypath)

        img_lab = rgb2lab(img)
        if self.image_transform is not None:
            img = self.image_transform(img)
            img_lab = self.image_transform(img_lab)

        return img, img_lab


    def __len__(self):
        return len(self.path)

def rgb2lab(im):
    srgb_profile = ImageCms.createProfile("sRGB")
    lab_profile  = ImageCms.createProfile("LAB")
    rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")
    lab_im = ImageCms.applyTransform(im, rgb2lab_transform)
    return lab_im

def lab2rgb(im):
    srgb_profile = ImageCms.createProfile("sRGB")
    lab_profile  = ImageCms.createProfile("LAB")
    lab2rgb_transform = ImageCms.buildTransformFromOpenProfiles(lab_profile, srgb_profile, "LAB", "RGB")
    rgb_im = ImageCms.applyTransform(im, lab2rgb_transform)
    return rgb_im


if __name__ == '__main__':
    data_root = '/home/htd/Documents/DATA/LFW/'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    image_transform = transforms.Compose([
                              Scale((224, 224), Image.BILINEAR),
                              transforms.ToTensor(),
                            #   normalize,
                          ])

    lfw = lfw_Dataset(data_root, mode='test',
                      transform=image_transform)

    data_loader = data.DataLoader(lfw,
                                  batch_size=4,
                                  shuffle=False,
                                  num_workers=4)

    for i, (imgs, img_labs) in enumerate(data_loader):
        import pdb; pdb.set_trace()
        plt.imshow(torch.squeeze(imgs[0]).transpose(0,1).transpose(1,2).numpy())
        plt.show()
        sys.exit()
