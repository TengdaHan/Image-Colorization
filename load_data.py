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
from skimage import color
from transform import ReLabel, ToLabel, ToSP, Scale

from sklearn.neighbors import NearestNeighbors


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
        classify=False,
        loader=pil_loader):

        tic = time.time()
        self.root = root
        self.loader = loader
        self.image_transform = transform
        self.imgpath = glob.glob(root + 'lfw_funneled/*/*')
        self.classify = classify

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

        if classify:
            ab_list = np.load('data/pts_in_hull.npy')
            self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ab_list)

        print('Load %d images, used %fs' % (self.path.__len__(), time.time()-tic))

    def __getitem__(self, index):
        mypath = self.path[index]
        img = self.loader(mypath) # PIL Image
        img_lab = color.rgb2lab(np.array(img)) # np array
        # print('start')
        # print(np.amax(img_lab[:,:,0]), np.amin(img_lab[:,:,0]))
        # print(np.amax(img_lab[:,:,1]), np.amin(img_lab[:,:,1]))
        # print(np.amax(img_lab[:,:,2]), np.amin(img_lab[:,:,2]))

        img_lab = img_lab[13:13+224, 13:13+224, :]

        if self.classify:
            X_a = np.ravel(img_lab[:,:,1])
            X_b = np.ravel(img_lab[:,:,2])
            img_ab = np.vstack((X_a, X_b)).T
            _, ind = self.nbrs.kneighbors(img_ab)
            ab_class = np.reshape(ind, (224,224))
            # print(ab_class.shape, ab_class.dtype, np.amax(ab_class), np.amin(ab_class))
            ab_class = torch.unsqueeze(torch.LongTensor(ab_class), 0)

        img_lab = torch.FloatTensor(np.transpose(img_lab, (2,0,1)))
        # print('start')
        # print(torch.max(img_lab[0,:,:]), torch.min(img_lab[0,:,:]))
        # print(torch.max(img_lab[1,:,:]), torch.min(img_lab[1,:,:]))
        # print(torch.max(img_lab[2,:,:]), torch.min(img_lab[2,:,:]))

        img_l = torch.unsqueeze(img_lab[0],0) / 100. # L channel 0-100
        img_ab = (img_lab[1::] + 0) / 110. # ab channel -110 - 110

        if self.classify:
            return img_l, ab_class
        return img_l, img_ab


    def __len__(self):
        return len(self.path)


if __name__ == '__main__':
    data_root = '/home/htd/Documents/DATA/LFW/'
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    image_transform = transforms.Compose([
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                          ])

    lfw = lfw_Dataset(data_root, mode='test',
                      transform=image_transform, classify=True)

    data_loader = data.DataLoader(lfw,
                                  batch_size=4,
                                  shuffle=False,
                                  num_workers=4)

    for i, (data, target) in enumerate(data_loader):
        import pdb; pdb.set_trace()
        # plt.imshow(torch.squeeze(imgs[0]).transpose(0,1).transpose(1,2).numpy())
        sys.exit()
