import os
import os.path as osp
import sys
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import time

import torch
import torchvision
from torchvision import transforms
from torch.utils import data
import scipy.io as io
import scipy.misc as misc
import glob
import csv
from skimage import color
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
        types='',
        show_ab=False,
        loader=pil_loader):

        tic = time.time()
        self.root = root
        self.loader = loader
        self.image_transform = transform
        self.imgpath = glob.glob(root + 'lfw_funneled/*/*')
        self.types = types
        self.show_ab = show_ab # show ab channel in classify mode

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

        np.random.seed(0)
        if shuffle:
            perm = np.random.permutation(len(self.path))
            self.path = [self.path[i] for i in perm]

        if types == 'classify':
            ab_list = np.load('data/pts_in_hull.npy')
            self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ab_list)

        print('Load %d images, used %fs' % (self.path.__len__(), time.time()-tic))

    def __getitem__(self, index):
        mypath = self.path[index]
        img = self.loader(mypath) # PIL Image
        img = np.array(img)[13:13+224, 13:13+224, :]

        img_lab = color.rgb2lab(np.array(img)) # np array
        # img_lab = img_lab[13:13+224, 13:13+224, :]

        if self.types == 'classify':
            X_a = np.ravel(img_lab[:,:,1])
            X_b = np.ravel(img_lab[:,:,2])
            img_ab = np.vstack((X_a, X_b)).T
            _, ind = self.nbrs.kneighbors(img_ab)
            ab_class = np.reshape(ind, (224,224))
            # print(ab_class.shape, ab_class.dtype, np.amax(ab_class), np.amin(ab_class))
            ab_class = torch.unsqueeze(torch.LongTensor(ab_class), 0)

        img = (img - 127.5) / 127.5 # -1 to 1
        img = torch.FloatTensor(np.transpose(img, (2,0,1)))
        img_lab = torch.FloatTensor(np.transpose(img_lab, (2,0,1)))

        img_l = torch.unsqueeze(img_lab[0],0) / 100. # L channel 0-100
        img_ab = (img_lab[1::] + 0) / 110. # ab channel -110 - 110

        if self.types == 'classify':
            if self.show_ab:
                return img_l, ab_class, img_ab
            return img_l, ab_class
        elif self.types == 'raw':
            return img_l, img
        else:
            return img_l, img_ab

    def __len__(self):
        return len(self.path)

class Flower_Dataset(data.Dataset):
    def __init__(self, root,
        shuffle=False,
        small=False,
        mode='test',
        transform=None,
        target_transform=None,
        types='',
        show_ab=False,
        loader=pil_loader):

        tic = time.time()
        self.root = root
        self.loader = loader
        self.image_transform = transform
        self.imgpath = glob.glob(root + 'jpg/*.jpg')
        self.types = types
        self.show_ab = show_ab # show ab channel in classify mode

        # read split
        split_file = io.loadmat(root + 'datasplits.mat')

        self.train_file = set([str(i).zfill(4) for i in np.hstack((split_file['trn1'][0], split_file['val1'][0]))])
        self.test_file = set([str(i).zfill(4) for i in split_file['tst1'][0]])
        assert self.train_file.__len__() == 1020
        assert self.test_file.__len__() == 340

        self.path = []
        if mode == 'train':
            for item in self.imgpath:
                if item.split('/')[-1][6:6+4] in self.train_file:
                    self.path.append(item)
        elif mode == 'test':
            for item in self.imgpath:
                if item.split('/')[-1][6:6+4] in self.test_file:
                    self.path.append(item)

        self.path = sorted(self.path)

        np.random.seed(0)
        if shuffle:
            perm = np.random.permutation(len(self.path))
            self.path = [self.path[i] for i in perm]

        if types == 'classify':
            ab_list = np.load('data/pts_in_hull.npy')
            self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ab_list)

        print('Load %d images, used %fs' % (self.path.__len__(), time.time()-tic))

    def __getitem__(self, index):
        mypath = self.path[index]
        img = self.loader(mypath) # PIL Image
        img = np.array(img)
        img = misc.imresize(img, (224, 224))

        img_lab = color.rgb2lab(np.array(img)) # np array
        # img_lab = img_lab[13:13+224, 13:13+224, :]

        if self.types == 'classify':
            X_a = np.ravel(img_lab[:,:,1])
            X_b = np.ravel(img_lab[:,:,2])
            img_ab = np.vstack((X_a, X_b)).T
            _, ind = self.nbrs.kneighbors(img_ab)
            ab_class = np.reshape(ind, (224,224))
            ab_class = torch.unsqueeze(torch.LongTensor(ab_class), 0)

        img = (img - 127.5) / 127.5 # -1 to 1
        img = torch.FloatTensor(np.transpose(img, (2,0,1)))
        img_lab = torch.FloatTensor(np.transpose(img_lab, (2,0,1)))

        img_l = torch.unsqueeze(img_lab[0],0) / 100. # L channel 0-100
        img_ab = (img_lab[1::] + 0) / 110. # ab channel -110 - 110

        if self.types == 'classify':
            if self.show_ab:
                return img_l, ab_class, img_ab
            return img_l, ab_class
        elif self.types == 'raw':
            return img_l, img
            # if self.show_ab:
            #     return img_l, img_ab, None
        else:
            return img_l, img_ab

    def __len__(self):
        return len(self.path)

class Spongebob_Dataset(data.Dataset):
    def __init__(self, root,
        shuffle=False,
        small=False,
        mode='test',
        transform=None,
        target_transform=None,
        types='',
        show_ab=False,
        large=False,
        loader=pil_loader):

        tic = time.time()
        self.root = root
        self.loader = loader
        self.image_transform = transform
        if large:
            self.size = 480
            self.imgpath = glob.glob(root + 'img_480/*.png')
        else:
            self.size = 224
            self.imgpath = glob.glob(root + 'img/*.png')
        self.types = types
        self.show_ab = show_ab # show ab channel in classify mode

        # read split
        self.train_file = set()
        with open(self.root + 'train_split.csv', 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                self.train_file.add(str(row[0]).zfill(4))

        assert self.train_file.__len__() == 1392

        self.test_file = set()
        with open(self.root + 'test_split.csv', 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                self.test_file.add(str(row[0]).zfill(4))
        assert self.test_file.__len__() == 348

        self.path = []
        if mode == 'train':
            for item in self.imgpath:
                if item.split('/')[-1][6:6+4] in self.train_file:
                    self.path.append(item)
        elif mode == 'test':
            for item in self.imgpath:
                if item.split('/')[-1][6:6+4] in self.test_file:
                    self.path.append(item)

        self.path = sorted(self.path)

        np.random.seed(0)
        if shuffle:
            perm = np.random.permutation(len(self.path))
            self.path = [self.path[i] for i in perm]

        if types == 'classify':
            ab_list = np.load('data/pts_in_hull.npy')
            self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ab_list)

        print('Load %d images, used %fs' % (self.path.__len__(), time.time()-tic))

    def __getitem__(self, index):
        mypath = self.path[index]
        img = self.loader(mypath) # PIL Image
        img = np.array(img)
        if (img.shape[0] != self.size) or (img.shape[1] != self.size):
            img = misc.imresize(img, (self.size, self.size))

        img_lab = color.rgb2lab(np.array(img)) # np array
        # img_lab = img_lab[13:13+224, 13:13+224, :]

        if self.types == 'classify':
            X_a = np.ravel(img_lab[:,:,1])
            X_b = np.ravel(img_lab[:,:,2])
            img_ab = np.vstack((X_a, X_b)).T
            _, ind = self.nbrs.kneighbors(img_ab)
            ab_class = np.reshape(ind, (self.size,self.size))
            # print(ab_class.shape, ab_class.dtype, np.amax(ab_class), np.amin(ab_class))
            ab_class = torch.unsqueeze(torch.LongTensor(ab_class), 0)

        img = (img - 127.5) / 127.5 # -1 to 1
        img = torch.FloatTensor(np.transpose(img, (2,0,1)))
        img_lab = torch.FloatTensor(np.transpose(img_lab, (2,0,1)))

        img_l = torch.unsqueeze(img_lab[0],0) / 100. # L channel 0-100
        img_ab = (img_lab[1::] + 0) / 110. # ab channel -110 - 110

        if self.types == 'classify':
            if self.show_ab:
                return img_l, ab_class, img_ab
            return img_l, ab_class
        elif self.types == 'raw':
            return img_l, img
            # if self.show_ab:
            #     return img_l, img_ab, None
        else:
            return img_l, img_ab

    def __len__(self):
        return len(self.path)

class SC2_Dataset(data.Dataset):
    def __init__(self, root,
        shuffle=False,
        small=False,
        mode='test',
        transform=None,
        target_transform=None,
        types='',
        show_ab=False,
        large=False,
        loader=pil_loader):

        tic = time.time()
        self.root = root
        self.loader = loader
        self.image_transform = transform
        if large:
            self.size = 480
            self.imgpath = glob.glob(root + 'img_480/*.png')
        else:
            self.size = 224
            self.imgpath = glob.glob(root + 'img/*.png')
        self.types = types
        self.show_ab = show_ab # show ab channel in classify mode

        # read split
        self.train_file = set()
        with open(self.root + 'train_split.csv', 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                self.train_file.add(str(row[0]).zfill(4))
        assert self.train_file.__len__() == 1383

        self.test_file = set()
        with open(self.root + 'test_split.csv', 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                self.test_file.add(str(row[0]).zfill(4))
        assert self.test_file.__len__() == 345

        self.path = []
        if mode == 'train':
            for item in self.imgpath:
                if item.split('/')[-1][6:6+4] in self.train_file:
                    self.path.append(item)
        elif mode == 'test':
            for item in self.imgpath:
                if item.split('/')[-1][6:6+4] in self.test_file:
                    self.path.append(item)

        self.path = sorted(self.path)

        np.random.seed(0)
        if shuffle:
            perm = np.random.permutation(len(self.path))
            self.path = [self.path[i] for i in perm]

        if types == 'classify':
            ab_list = np.load('data/pts_in_hull.npy')
            self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ab_list)

        print('Load %d images, used %fs' % (self.path.__len__(), time.time()-tic))

    def __getitem__(self, index):
        mypath = self.path[index]
        img = self.loader(mypath) # PIL Image
        img = np.array(img)
        if (img.shape[0] != self.size) or (img.shape[1] != self.size):
            img = misc.imresize(img, (self.size, self.size))

        img_lab = color.rgb2lab(np.array(img)) # np array
        # img_lab = img_lab[13:13+224, 13:13+224, :]

        if self.types == 'classify':
            X_a = np.ravel(img_lab[:,:,1])
            X_b = np.ravel(img_lab[:,:,2])
            img_ab = np.vstack((X_a, X_b)).T
            _, ind = self.nbrs.kneighbors(img_ab)
            ab_class = np.reshape(ind, (self.size,self.size))
            # print(ab_class.shape, ab_class.dtype, np.amax(ab_class), np.amin(ab_class))
            ab_class = torch.unsqueeze(torch.LongTensor(ab_class), 0)

        img = (img - 127.5) / 127.5 # -1 to 1
        img = torch.FloatTensor(np.transpose(img, (2,0,1)))
        img_lab = torch.FloatTensor(np.transpose(img_lab, (2,0,1)))

        img_l = torch.unsqueeze(img_lab[0],0) / 100. # L channel 0-100
        img_ab = (img_lab[1::] + 0) / 110. # ab channel -110 - 110

        if self.types == 'classify':
            if self.show_ab:
                return img_l, ab_class, img_ab
            return img_l, ab_class
        elif self.types == 'raw':
            if img.size(1) == 479 or img.size(2) == 479:
                print(mypath)
            return img_l, img
            # if self.show_ab:
            #     return img_l, img_ab, None
        else:
            return img_l, img_ab

    def __len__(self):
        return len(self.path)


if __name__ == '__main__':
    data_root = '/home/users/u5612799/DATA/SCReplay/'
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    image_transform = transforms.Compose([
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                          ])

    lfw = SC2_Dataset(data_root, mode='train',
                      transform=image_transform, large=True, types='raw')

    data_loader = data.DataLoader(lfw,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=4)

    for i, (data, target) in enumerate(data_loader):
        print(i, len(lfw))
