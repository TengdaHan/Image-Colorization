from loss import CrossEntropy2d
from transform import ReLabel, ToLabel, ToSP, Scale
from model import *
from utils import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from skimage import color

import time
import os
import sys
from PIL import Image
import argparse
import numpy as np
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description='Colorization Main')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size: default 4')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='Learning rate for optimizer')
parser.add_argument('--weight_decay', default=2e-5, type=float,
                    help='Weight decay for optimizer')
parser.add_argument('--num_epoch', default=20, type=int,
                    help='Number of epochs')
parser.add_argument('--test', default='', type=str,
                    help='Path to the model')
parser.add_argument('--resume', default='', type=str,
                    help='Path to resume')

parser.add_argument('-c','--classify', action="store_true",
                    help='Classify Color? (Zhang et al. 2016)')
parser.add_argument('-p', '--plot', action="store_true",
                    help='Plot accuracy and loss?')
parser.add_argument('-s','--save', action="store_true",
                    help='Save model?')
parser.add_argument('--gpu', default=0, type=int,
                    help='Which GPU to use?')

def main():
    global args
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)


    # model
    # model = FCN32(n_class=2)
    if args.classify:
        model = Simple_Classify(n_class=313)
    else:
        model = Simple(n_class=2)

    if args.resume:
        print('Resume model: %s' % resume)
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        print('No Resume')
        start_epoch = 0

    model.cuda()

    # optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    if args.classify:
        global ab_list
        ab_list = np.load('data/pts_in_hull.npy')

    # loss function
    global criterion
    criterion = nn.MSELoss()

    # dataset
    data_root = '/home/users/u5612799/DATA/Spongebob/'
    from load_data import Spongebob_Dataset as myDataset

    image_transform = transforms.Compose([transforms.CenterCrop(224),
                                          transforms.ToTensor()])

    lfw_train = myDataset(data_root, mode='train',
                      transform=image_transform,
                      types='classify',
                      shuffle=True)

    train_loader = data.DataLoader(lfw_train,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=4)

    lfw_val = myDataset(data_root, mode='test',
                      transform=image_transform,
                      types='classify',
                      show_ab=True,
                      shuffle=True)

    val_loader = data.DataLoader(lfw_val,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=4)

    global val_bs
    val_bs = val_loader.batch_size

    # setup
    global iteration, print_interval, plotter, plotter_basic
    iteration = 0
    print_interval = 5
    plotter = Plotter_Single()
    plotter_basic = Plotter_Single()

    global img_path
    img_path = 'img/1116/ClassifyBob_bs%d_%s_lr%s_test/' \
               % (args.batch_size, 'Adam', str(args.lr))
    model_path = 'model/1116/ClassifyBob_bs%d_%s_lr%s_test/' \
               % (args.batch_size, 'Adam', str(args.lr))
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # start loop
    start_epoch = 0

    for epoch in range(start_epoch, args.num_epoch):
        print('Epoch {}/{}'.format(epoch, args.num_epoch - 1))
        print('-' * 20)
        if epoch == 0:
            val_loss = validate(val_loader, model, optimizer, epoch=-1)
        train_loss = train(train_loader, model, optimizer, epoch, iteration)
        val_loss = validate(val_loader, model, optimizer, epoch)

        plotter.train_update(train_loss)
        plotter.val_update(val_loss)
        plotter.draw(img_path + 'train_val.png')

        if args.save:
            print('Saving check point')
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             },
                             filename=model_path+'epoch%d.pth.tar' \
                             % epoch)



def train(train_loader, model, optimizer, epoch, iteration):
    losses_basic = AverageMeter()
    losses = AverageMeter()
    model.train()

    for i, (data, target) in enumerate(train_loader):
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = model(data)
        if args.classify:
            loss = CrossEntropy2d(output, target)
        else:
            loss = criterion(output.view(output.size(0), -1), target.view(target.size(0), -1))
        losses_basic.update(loss.data[0], target.size(0), history=1)
        losses.update(loss.data[0], target.size(0), history=1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % print_interval == 0:
            print('Epoch%d[%d/%d]: Loss: %0.4f(%0.4f)' \
                % (epoch, i, len(train_loader), losses.val, losses.avg))

            plotter_basic.train_update(losses_basic.avg)
            plotter_basic.draw(img_path + 'train_basic.png')

            losses_basic.reset()

        iteration += 1

    return losses.avg


def validate(val_loader, model, optimizer, epoch):
    losses = AverageMeter()
    model.eval()

    for i, mydata in enumerate(val_loader):
        if args.classify:
            (data, target, ab) = mydata
        else:
            (data, target) = mydata
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = model(data)
        if args.classify:
            loss = CrossEntropy2d(output, target)
        else:
            loss = criterion(output.view(output.size(0), -1), target.view(target.size(0), -1))
        losses.update(loss.data[0], target.size(0), history=1)

        if i == 0:
            if args.classify:
                vis_lab_target_classify(data.data, target.data, output.data, ab, epoch)
            else:
                vis_lab_target(data.data, target.data, output.data, epoch)

        if i % 50 == 0:
            print('Validating Epoch %d: [%d/%d]' \
                % (epoch, i, len(val_loader)))

    print('Validation Loss: %0.4f' % losses.avg)

    return losses.avg


def vis_lab_target(data, target, output, epoch):
    '''visualize images for regression mode'''
    img_list = []
    for i in range(min(32, val_bs)):
        l = torch.unsqueeze(torch.squeeze(data[i]), 0)
        ab_raw = target[i]
        ab_pred = output[i]

        raw = torch.cat((l*100., ab_raw*110.), 0).cpu().numpy()
        pred = torch.cat((l*100., ab_pred*110.), 0).cpu().numpy()
        raw = np.transpose(raw, (1,2,0))
        pred = np.transpose(pred, (1,2,0))
        raw_rgb = color.lab2rgb(np.float64(raw))
        pred_rgb = color.lab2rgb(np.float64(pred))

        grey = l.cpu().numpy()
        grey = np.transpose(grey, (1,2,0))
        grey = np.repeat(grey, 3, axis=2).astype(np.float64)
        img_list.append(np.concatenate((grey, raw_rgb, pred_rgb), 1))

    img_list = [np.concatenate(img_list[4*i:4*(i+1)], axis=1) for i in range(len(img_list) // 4)]
    img_list = np.concatenate(img_list, axis=0)

    plt.figure(figsize=(24,18))
    plt.imshow(img_list)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(img_path + 'epoch%d_val.png' % epoch)
    plt.clf()

def vis_lab_target_classify(data, target, output, ab, epoch):
    '''visualize images for regression mode'''
    img_list = []
    for i in range(min(32, val_bs)):
        l = torch.unsqueeze(torch.squeeze(data[i]), 0).cpu()
        ab_raw = ab[i]

        ab_pred = output[i]
        ab_pred = ab_pred.view(ab_pred.size(0), -1).cpu().numpy()
        ab_pred = ab_list[np.argmax(ab_pred, 0)]
        X_a = ab_pred[:,0].reshape((1,224,224))
        X_b = ab_pred[:,1].reshape((1,224,224))
        ab_pred = np.concatenate((X_a, X_b), 0)

        ab_raw_test = target[i].view(-1).cpu().numpy()
        ab_raw_test = ab_list[ab_raw_test]
        X_a_raw = ab_raw_test[:,0].reshape((1,224,224))
        X_b_raw = ab_raw_test[:,1].reshape((1,224,224))
        ab_raw_test = np.concatenate((X_a_raw, X_b_raw), 0)

        # raw = torch.cat((l*100., ab_raw*110.), 0).numpy()
        raw = np.concatenate((l.numpy()*100., ab_raw_test), 0)
        pred = np.concatenate((l.numpy()*100., ab_pred), 0)
        raw = np.transpose(raw, (1,2,0))
        pred = np.transpose(pred, (1,2,0))
        raw_rgb = color.lab2rgb(np.float64(raw))
        pred_rgb = color.lab2rgb(np.float64(pred))

        grey = l.cpu().numpy()
        grey = np.transpose(grey, (1,2,0))
        grey = np.repeat(grey, 3, axis=2).astype(np.float64)
        img_list.append(np.concatenate((grey, raw_rgb, pred_rgb), 1))

    img_list = [np.concatenate(img_list[4*i:4*(i+1)], axis=1) for i in range(len(img_list) // 4)]
    img_list = np.concatenate(img_list, axis=0)

    plt.figure(figsize=(24,18))
    plt.imshow(img_list)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(img_path + 'epoch%d_val.png' % epoch)
    plt.clf()

if __name__ == '__main__':
    main()
