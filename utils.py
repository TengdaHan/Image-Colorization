import torch
import shutil
import numpy as np
import pickle

import re
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def save_checkpoint(state, is_best=0, filename='models/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'models/model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history = []
        self.dict = {} # save all data values here
        self.save_dict = {} # save mean and std here, for summary table

    def update(self, val, n=1, history=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if history:
            self.history.append(val)

    def dict_update(self, val, key):
        # import pdb; pdb.set_trace()
        if key in self.dict.keys():
            self.dict[key].append(val)
        else:
            self.dict[key] = [val]

    def print_dict(self, title='IoU', save_data=False):
        """Print summary, clear self.dict and save mean+std in self.save_dict"""
        total = []
        for key in self.dict.keys():
            val = self.dict[key]
            avg_val = np.average(val)
            len_val = len(val)
            std_val = np.std(val)

            if key in self.save_dict.keys():
                self.save_dict[key].append([avg_val, std_val])
            else:
                self.save_dict[key] = [[avg_val, std_val]]

            print('Activity:%s, mean %s is %0.4f, std %s is %0.4f, length of data is %d' \
                % (key, title, avg_val, title, std_val, len_val))

            total.extend(val)

        self.dict = {}
        avg_total = np.average(total)
        len_total = len(total)
        std_total = np.std(total)
        print('\nOverall: mean %s is %0.4f, std %s is %0.4f, length of data is %d \n' \
            % (title, avg_total, title, std_total, len_total))

        if save_data:
            print('Save %s pickle file' % title)
            with open('img/%s.pickle' % title, 'wb') as f:
                pickle.dump(self.save_dict, f)

    def __len__(self):
        return self.count

def print_table(obj_list, title_list, save_list):
    '''print_table([losses, iou_score, precision, recall, c_losses, c_accuracy])
       generates a latex table
    '''
    key_list = obj_list[0].dict.keys()
    assert key_list.__len__() == 21
    for key in key_list:
        for obj, title in zip(obj_list, title_list):
            pass



def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

class Plotter(object):
    """plot loss and accuracy, require import matplotlib.pyplot as plt"""
    def __init__(self):
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []

        # for classification
        self.train_c_loss = []
        self.train_c_acc = []
        self.val_c_loss = []
        self.val_c_acc = []

    def train_update(self, loss, acc, c_loss=None, c_acc=None):
        if type(loss) != float:
            loss = float(loss)
        if type(acc) != float:
            acc = float(acc)
        self.train_loss.append(loss)
        self.train_acc.append(acc)

        # for classification
        if (c_loss != None) and (c_acc != None):
            if type(c_loss) != float:
                c_loss = float(c_loss)
            if type(c_acc) != float:
                c_acc = float(c_acc)
            self.train_c_loss.append(c_loss)
            self.train_c_acc.append(c_acc)

    def val_update(self, loss, acc, c_loss=None, c_acc=None):
        if type(loss) != float:
            loss = float(loss)
        if type(acc) != float:
            acc = float(acc)
        self.val_loss.append(loss)
        self.val_acc.append(acc)

        # for classification
        if (c_loss != None) and (c_acc != None):
            if type(c_loss) != float:
                c_loss = float(c_loss)
            if type(c_acc) != float:
                c_acc = float(c_acc)
            self.val_c_loss.append(c_loss)
            self.val_c_acc.append(c_acc)

    def export_valacc(self, filename):
        pickle.dump(self.val_acc, open(filename+'.pickle', 'wb'))

    def draw(self, filename):
        if len(self.train_c_loss) == 0:
            if len(self.val_loss) == len(self.train_loss):
                zipdata = zip([[self.train_loss, self.val_loss],
                               [self.train_acc, self.val_acc]],
                              ['loss', 'iou'])
                for i, (data, name) in enumerate(zipdata):
                    plt.subplot(1, 2, i+1)
                    plt.plot(data[0], label='train')
                    plt.plot(data[1], label='val')
                    plt.legend(loc='upper left')
                    plt.xlabel('epoch')
                    plt.title(name)
                plt.tight_layout()
                plt.savefig(filename)
                plt.clf()
            else: # no validation data
                zipdata = zip([self.train_loss, self.train_acc],
                              ['loss', 'iou'])
                for i, (data, name) in enumerate(zipdata):
                    plt.subplot(1, 2, i+1)
                    plt.plot(data, label='train')
                    plt.legend(loc='upper left')
                    plt.xlabel('iteration')
                    plt.title(name)
                plt.tight_layout()
                plt.savefig(filename)
                plt.clf()
        else:
            if (len(self.val_loss) == len(self.train_loss)) \
                and (len(self.val_c_loss) == len(self.train_c_loss)): # train val
                zipdata = zip([[self.train_loss, self.val_loss],
                               [self.train_acc, self.val_acc],
                               [self.train_c_loss, self.val_c_loss],
                               [self.train_c_acc, self.val_c_acc]],
                              ['segmentation loss',
                               'segmentation iou',
                               'classification loss',
                               'classification accuracy'])
                for i, (data, name) in enumerate(zipdata):
                    plt.subplot(2,2,i+1)
                    plt.plot(data[0], label='train')
                    plt.plot(data[1], label='val')
                    plt.legend(loc='upper left')
                    plt.xlabel('iteration')
                    plt.ylabel(name)
                plt.tight_layout()
                plt.savefig(filename)
                plt.clf()
            elif (len(self.val_loss) != len(self.train_loss)) \
                and (len(self.val_c_loss) != len(self.train_c_loss)): # No val
                zipdata = zip([self.train_loss,
                               self.train_acc,
                               self.train_c_loss,
                               self.train_c_acc],
                              ['segmentation loss',
                               'segmentation iou',
                               'classification loss',
                               'classification accuracy'])
                for i, (data, name) in enumerate(zipdata):
                    plt.subplot(2,2,i+1)
                    plt.plot(data, label='train')
                    plt.legend(loc='upper left')
                    plt.xlabel('iteration')
                    plt.ylabel(name)
                plt.tight_layout()
                plt.savefig(filename)
                plt.clf()
            else:
                raise 'Error: data error in Plotter'



class AccuracyTable(object):
    """compute accuracy for each class"""
    def __init__(self):
        self.dict = {}

    def update(self, pred, tar):
        pred = torch.squeeze(pred)
        tar = torch.squeeze(tar)
        for i, j in zip(pred, tar):
            i = int(i)
            j = int(j)
            if j not in self.dict.keys():
                self.dict[j] = {'count':0,'correct':0}
            self.dict[j]['count'] += 1
            if i == j:
                self.dict[j]['correct'] += 1

    def print_table(self, label):
        for key in self.dict.keys():
            acc = self.dict[key]['correct'] / self.dict[key]['count']
            print('%s: %2d, accuracy: %3d/%3d = %0.6f' \
                % (label, key, self.dict[key]['correct'], self.dict[key]['count'], acc))


# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count

class ConfusionMeter(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.mat = np.zeros((num_class, num_class))
        self.precision = []
        self.recall = []

    def update(self, pred, tar):
        pred = torch.squeeze(pred)
        tar = torch.squeeze(tar)
        for p,t in zip(pred, tar):
            self.mat[p][t] += 1

    def print_mat(self):
        print('Confusion Matrix for Testing')
        print(self.mat)

    def plot_mat(self, path):
        plt.imshow(self.mat,
            cmap=plt.cm.jet,
            interpolation='nearest',
            extent=(0.5, np.shape(self.mat)[0]+0.5, np.shape(self.mat)[1]+0.5, 0.5))
        width, height = self.mat.shape
        for x in range(width):
            for y in range(height):
                plt.annotate(str(int(self.mat[x][y])), xy=(y+1, x+1),
                             horizontalalignment='center',
                             verticalalignment='center')
        plt.colorbar()
        plt.savefig(path)
        plt.clf()

        for i in range(width):
            if np.sum(self.mat[i,:]) != 0:
                self.precision.append(self.mat[i,i] / np.sum(self.mat[i,:]))
            if np.sum(self.mat[:,i]) != 0:
                self.recall.append(self.mat[i,i] / np.sum(self.mat[:,i]))
        print('Average Precision: %0.4f' % np.mean(self.precision))
        print('Average Recall: %0.4f' % np.mean(self.recall))
