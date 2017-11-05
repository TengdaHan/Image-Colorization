import torch
import torch.nn.functional as F
import torch.nn as nn


# this may be unstable sometimes.Notice set the size_average
def CrossEntropy2d(input, target, weight=None, size_average=True):
    # input:(n, c, h, w) target:(n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c).view(-1, c) >= 0]
    log_p = log_p.view(-1, c)

    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss


def BCE2d(input, target, weight=None, size_average=True):
    # input:(n, 1, h, w) target:(n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    BCEWithLogits = nn.BCEWithLogitsLoss(weight=weight, size_average=False)
    log_p = input
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    # import pdb; pdb.set_trace()

    loss = BCEWithLogits(torch.squeeze(log_p), target.type(torch.FloatTensor).cuda())
    # import pdb; pdb.set_trace()

    if size_average:
        loss /= mask.data.sum()

    return loss
