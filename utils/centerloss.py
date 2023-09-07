import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


class CenterLoss(nn.Module):
    """
    Adapted from the github repo of the CornerNet paper
    https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py
    """

    def __init__(self, alpha=2, beta=4, eps=1e-4):
        super(CenterLoss, self).__init__()
        self.a = alpha
        self.b = beta
        self.eps = eps

    def forward(self, preds, gt):
        pred = preds.permute(0, 2, 3, 1).contiguous().view(-1, preds.shape[1])
        g = gt.reshape(-1, preds.shape[1])

        pos_inds = g.eq(1)
        neg_inds = g.lt(1)
        num_pos = pos_inds.float().sum()
        loss = 0
        
        pred = pred.clamp(min=self.eps,max=1-self.eps)
        
        pos_loss = torch.log(pred[pos_inds]) * torch.pow(1 - pred[pos_inds], self.a)
        pos_loss = pos_loss.sum()

        neg_loss = torch.log(1 - pred[neg_inds])
        neg_loss = neg_loss*torch.pow(pred[neg_inds], self.a)
        neg_loss = neg_loss*torch.pow(1-g[neg_inds], self.b)
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss)/num_pos
        return loss
    
