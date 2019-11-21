from torch.nn import Module
import torch


class FocalLoss(Module):

    def __init__(self, a=2, b=4):
        super(FocalLoss, self).__init__()
        self.a = a
        self.b = b

    def forward(self, pred, gt):
        pos_inds = gt.gt(0.5).float()
        neg_inds = gt.lt(0.5).float()

        neg_weights = torch.pow(1 - gt, self.b).float()
        loss = 0

        map = pred.sigmoid().clamp(1e-07, 1 - 1e-07)

        pos_loss = torch.log(map) * torch.pow(1 - map, self.a) * pos_inds
        neg_loss = torch.log(1 - map) * torch.pow(map, self.a) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


def get_focal_loss(a=2, b=4):
    return FocalLoss(a, b)
