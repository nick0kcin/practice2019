from torch.nn import Module
import torch


class FocalLoss(Module):

    def __init__(self, sigmoid=True, a=2, b=4):
        super(FocalLoss, self).__init__()
        self.a = a
        self.b = b
        self.sigmoid = sigmoid

    def forward(self, predict, gt):
        pred = predict[:, :, :gt.shape[2], :gt.shape[3]]
        pos_inds = gt.gt(0.99).float()
        neg_inds = gt.lt(0.99).float()

        neg_weights = torch.pow(1 - gt, self.b).float()
        loss = 0

        c_map = pred.sigmoid().clamp(1e-07, 1 - 1e-07) if self.sigmoid else pred.clamp(1e-07, 1 - 1e-07)

        pos_loss = torch.log(c_map) * torch.pow(1 - c_map, self.a) * pos_inds
        neg_loss = torch.log(1 - c_map) * torch.pow(c_map, self.a) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


def get_focal_loss(sigmoid=True, a=2, b=4):
    def get_loss():
        return FocalLoss(sigmoid, a, b)
    return get_loss
