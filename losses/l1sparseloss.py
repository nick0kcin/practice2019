from torch.nn import Module
from torch.nn.functional import l1_loss, max_pool2d
import  torch


class L1SparseLoss(Module):
    def __init__(self):
        super(L1SparseLoss, self).__init__()

    def forward(self, predict, gt):
        pred = predict[:, :, :gt.shape[2], :gt.shape[3]]

        extrema_map = (max_pool2d(gt, 3, stride=1) == gt[:, :, 1:-1, 1:-1]) & (gt[:, :, 1:-1, 1:-1] > 0)
        points = extrema_map.nonzero()
        points[:, 2:] += 1
        gt_wh = gt[points[:, 0], points[:, 1], points[:, 2], points[:, 3]]
        pred_wh = pred[points[:, 0], points[:, 1], points[:, 2], points[:, 3]]

        return l1_loss(pred_wh,  gt_wh, reduction="sum") / (gt_wh.shape[0] + 1e-04)


def get_l1_sparse_loss():
    return L1SparseLoss()
