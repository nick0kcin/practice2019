from torch.nn import Module
from torch.nn.functional import l1_loss


class L1Loss(Module):

    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, pred, gt):
        return l1_loss(pred, gt)


def get_l1_loss():
    return L1Loss()
