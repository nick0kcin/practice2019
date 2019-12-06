from .focal_loss import get_focal_loss
from .l1loss import get_l1_loss
from .l1sparseloss import get_l1_sparse_loss

_loss_factory = {
    'focal_loss': get_focal_loss(),
    'l1_loss': get_l1_loss,
    'super_class_focal_loss': get_focal_loss(sigmoid=False),
    'l1_sparse_loss': get_l1_sparse_loss,
}


def create_loss(loss):
    get_loss = _loss_factory[loss]
    model = get_loss()
    return model
