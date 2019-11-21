
import os
import torch.utils.data
from torch.cuda import FloatTensor
from torch import device
from opts import opts
from models.models import create_model, load_model, save_model
from trainer import Trainer
from datasets.toloka_dataset import TolokaDataset
from losses.losses import create_loss


if __name__ == '__main__':
    opt = opts().parse()

    torch.manual_seed(opt.seed)
    torch.set_default_tensor_type(FloatTensor)
    torch.backends.cudnn.benchmark = True
    train_dataset = TolokaDataset(
        "/workspace/singapore/data.tsv",
        ("amedataimages", "/workspace/amedataimages"),
        down_ratio=opt.down_ratio, output_dim=opt.input_res, rotate=opt.rotate)
    val_dataset = TolokaDataset(
        "/workspace/singapore/data.tsv",
        ("amedataimages", "/workspace/amedataimages"),
        down_ratio=opt.down_ratio, output_dim=opt.input_res, augment=False)
    print(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    model = create_model(opt.arch, {"center": train_dataset.num_classes, "wh": 2}, opt.head_conv)
    params = model.__getattribute__(opt.train_params)()
    optimizer = torch.optim.Adam(params, opt.lr)
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)
    losses = {"center":  create_loss("focal_loss"), "wh": create_loss("l1_loss")}
    loss_weights = {"center": opt.center_weight, "wh": opt.wh_weight}
    trainer = Trainer(model, losses, loss_weights, optimizer=optimizer, device=opt.device, print_iter=opt.print_iter,
                      num_iter=opt.num_iters, batches_per_update=opt.batches_per_update)
    trainer.set_device(opt.gpus, opt.device)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    best = 1e10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train = trainer.train(epoch, train_loader)
        if opt.val_intervals > 0 and not(epoch % opt.val_intervals):
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
            with torch.no_grad():
                log_dict_val = trainer.val(epoch, val_loader)
            if log_dict_val[opt.metric] < best:
                best = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                           epoch, model)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if epoch in opt.batch_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            opt.batches_per_update *= 2
            print('Increase Batch size to', opt.batches_per_update * opt.batch_size)