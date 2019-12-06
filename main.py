import torchvision
import os
import torch.utils.data
from torch import device
from opts import opts
from models.models import create_model, load_model, save_model
from trainer import Trainer
from datasets.toloka_dataset import TolokaDataset
from losses.losses import create_loss
import json
from logger import Logger
from pycocotools.cocoeval import COCOeval
from transforms.random_erase import random_erase_transform


def test(coco_gt, coco_pred):
    coco_eval = COCOeval(coco_gt, coco_pred, "bbox")
    coco_eval.params.useCats = False
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    opt = opts().parse()

    torch.manual_seed(opt.seed)
    #torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.backends.cudnn.benchmark = True
    train_dataset = TolokaDataset(
        "/workspace/singapore/train_data2.tsv",
        ("amedataimages", "/workspace/amedataimages"),
        transforms=torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            torchvision.transforms.ToTensor(),
            random_erase_transform()
        ]),
        down_ratio=opt.down_ratio, output_dim=opt.input_res, rotate=opt.rotate)
    val_dataset = TolokaDataset(
        "/workspace/singapore/val_data2.tsv",
        ("amedataimages", "/workspace/amedataimages"),
        scales=(2,),
        down_ratio=opt.down_ratio, output_dim=opt.input_res, augment=False)

    test_dataset = TolokaDataset(
        "/workspace/singapore/val_data2.tsv",
        ("amedataimages", "/workspace/amedataimages"),
        scales=(4, 2, 1),
        down_ratio=opt.down_ratio, output_dim=opt.input_res, augment=False)
    print(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    model = create_model(opt.arch, {"center": train_dataset.num_classes, "dim": 2}, opt.head_conv)
    params = model.set_group_param(opt.train_params)
    optimizer = torch.optim.Adam(params, opt.lr)
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)
    losses = {"center":  create_loss("focal_loss"), "dim": create_loss("l1_sparse_loss")}
    loss_weights = {"center": opt.center_weight, "dim": opt.wh_weight}
    trainer = Trainer(model, losses, loss_weights, optimizer=optimizer, device=opt.device, print_iter=opt.print_iter,
                      num_iter=opt.num_iters, batches_per_update=opt.batches_per_update, k=opt.K, thr=opt.thr,
                      window_r=opt.window_radius)
    trainer.set_device(opt.gpus, opt.device)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
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
    coco = None
    logger = Logger(opt.save_dir, opt.resume)
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train = trainer.train(epoch, train_loader)
        if opt.val_intervals > 0 and not(epoch % opt.val_intervals):
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
            require_predict = opt.test_intervals > 0 and not(epoch / opt.val_intervals % opt.test_intervals)
            with torch.no_grad():
                if require_predict:
                    log_dict_val = trainer.val(epoch, test_loader, require_predict=require_predict)
                else:
                    log_dict_val = trainer.val(epoch, val_loader, require_predict=require_predict)

            logger.log(epoch, log_dict_train, log_dict_val[0] if require_predict else log_dict_val)

            if require_predict:
                if not coco:
                    coco = val_dataset.coco(True)
                json.dump(log_dict_val[1], open(opt.save_dir + "/predict.json", "w"))
                coco_predicts = coco.loadRes(opt.save_dir + "/predict.json")
                test(coco, coco_predicts)

            loss = log_dict_val[0]["loss"] if require_predict else log_dict_val["loss"]
            if loss < best:
                best = loss
                save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                           epoch, model)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)
            logger.log(epoch, log_dict_train, {})
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * ((1 / opt.lr_factor) ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            model, optimizer, start_epoch = load_model(
                model, os.path.join(opt.save_dir, 'model_best.pth'), optimizer, True, lr, opt.lr_step)

        if epoch in opt.params_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            opt.train_params += 1
            lr = opt.lr * ((1 / opt.lr_factor) ** (opt.lr_step.index(epoch) + 1))

            model, _, start_epoch = load_model(
                model, os.path.join(opt.save_dir, 'model_best.pth'), optimizer, True, lr, opt.lr_step)

            params = model.set_group_param(opt.train_params)
            optimizer = torch.optim.Adam(params, lr)
            trainer.optimizer = optimizer
            print("changle params to ", opt.train_params)


        if epoch in opt.batch_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            opt.batches_per_update *= 2
            print('Increase Batch size to', opt.batches_per_update * opt.batch_size)