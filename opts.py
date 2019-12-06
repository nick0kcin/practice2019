import argparse
import os


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--gray', action='store_true', help='convert rgb images to gray')

        self.parser.add_argument('--exp_id', default='default')

        self.parser.add_argument('--load_model', default='',
                                 help='path to pretrained model')

        self.parser.add_argument('--resume', action='store_true',
                                 help='resume an experiment. ')

        self.parser.add_argument('--gpus', default='0',
                                 help='-1 for CPU, use comma for multiple gpus')

        self.parser.add_argument('--num_workers', type=int, default=4,
                                 help='dataloader threads. 0 for single-thread.')

        self.parser.add_argument('--seed', type=int, default=317,
                                 help='random seed')  # from CornerNet

        self.parser.add_argument('--print_iter', type=int, default=0,
                                 help='disable progress bar and print to screen.')

        self.parser.add_argument('--save_all', action='store_true',
                                 help='save model to disk every 5 epochs.')

        self.parser.add_argument('--arch', default='res_18',
                                 help='model architecture.')

        self.parser.add_argument('--head_conv', type=int, default=-1,
                                 help='conv layer channels for output head')

        self.parser.add_argument('--down_ratio', type=int, default=4,
                                 help='output stride. ')

        self.parser.add_argument('--input_res', type=int, default=512,
                                 help='input height and width. -1.')

        self.parser.add_argument('--lr', type=float, default=1.25e-4,
                                 help='learning rate')

        self.parser.add_argument('--lr_step', type=str, default= '30, 60',
                                 help='drop learning rate by lr_factor')

        self.parser.add_argument('--lr_factor', type=float, default='3',
                                 help='drop learning rate factor')

        self.parser.add_argument('--batch_step', type=str, default='100,200',
                                 help='increase virtual batch_size by 2.')

        self.parser.add_argument('--num_epochs', type=int, default=180,
                                 help='total training epochs.')

        self.parser.add_argument('--batch_size', type=int, default=32,
                                 help='batch size')

        self.parser.add_argument('--num_iters', type=int, default=-1,
                                 help='default: #samples / batch_size.')

        self.parser.add_argument('--val_intervals', type=int, default=1,
                                 help='number of epochs to run validation.')

        self.parser.add_argument('--test_intervals', type=int, default=10,
                                 help='number of validation runs to run metrics computation')

        self.parser.add_argument("--batches_per_update", type=int, default=1,
                                 help="number of processed batches per one weights update")

        self.parser.add_argument('--train_params', type=int, default='0',
                                 help='name of function, which return trainable parameters'
                                 )

        self.parser.add_argument('--train_params_step', type=str, default='2',
                                 help='step  to change  trainable parameters'
                                 )

        self.parser.add_argument('--rotate', type=float, default=0,
                                 help='when not using random crop'
                                      'apply rotation augmentation.')

        self.parser.add_argument('--center_weight', type=float, default=1,
                                 help='loss weight for keypoint heatmaps.')

        self.parser.add_argument('--wh_weight', type=float, default=0.1,
                                 help='loss weight for bounding box size.')

        self.parser.add_argument('--superclass_weight', type=float, default=0,
                                 help='loss weight for superclass keypoints heatmap.')

        self.parser.add_argument("--K", type=int, default=100,
                                 help="number of best rectangles for testing")
        self.parser.add_argument("--thr", type=float, default=0.2,
                                 help="threshold of center map for testing")
        self.parser.add_argument("--window_radius", type=int, default=7,
                                 help="local max radius for center map")

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
        opt.batch_step = [int(i) for i in opt.batch_step.split(',')]

        opt.params_step = [int(i) for i in opt.train_params_step.split(',')]

        if opt.head_conv == -1:  # init default head_conv
            opt.head_conv = 256 if 'dla' in opt.arch else 64

        opt.root_dir = os.path.dirname(__file__)
        opt.data_dir = '/workspace'
        # opt.data_dir = os.path.join(opt.root_dir, '../')
        opt.exp_dir = os.path.join(opt.root_dir, 'exp')
        opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
        print('The output will be saved to ', opt.save_dir)

        if opt.resume and opt.load_model == '':
            opt.load_model = os.path.join(opt.save_dir, 'model_last.pth')
        return opt
