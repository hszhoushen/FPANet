# Developed by Liguang Zhou, 2020.9.30

import argparse
import torchvision.models as models


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
# print('model_names:', model_names)
BATCH_SIZE = 32
Epochs = 40
Learning_rate = 0.01
Momentum = 0.9
Weight_decay = 1e-4
Num_classes = 14
DIS_SCALE = 1.0
Learning_rate_interval = 20

class arguments_parse(object):
    def argsParser():
        parser = argparse.ArgumentParser(description='Training of CIOM model')
        # parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
        parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                            help='model architecture: ' +
                                ' | '.join(model_names) +
                                ' (default: resnet50)')
        parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
        parser.add_argument('--epochs', default=Epochs, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        parser.add_argument('--batch_size', '--bs', default=BATCH_SIZE, type=int,
                            metavar='N', help='mini-batch size (default: 256)')
        parser.add_argument('--lr', '--learning-rate', default=Learning_rate, type=float,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)')
        parser.add_argument('--print-freq', '-p', default=500, type=int,
                            metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                            help='evaluate model on validation set')
        # parser.add_argument('--pretrain', default='AudioSet', type=str,
        #                     help='use pre-trained model on ImageNet, AudioSet, or None')
        parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")

        parser.add_argument('--num_classes',default=Num_classes, type=int,
                            help='num of class in the model')
        parser.add_argument('--om_type', default='copm_resnet50', type=str,
                            help='choose the type of object model')
        parser.add_argument('--DIS_SCALE',default=DIS_SCALE,type=float,
                            help='choose the scale of discriminative matrix')
        parser.add_argument('--model_type', type=str, default='audio_gcn',
                            help='choose the type of object model')
        parser.add_argument('--atten_type', type=str, default='fpam',
                            help='choose the type of attention to use')
        parser.add_argument('--graph_type', type=str, default='sag',            # sag, cag, fusion
                            help='choose the type of graph to use')

        parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                            help='mixed precision opt level, if O0, no amp is used')
        parser.add_argument('--lri', '--learning-rate-interval', type=int, default=Learning_rate_interval,
                            metavar='LRI', help='learning rate interval')
        parser.add_argument('--nodes_num', type=int, default=16,
                            help='num of nodes in sound texture graph')
        parser.add_argument('--dataset_name', default='Places365_7', type=str,
                            help='Choose the dataset used for training')
        parser.add_argument('--test_set_id', type=int, default=5, help='test set id 5')
        parser.add_argument('--experiment_id', type=str, default='7f', help='experiment_id configuration')
        parser.add_argument('--fusion', type=bool, default=False,
                            help='the model is the  single model or fusion model')
        parser.add_argument('--status', default='train', type=str,
                            help='the training statues or evaluation status')
        parser.add_argument('--model_name', default='image_gcn_med_7f_pt_pafm_2_16_epoch_45.pth.tar', type=str,
                            help='choose the model name for evaluation')
        parser.add_argument('--local_rank', default=0, type=int,
                            help='rank for distributed training')
        # 不要改该参数，系统会自动分配
        parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
        # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
        parser.add_argument('--world-size', default=1, type=int,
                            help='number of distributed processes')
        parser.add_argument('--audio_net_weights', type=str, default='./weights/audioset_audio_pretrain.pt',
                            help='audio net weights')


        args = parser.parse_args()

        return args
    
    def test_argsParser():
        parser = argparse.ArgumentParser(description='Testing of CIOM model')
        parser.add_argument('--dataset',default='Places365-14',type=str,
                            help='Choose the dataset used for training')
        parser.add_argument('--num_classes',default=14, type=int,
                            help='num of class in the model')
        parser.add_argument('--om_type', default='copm_resnet50', type=str,
                            help='choose the type of object model')
        parser.add_argument('--DIS_SCALE',default=DIS_SCALE,type=float,
                            help='choose the scale of discriminative matrix')

        args = parser.parse_args()
        return args

