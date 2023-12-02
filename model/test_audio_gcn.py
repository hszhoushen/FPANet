import sys

sys.path.append('../')
sys.path.append('./')
sys.path.append("../model/")
sys.path.append("../resnet-audio/")
sys.path.append("../resnet-image/")
sys.path.append("./data/")
sys.path.append('../utils/')

import torch
import time
import shutil
import torch.nn as nn
from torch.utils.data import DataLoader
from model.CVS_dataset import CVS_Audio
from model.CVS_dataset import sound_inference
from model.CVS_dataset import sound_plot
import os
import numpy as np

from model.network import GCN_audio
from utils.arguments import arguments_parse
from resnet_audio import resnet50
from tensorboardX import SummaryWriter

from utils.arguments import dataset_selection

best_prec1 = 0

def main():

    global best_prec1

    # argsPaser
    argsPaser = arguments_parse()
    args = argsPaser.argsParser()
    print('args:', args)

    # data construction
    print('args.dataset_name:', args.dataset_name)
    data_selection = dataset_selection()
    data_dir, data_sample = data_selection.datsetSelection(args)


    # training set
    # audio_dataset = CVS_Audio(args, data_dir, data_sample, data_type='train')
    # audio_dataloader = DataLoader(dataset=audio_dataset, batch_size=args.batch_size, shuffle=True,
    #                               num_workers=args.num_threads)

    # test set
    test_audio_dataset = CVS_Audio(args, data_dir, data_sample, data_type='test')
    test_audio_dataloader = DataLoader(dataset=test_audio_dataset, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_threads)

    # best model name and latest model name
    latest_model_name =  '../weights/' + args.model_type + '_latest' + '.pth.tar'
    best_model_name = '../weights/' + args.model_type + '_best' + '.pth.tar'
    best_model_name = '/data/lgzhou/esc/weights/audio_gcn_esc50_3_best.pth.tar'

    # load the pretrained model
    audio_gcn_state = torch.load(best_model_name)

    # create the model for classification
    audio_net = resnet50(num_classes=args.num_classes)
    audio_net.load_state_dict(audio_gcn_state['audio_net_state_dict'])
    audio_net = audio_net.cuda()

    # create gcn model
    audio_gcn_model = GCN_audio(args.num_classes).cuda()
    audio_gcn_model.load_state_dict(audio_gcn_state['audio_gcn_model_state_dict'])
    audio_gcn_model = audio_gcn_model.cuda()

    # for testing
    audio_net.eval()
    audio_gcn_model.eval()

    # test test
        # test_losses, test_acc1, test_acc5 = net_validate(args, audio_net, audio_gcn_model,
    #                                                   test_audio_dataloader)
    # is_best = test_acc1 > best_prec1
    # best_prec1 = max(test_acc1, best_prec1)
    # print("The best accuracy obtained during training is = {}".format(best_prec1))

    # test one audio
    data_dir = '/data/lgzhou/dataset/ESC-50/audio/'
    wave_file = '1-104089-B-22.wav'     # clapping
    wave_file = '1-12654-A-15.wav'      # water drops
    wave_file = '1-160563-A-48.wav'     # fireworks
    wave_file = '3-203371-A-39.wav'     # glass_breaking
    # wave_file = '3-253084-E-2.wav'      # pig


    label = np.array(int(wave_file.split('-')[3].split('.')[0]))
    label = torch.from_numpy(label)
    label = label.cuda()

    sound_path = os.path.join(data_dir, wave_file)
    dataset_name = 'esc50'

    sound_plot(dataset_name, sound_path, label)
    sound = sound_inference(dataset_name, sound_path)
    sound = torch.from_numpy(sound)
    sound = sound.type(torch.FloatTensor).cuda()
    print('sound.shape:', sound.shape)

    output = audio_net(sound)  # here for extracting the event predictions
    # print('output.shape:', output.shape)
    Faudio = output[2]
    print('Faudio.shape:', Faudio.shape)
    audio_features = my_forward(Faudio)
    # print('audio_features.shape:', audio_features.shape)
    print('finish my forward')
    gcn_output = audio_gcn_model(audio_features)
    print('gcn_output.shape:', gcn_output.shape)

    print('gcn_output:', gcn_output)



def sound_test(args, audio_net, audio_gcn_model, data_loader):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    audio_net.eval()
    audio_gcn_model.eval()

    criterion = nn.CrossEntropyLoss().cuda()

    end = time.time()

    with torch.no_grad():

        for i, data in enumerate(data_loader, 0):
            # measure data loading time
            data_time.update(time.time() - end)

            aud, label = data
            label = label.cuda()
            aud = aud.type(torch.FloatTensor).cuda()
            print('aud.shape:', aud.shape)

            if (aud.shape[0] != 16):
                continue

            output = audio_net(aud)            # here for extracting the event predictions
            Faudio = output[2]
            audio_features = my_forward(Faudio)
            gcn_output = audio_gcn_model(audio_features)

            # compute gradient and loss for SGD step
            loss = criterion(gcn_output, label)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(gcn_output, label, topk=(1, 5))

            losses.update(loss, aud.size(0))
            top1.update(prec1, aud.size(0))
            top5.update(prec5, aud.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            break


    return losses.avg, top1.avg, top5.avg

def save_checkpoint(state, is_best, best_model_name, latest_model_name):
    torch.save(state, latest_model_name)
    if is_best:
        shutil.copyfile(latest_model_name, best_model_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def net_validate(args, audio_net, audio_gcn_model, data_loader):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    audio_net.eval()
    audio_gcn_model.eval()

    criterion = nn.CrossEntropyLoss().cuda()

    end = time.time()

    with torch.no_grad():

        for i, data in enumerate(data_loader, 0):
            # measure data loading time
            data_time.update(time.time() - end)

            aud, label = data
            label = label.cuda()
            aud = aud.type(torch.FloatTensor).cuda()
            print('aud.shape:', aud.shape)
            print('label:', label, type(label))

            if (aud.shape[0] != 16):
                continue

            output = audio_net(aud)            # here for extracting the event predictions
            Faudio = output[2]
            audio_features, topK_index = my_forward(Faudio)
            print('topK_index:', topK_index)

            gcn_output = audio_gcn_model(audio_features)

            # compute gradient and loss for SGD step
            loss = criterion(gcn_output, label)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(gcn_output, label, topk=(1, 5))

            losses.update(loss, aud.size(0))
            top1.update(prec1, aud.size(0))
            top5.update(prec5, aud.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(
                      #'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      #'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(data_loader), loss=losses, top1=top1, top5=top5))
            break

    return losses.avg, top1.avg, top5.avg


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lri))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print('args.lr after adjustment:', param_group['lr'])


def my_forward(Faudio):

    N = Faudio.shape[0]
    C = Faudio.shape[1]
    W = Faudio.shape[2]
    H = Faudio.shape[3]
    print('N:', N, 'C:', C, 'W:', W, 'H:', H)

    K = 16
    Fre = torch.empty(N, W, H).cuda()

    # acquire Fre (N,H,W) from Frgb (N,C,H,W)
    for channel in range(C):
        Fre = torch.add(Faudio[:,channel,:,:], Fre)
    # print('Fre:', Fre.shape, Fre[0,:,:])

    # Acquire the top K index of Frgb
    feature = Fre.view(N, H*W)
    # print('feature:', feature.shape)
    sorted, indices = torch.sort(feature, descending=True)
    # print('sorted:', sorted.shape, sorted)
    print('indices:', indices.shape, indices)

    topK_index = indices[:, 0:K]
    print('topK_index:', topK_index.shape, topK_index)

    nodes = []
    rows = []
    columns = []
    cnt = 0

    # shape of topK_index is (N, nodes)
    for i in range(N):      # lop from one image to other
        for j in range(K):  # loop from one node to other
        # for index in topK_index[0, :]:

            # print('i:', i, 'j:', j)
            index = topK_index[i, j]

            if index / H == 0:      # first row
                rows.append(index / H)
            else:
                rows.append(index / H - 1)

            if index % H == 0:      # e.g., 36 (8,3)
                columns.append(H-1)
            else:
                columns.append(index % H - 1)

            # print('index:', index, 'row:', rows[cnt], 'column:', columns[cnt])

            # switch the feature from 1x1024 to 1024x1
            nodes.append(Faudio[i, :, rows[cnt], columns[cnt]].reshape(-1,1))
            cnt = cnt + 1

        imgs = torch.stack(nodes)

    print('rows:')
    for row in rows:
        print(row.data)
    print('columns:', columns)
    for column in columns:
        print(column.data)

    print('imgs:', imgs.shape)
    print('topK_index:', topK_index)

    return imgs, topK_index


# main function
if __name__ == '__main__':
    main()



