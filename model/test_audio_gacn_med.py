# Evaluation
import sys
sys.path.append('../')
sys.path.append('./')
sys.path.append("../model/")
sys.path.append("../resnet-audio/")
sys.path.append("../resnet-image/")
sys.path.append("./data/")
sys.path.append("../utils/")

import torch
import time
import shutil
import argparse

from torch.utils.data import DataLoader
from model.CVS_dataset import CVS_Audio
from model.network import GCN_audio_top_med
from arguments import arguments_parse
from arguments import dataset_selection
from model.CVS_dataset import sound_inference

from tensorboardX import SummaryWriter
from model.CVS_dataset import sound_plot
from matplotlib.patches import Circle, Rectangle, Arc, Ellipse

import numpy as np
import os
import torch.nn as nn
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.models as models

from model.network import AFM
from model.network import Audio_Fusion_Classifier
from model.network import audio_med_forward


best_prec1 = 0
Best_model_path = '/data/lgzhou/esc/weights/esc50/audio_gacn_med_5_16_epoch_20_best.pth.tar'

def rec_cal(index_width, index_height, feature_width, feature_height, timemax):
    feature_height_max = 8192
    min_f0, max_f0 = (index_width) * (timemax / feature_width), (index_width + 1) * (timemax / feature_width)
    min_f1, max_f1 = (index_height) * (feature_height_max / feature_height), (index_height + 1) * (
                feature_height_max / feature_height)
    width = max_f0 - min_f0
    height = max_f1 - min_f1

    return min_f0, min_f1, width, height


def drawgraph(soundpath, figname, rows, columns, feature_width=13, feature_height=4):
    wav, sr = librosa.load(soundpath, sr=16000)

    timemax = librosa.get_duration(y=wav, sr=16000, S=None, n_fft=2048, center=True, filename=None)

    melspec = librosa.feature.melspectrogram(wav, sr, n_fft=1024, hop_length=512, n_mels=64)  # n_mels 表示频域
    # convert to log scale
    logmelspec = librosa.power_to_db(melspec)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    librosa.display.specshow(logmelspec, sr=sr, x_axis='time', y_axis='mel')

    margin = .1

    pic_elements = []

    print('len(rows):', len(rows))
    for i in range(len(rows)):
        min_f0, min_f1, width, height = rec_cal(rows[i], columns[i], feature_width, feature_height, timemax)
        rectangle = patches.Rectangle(
            xy=(min_f0, min_f1),  # point of origin.
            width=width,
            height=height,
            linewidth=1,
            color='green',
            fill=False
        )
        pic_elements.append(rectangle)
        if (i<8):
            clr = "gold"
        elif(i>=16 and i<24):
            clr = "deepskyblue"
        elif(i>=8 and i<16):
            clr = "khaki"
        else:
            clr = "skyblue"
        # add marker
        if (i<16):
            node_num = i
            al = 0.9
        else:
            node_num = i-16
            al = 0.8
        if ((i + 1) < 10):
            plt.annotate(str(node_num + 1), (min_f0 + width / 3.5, min_f1 + height / 3),
                         bbox={"boxstyle": "circle", "color": clr, "pad": 0.001, "alpha": al})
        else:
            plt.annotate(str(node_num + 1), (min_f0 + width / 10, min_f1 + height / 3),
                         bbox={"boxstyle": "circle", "color": clr, "pad": 0.0002, "alpha": al})

    # List of elements to be plotted
    for element in pic_elements:
        ax.add_patch(element)

    plt.savefig(figname)
    # plt.show()

def returnCAM(feature_conv, weight_softmax):
    # generate the class activation maps upsample to 256x256
    feature_conv = feature_conv.cuda().data.cpu().numpy()
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in range(208):
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
#        output_cam.append(np.array(Image.fromarray(cam_img).resize(size_upsample)))
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def main():
    global best_prec1

    # argsPaser
    argsPaser = arguments_parse()
    args = argsPaser.argsParser()
    print('args:', args)

    # esc50 construction
    print('args.dataset_name:', args.dataset_name)
    data_selection = dataset_selection()
    data_dir, data_sample = data_selection.datsetSelection(args)


    # test set
    test_audio_dataset = CVS_Audio(args, data_dir, data_sample, data_type='test')
    test_audio_dataloader = DataLoader(dataset=test_audio_dataset, batch_size=args.bs, shuffle=True,
                                       num_workers=args.num_threads)

    # best model name and latest model name
    best_model_name = Best_model_path

    # load the pretrained model
    audio_gcn_state = torch.load(best_model_name)

    # Attention Model Initialization
    AFM_net = AFM(args, pretrain=False)
    AFM_net = AFM_net.cuda()
    state = audio_gcn_state['audio_net_state_dict']
    AFM_net.load_state_dict(state)


    # audio gcn model
    K = args.nodes_num
    audio_gcn_model = GCN_audio_top_med(args).cuda()
    state = audio_gcn_state['audio_gcn_model_state_dict']
    audio_gcn_model.load_state_dict(state)
    audio_gcn_model = audio_gcn_model.cuda()

    # audio_classifier
    audio_classifier = Audio_Fusion_Classifier(args.num_classes).cuda()
    state = audio_gcn_state['audio_classifier_state_dict']
    audio_classifier.load_state_dict(state)

    # for testing
    AFM_net.eval()
    audio_gcn_model.eval()
    audio_classifier.eval()

    # test one audio
    data_dir = '/data/lgzhou/dataset/ESC-50/audio/'


    # esc10
    # wave_file = '5-198321-A-10.wav' # rain
    # wave_file = '5-198411-C-20.wav' # crying baby
    # wave_file = '5-200334-A-1.wav'  # roster
    # wave_file = '5-200461-A-11.wav' # sea_waves
    # wave_file = '1-104089-B-22.wav' # clapping
    # wave_file = '1-12654-A-15.wav'  # water drops
    # wave_file = '1-160563-A-48.wav' # fireworks
    # wave_file = '3-203371-A-39.wav' # glass_breaking

    # wave_file = '3-253084-E-2.wav'  # pig
    # wave_file = '1-100038-A-14.wav' # chirping_birds
    # wave_file = '1-100210-A-36.wav' # vacuum_cleaner
    # wave_file = '1-101296-A-19.wav' # thunderstorm
    # wave_file = '1-101336-A-30.wav' # door_wood_knock
    # wave_file = '1-101404-A-34.wav' # can_opening
    # wave_file = '1-103298-A-9.wav'  # crow
    # wave_file = '1-11687-A-47.wav'  # airplane

    # wave_file = '1-121951-A-8.wav'  # sheep
    # wave_file = '1-118206-A-31.wav' # mouse_click
    # wave_file = '1-118559-A-17.wav' # pouring_water
    # wave_file = '1-13571-A-46.wav'  # church_bells
    # wave_file = '1-15689-A-4.wav'   # frog
    # wave_file = '1-17092-A-27.wav'  # brushing_teeth
    # wave_file = '1-172649-B-40.wav' # helicopter
    # wave_file = '1-17295-A-29.wav'  # drinking_sipping

    # wave_file = '1-17367-A-10.wav'  # rain
    # wave_file = '1-39901-A-11.wav'  # sea_waves
    # wave_file = '1-60997-A-20.wav'  # crying_baby

    wave_file_lst = ['5-198321-A-10.wav', '5-198411-C-20.wav', '5-200334-A-1.wav', '5-200461-A-11.wav',
                     '1-104089-B-22.wav','1-12654-A-15.wav', '1-160563-A-48.wav','3-203371-A-39.wav',
                     '3-253084-E-2.wav','1-100038-A-14.wav','1-100210-A-36.wav','1-101296-A-19.wav',
                     '1-101336-A-30.wav','1-101404-A-34.wav','1-103298-A-9.wav' ,'1-11687-A-47.wav',
                     '1-121951-A-8.wav', '1-118206-A-31.wav','1-118559-A-17.wav','1-13571-A-46.wav',
                     '1-15689-A-4.wav', '1-17092-A-27.wav', '1-172649-B-40.wav', '1-17295-A-29.wav']
    label_lst = ['rain', 'crying baby', 'roster', 'sea_waves',
                 'clapping', 'water_drops','fireworks','glass_breaking',
                 'pig','chirping_birds','vacuum_cleaner','thunderstorm',
                 'door_wood_knock','can_opening','crow','airplane',
                 'sheep','mouse_click','pouring_water','church_bells',
                 'frog','brushing_teeth','helicopter','drinking_sipping']
    

    def sound_feature_extract(sound_path, wave_file):
        dataset_name = 'esc50'
        sound = sound_inference(dataset_name, sound_path)
        sound = torch.from_numpy(sound)
        sound = sound.type(torch.FloatTensor).cuda()
        print('sound.shape:', sound.shape)

        label = np.array(int(wave_file.split('-')[3].split('.')[0]))
        label = torch.from_numpy(label)
        label = label.cuda()

        return sound, label

    for i in range(len(wave_file_lst)):
        sound_path = os.path.join(data_dir, wave_file_lst[i])
        sound, label = sound_feature_extract(sound_path, wave_file_lst[i])
        audio_output, _ = AFM_net(sound)
        print('audio_output:', audio_output.shape)

        gcn_output = audio_gcn_model(audio_output)

        sound_graph, Lnormtop, Lnormmed, rows, columns = audio_med_forward(K, audio_output)

        # print('len:', len(rows), len(columns))
        # for (row, column) in zip(rows, columns):
        #     print(row.data, column.data)
        Fig_name = os.path.join('STG', label_lst[i])
        drawgraph(sound_path, Fig_name, rows, columns, feature_width=26, feature_height=8)
        
        params = list(audio_gcn_model.parameters())
        weight_softmax = params[-2].cuda().data.cpu().numpy()
        weight_softmax[weight_softmax<0] = 0
        print('weight size:', weight_softmax[0].shape)
        CAMs = returnCAM(audio_output[0], weight_softmax)
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(26, 8)), cv2.COLORMAP_JET)
        cv2.imwrite('./STG'+label_lst[i]+'heatimg.png', heatmap)

    # audio_features = audio_med_forward(K, Faudio)
    # print('audio_features.shape:', audio_features.shape)
    # gcn_output = audio_gcn_model(audio_features)
    # print('gcn_output.shape:', gcn_output.shape)
    # print('gcn_output:', gcn_output)


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

            output = audio_net(aud)  # here for extracting the event predictions
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
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def net_train(args, audio_net, audio_gcn_model, data_loader, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    audio_net.train()
    audio_gcn_model.train()
    criterion = nn.CrossEntropyLoss().cuda()

    end = time.time()
    for i, data in enumerate(data_loader, 0):
        # measure data loading time
        data_time.update(time.time() - end)

        # clear optimizer
        optimizer.zero_grad()
        aud, label = data

        # to cuda
        label = label.cuda()
        aud = aud.type(torch.FloatTensor).cuda()

        # output = audio_net(aud)
        # output[0].shape, bs x 2048
        # output[1].shape, bs x args.num_classes

        output = audio_net(aud)

        Faudio = output[2]

        gcn_output = audio_gcn_model(Faudio)

        # compute gradient and loss for SGD step
        loss = criterion(gcn_output, label)
        loss.backward()  # loss propagation
        optimizer.step()  # optimizing

        # measure accuracy and record loss
        prec1, prec5 = accuracy(gcn_output, label, topk=(1, 5))

        # print('aud.size(0):', aud.size(0))
        # print('loss:', loss.shape, loss)
        # print('prec1:', prec1.shape, prec1)

        losses.update(loss, aud.size(0))
        top1.update(prec1, aud.size(0))
        top5.update(prec5, aud.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
            # 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            # 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(data_loader), loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def net_validate(args, audio_net, audio_gcn_model, data_loader, optimizer, epoch):
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

            output = audio_net(aud)
            Faudio = output[2]

            gcn_output = audio_gcn_model(Faudio)

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
                print('Epoch: [{0}][{1}/{2}]\t'
                # 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                # 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(data_loader), loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lri))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# main function
if __name__ == '__main__':
    main()


def my_forward_test(Faudio):
    # gcn_model = GCN_audio().cuda()
    # gcn_model.eval()
    # print('gcn_model:', gcn_model)
    crossentropyloss = nn.CrossEntropyLoss().cuda()

    print('Faudio:', Faudio.shape)

    N = Faudio.shape[0]
    C = Faudio.shape[1]
    W = Faudio.shape[2]
    H = Faudio.shape[3]

    Fre = torch.empty(N, W, H).cuda()

    # acquire Fre (N,H,W) from Frgb (N,C,H,W)
    for channel in range(C):
        Fre = torch.add(Faudio[:, channel, :, :], Fre)
    print('Fre:', Fre.shape, Fre[0, :, :])

    # Acquire the top K index of Frgb
    feature = Fre.view(N, H * W)
    print('feature:', feature.shape)
    sorted, indices = torch.sort(feature, descending=True)
    print('sorted:', sorted.shape, sorted)
    print('indices:', indices.shape, indices)

    # feature = feature.view(x.size(0), -1)
    # print('feature:', feature.shape)

    K = 16
    topK_index = indices[:, 0:K]
    print('topK_index:', topK_index.shape, topK_index)

    nodes = []
    rows = []
    columns = []
    cnt = 0

    # shape of topK_index is (N, nodes)
    for i in range(K):  # lop from one image to other
        for j in range(K):  # loop from one node to other
            # for index in topK_index[0, :]:

            index = topK_index[i, j]
            rows.append(index / H - 1)
            columns.append(index % H - 1)
            print('row:', rows[cnt], 'column:', columns[cnt])

            # switch the feature from 1x1024 to 1024x1
            nodes.append(Faudio[i, :, rows[cnt], columns[cnt]].reshape(-1, 1))

            cnt = cnt + 1

        imgs = torch.stack(nodes)
        # nodes = []
    print('imgs:', imgs.shape)

    return feature


