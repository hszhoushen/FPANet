# Developed by Liguang Zhou

import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from model.network import GCN_audio_top_med_fea, GCN_max_med_fusion
from model.network import GCN_audio_top_med
#from model.network import AFM, FPAM
from model.network import Backbone_ESC
from model.network import FPAM_Audio_Visual, FPAM_Fusion_Net

from model.CVS_dataset import CVS_Audio, CVS_Audio_Visual, CVS_Visual
from model.CVS_dataset import sound_inference

from utils.arguments import arguments_parse
from data.dataset import ImageFolderWithPaths, DatasetSelection
from model.CVS_dataset import CVSDataset, Visual_Dataset

import sys

# t-SNE
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt

sys.path.append("./resnet-audio/")
sys.path.append("./resnet_audio/")
sys.path.append("./utils/")

best_prec1 = 0

import librosa
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from PIL import Image

from config import get_config
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data import Mixup
from utils.logger import create_logger
from utils.lr_scheduler import build_scheduler

import torch.distributed as dist
import datetime

import matplotlib
matplotlib.use('Agg')

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def model_path_config(args, epoch):
    # best model name and latest model name
    model_dir = os.path.join('./weights', args.dataset_name)
    # if dir is not exist, make one
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    last_model_name = args.model_type + '_' + str(args.atten_type) + '_' + str(
        args.experiment_id) + '_' + str(args.nodes_num) + \
                      '_epoch_' + str(epoch) + '_' + str(args.lri) + '.pth.tar'

    best_model_name = args.model_type + '_' + str(args.atten_type) + '_' + str(
        args.experiment_id) + '_' + str(args.nodes_num) + \
                      '_epoch_' + str(epoch) + '_' + str(args.lri) + '_best' + '.pth.tar'

    last_model_path = os.path.join(model_dir, last_model_name)
    best_model_path = os.path.join(model_dir, best_model_name)

    return last_model_path, best_model_path


def sound_feature_extract(sound_path, wave_file):
    dataset_name = 'ESC50'
    sound = sound_inference(dataset_name, sound_path)
    sound = torch.from_numpy(sound)
    sound = sound.type(torch.FloatTensor).cuda()

    label = np.array(int(wave_file.split('-')[3].split('.')[0]))
    label = torch.from_numpy(label)
    label = label.cuda()

    return sound, label


def rec_cal(index_width, index_height, feature_width, feature_height, timemax):
    feature_height_max = 8192
    min_f0, max_f0 = (index_width) * (timemax / feature_width), (index_width + 1) * (timemax / feature_width)
    min_f1, max_f1 = (index_height) * (feature_height_max / feature_height), (index_height + 1) * (
            feature_height_max / feature_height)
    width = max_f0 - min_f0
    height = max_f1 - min_f1

    return min_f0, min_f1, width, height


def crop_center(img, cropx, cropy):
    y, x, c = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:starty + cropy, startx:startx + cropx, :]


def rec_cal(index_width, index_height, feature_width, feature_height, timemax):
    feature_height_max = 8192
    min_f0, max_f0 = (index_width) * (timemax / feature_width), (index_width + 1) * (timemax / feature_width)
    min_f1, max_f1 = (index_height) * (feature_height_max / feature_height), (index_height + 1) * (
            feature_height_max / feature_height)
    width = max_f0 - min_f0
    height = max_f1 - min_f1

    return min_f0, min_f1, width, height


def audio_drawgraph(soundpath, figname, rows, columns, feature_width=26, feature_height=8):
    wav, sr = librosa.load(soundpath, sr=16000)

    timemax = librosa.get_duration(y=wav, sr=16000, S=None, n_fft=2048, center=True, filename=None)

    melspec = librosa.feature.melspectrogram(wav, sr, n_fft=1024, hop_length=512, n_mels=64)  # n_mels, frequency domain
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

        if i < len(rows) // 2:
            clr = "gold"
        else:
            clr = "skyblue"

        # if (i<8):
        #     clr = "gold"
        # elif(i>=16 and i<24):
        #     clr = "deepskyblue"
        # elif(i>=8 and i<16):
        #     clr = "khaki"
        # else:
        #     clr = "skyblue"

        # add marker
        if (i < 16):
            node_num = i
            al = 0.9
        else:
            node_num = i - 16
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
    # plt.clf()
    plt.close()
    # plt.show()


def performance_matrix(true, pred):
    from sklearn import metrics
    precision = metrics.precision_score(true, pred, average='macro')
    recall = metrics.recall_score(true, pred, average='macro')
    accuracy = metrics.accuracy_score(true, pred)
    f1_score = metrics.f1_score(true, pred, average='macro')
    matrix = metrics.confusion_matrix(true, pred)
    per_cls_acc = matrix.diagonal() / matrix.sum(axis=1)
    print('Precision: {} Recall: {}, Accuracy: {}: ,f1_score: {}'.format(precision * 100, recall * 100, accuracy * 100,
                                                                         f1_score * 100))
    print('per_cls_acc:', per_cls_acc)



def image_drawgraph(imgpath, savepath, rows, columns):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    margin = .1
    pic_elements = []
    img = Image.open(imgpath)
    # centorcrop
    img = img.resize([256, 256])
    img = crop_center(np.array(img), 224, 224)
    img = Image.fromarray(img)

    # plt.rcParams['figure.figsize'] = (15.0,15.0)
    # plt.rcParams['savefig.dpi'] = 300               # image pixels
    # plt.rcParams['figure.dpi'] = 300                # resolution

    plt.imshow(img)

    print('len(rows):', len(rows))
    show_color = False
    for i in range(len(rows)):
        # min_f0, min_f1, width, height = rec_cal(rows[i], columns[i], feature_width, feature_height, 56)
        # rectangle = patches.Rectangle(
        #     # xy=(min_f0, min_f1),  # point of origin.
        #     xy=(rows[i]*8,columns[i]*8),
        #     width=width,
        #     height=height,
        #     linewidth=1,
        #     color='green',
        #     fill=False
        # )
        # pic_elements.append(rectangle)
        # if (i < 8):
        #     clr = "gold"
        # elif(i >= 16 and i < 24):
        #     clr = "deepskyblue"
        # elif(i >= 8 and i < 16):
        #     clr = "khaki"
        # else:
        #     clr = "skyblue"
        if i < len(rows) // 2:
            clr = "gold"
        else:
            clr = "skyblue"

        if show_color:
            # al=0.9
            ax.add_patch(plt.Circle((rows[i] * 8, columns[i] * 8), 4, color=clr, alpha=0.8))
            # check reliablity
            # ax.annotate(str(rows[i]*8)+","+str(columns[i]*8), (rows[i]*8, columns[i]*8), color=clr, weight='bold', ha='center', va='center', size=4)
            # plt.annotate("", (rows[i]*8, columns[i]*8),bbox={"boxstyle": "circle", "color": clr, "pad": 0.0002, "alpha": al})
        else:
            if i < len(rows) // 2:
                num = i
            else:
                num = i - len(rows) // 2
            # ax.annotate(str(num + 1), (rows[i]*8, columns[i]*8), color=clr, weight='bold', ha='center', va='center', size=7)
            # ax.add_patch()
            ax.add_patch(plt.Circle((rows[i] * 8, columns[i] * 8), 4, color=clr, alpha=0.8))
            if num + 1 >= 10:
                plt.text(rows[i] * 8 - 3, columns[i] * 8 + 1, str(num + 1), color='black', fontsize=3)
            else:
                plt.text(rows[i] * 8 - 2, columns[i] * 8 + 1, str(num + 1), color='black', fontsize=3)
            # ax.add_patch()

            # ax.annotate(str(num + 1), (rows[i]*8, columns[i]*8),fontsize=7, color='black', bbox={"boxstyle": "circle", "color": clr, "pad": 0.001, "alpha": 0.8})
        # add marker
        # if (i < 16):
        #     node_num = i
        #     al = 0.9
        # else:
        #     node_num = i - 16
        #     al = 0.8
        # if ((i + 1) < 10):
        #     plt.annotate(str(node_num + 1), (rows[i]*8, columns[i]*8),
        #                  bbox={"boxstyle": "circle", "color": clr, "pad": 0.001, "alpha": al})
        # else:
        #     plt.annotate(str(node_num + 1), (rows[i]*8, columns[i]*8),
        #                  bbox={"boxstyle": "circle", "color": clr, "pad": 0.0002, "alpha": al})

    # List of elements to be plotted
    # for element in pic_elements:
    #     ax.add_patch(element)

    # remove axis
    plt.axis('off')
    # remove the white boundaries of image
    width = 480
    height = 480
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig(savepath, dpi=300)
    plt.clf()


def thop_params_calculation(network, inputs):
    from thop import profile
    flops, params = profile(network, inputs=(inputs,))

    return flops, params


def params_calculation(FPAM_net, gcn_max_med_model):
    fpam_total_params = sum(param.numel() for param in FPAM_net.parameters())
    gcn_max_med_total_params = sum(param.numel() for param in gcn_max_med_model.parameters())
    print('# FPAM_net parameters:', fpam_total_params)
    print('# gcn_max_med_model parameters:', gcn_max_med_total_params)
    total_params = fpam_total_params + gcn_max_med_total_params
    print('total_params:', total_params)

    return total_params, fpam_total_params, gcn_max_med_total_params


def draw_audio(args, FPAM_net, gcn_max_med_model):
    # load the pretrained-model
    best_model_name = os.path.join('./weights', args.dataset_name, args.model_name)
    print('best_model_name:', best_model_name)
    checkpoint = torch.load(best_model_name)
    print("=> loaded checkpoint '{}' (epoch {}) (accuracy {})"
          .format(args.resume, checkpoint['epoch'], checkpoint['best_prec1']))
    FPAM_net.load_state_dict(checkpoint['audio_net_state_dict'])
    gcn_max_med_model.load_state_dict(checkpoint['gcn_max_med_model_state_dict'])
    FPAM_net = FPAM_net.cuda()
    gcn_max_med_model = gcn_max_med_model.cuda()
    FPAM_net.eval()
    gcn_max_med_model.eval()
    # added
    for param in FPAM_net.parameters():  # freeze netT
        param.requires_grad = False
    for param in gcn_max_med_model.parameters():  # freeze netT
        param.requires_grad = False

    with torch.no_grad():
        FPAM_net = FPAM_net.cuda()
        gcn_max_med_model = gcn_max_med_model.cuda()
        FPAM_net.eval()
        gcn_max_med_model.eval()

        wave_file_lst = ['5-181766-A-10.wav', '5-198411-C-20.wav', '5-200334-A-1.wav', '5-200461-A-11.wav',
                         '5-177957-A-40.wav', '1-21935-A-38.wav', '1-30226-A-0.wav', '1-47250-B-41.wav',
                         '5-186924-A-12.wav', '1-26143-A-21.wav',
                         '1-104089-B-22.wav', '1-12654-A-15.wav', '1-160563-A-48.wav', '3-203371-A-39.wav',
                         '3-253084-E-2.wav', '1-100038-A-14.wav', '1-100210-A-36.wav', '1-101296-A-19.wav',
                         '1-101336-A-30.wav', '1-101404-A-34.wav', '1-103298-A-9.wav', '1-11687-A-47.wav',
                         '1-121951-A-8.wav', '1-118206-A-31.wav', '1-118559-A-17.wav', '1-13571-A-46.wav',
                         '1-15689-A-4.wav', '1-17092-A-27.wav', '1-17295-A-29.wav', ]

        label_lst = ['rain', 'crying baby', 'roster', 'sea_waves',
                     'helicopter', 'clock_tick', 'dog', 'chainsaw',
                     'crackling_fire', 'sneezing',
                     'clapping', 'water_drops', 'fireworks', 'glass_breaking',
                     'pig', 'chirping_birds', 'vacuum_cleaner', 'thunderstorm',
                     'door_wood_knock', 'can_opening', 'crow', 'airplane',
                     'sheep', 'mouse_click', 'pouring_water', 'church_bells',
                     'frog', 'brushing_teeth', 'drinking_sipping']


        for i in range(len(wave_file_lst)):
            sound_path = os.path.join(data_dir, wave_file_lst[i])
            sound, label = sound_feature_extract(sound_path, wave_file_lst[i])
            audio_output, resnet_output = FPAM_net(sound)
            print('audio_output.shape:', audio_output.shape, 'resnet_output.shape:', resnet_output.shape)

            gcn_output, rows, columns = gcn_max_med_model(audio_output, resnet_output)

            print('gcn_output.shape:', gcn_output.shape)

            rows = rows.squeeze()
            columns = columns.squeeze()

            Fig_name = os.path.join('visualization', args.dataset_name, label_lst[i])
            audio_drawgraph(sound_path, Fig_name, rows, columns, feature_width=26, feature_height=8)


def draw_atten(args, FPAM_net):
    os.makedirs('atten_visualization', exist_ok=True)
    os.makedirs(os.path.join('atten_visualization', args.dataset_name), exist_ok=True)
    # load the pretrained-model
    best_model_name = os.path.join('./weights', args.dataset_name, args.model_name)
    print('best_model_name:', best_model_name)
    checkpoint = torch.load(best_model_name)
    print("=> loaded checkpoint '{}' (epoch {}) (accuracy {})"
          .format(args.resume, checkpoint['epoch'], checkpoint['best_prec1']))
    FPAM_net.load_state_dict(checkpoint['audio_net_state_dict'])
    # gcn_max_med_model.load_state_dict(checkpoint['gcn_max_med_model_state_dict'])
    FPAM_net = FPAM_net.cuda()
    # gcn_max_med_model = gcn_max_med_model.cuda()
    FPAM_net.eval()
    # gcn_max_med_model.eval()
    # added
    for param in FPAM_net.parameters():  # freeze netT
        param.requires_grad = False
    # for param in gcn_max_med_model.parameters():  # freeze netT
    #     param.requires_grad = False
    data_dir = "/data/lgzhou/dataset/ESC-50/audio"
    with torch.no_grad():
        FPAM_net = FPAM_net.cuda()
        # gcn_max_med_model = gcn_max_med_model.cuda()
        FPAM_net.eval()
        # gcn_max_med_model.eval()

        # wave_file_lst = ['5-181766-A-10.wav', '5-198411-C-20.wav', '5-200334-A-1.wav', '5-200461-A-11.wav',
        #                  '5-177957-A-40.wav', '1-21935-A-38.wav', '1-30226-A-0.wav', '1-47250-B-41.wav',
        #                  '5-186924-A-12.wav', '1-26143-A-21.wav',
        #                  '1-104089-B-22.wav', '1-12654-A-15.wav', '1-160563-A-48.wav', '3-203371-A-39.wav',
        #                  '3-253084-E-2.wav', '1-100038-A-14.wav', '1-100210-A-36.wav', '1-101296-A-19.wav',
        #                  '1-101336-A-30.wav', '1-101404-A-34.wav', '1-103298-A-9.wav', '1-11687-A-47.wav',
        #                  '1-121951-A-8.wav', '1-118206-A-31.wav', '1-118559-A-17.wav', '1-13571-A-46.wav',
        #                  '1-15689-A-4.wav', '1-17092-A-27.wav', '1-17295-A-29.wav', ]
        #
        # label_lst = ['rain', 'crying baby', 'roster', 'sea_waves',
        #              'helicopter', 'clock_tick', 'dog', 'chainsaw',
        #              'crackling_fire', 'sneezing',
        #              'clapping', 'water_drops', 'fireworks', 'glass_breaking',
        #              'pig', 'chirping_birds', 'vacuum_cleaner', 'thunderstorm',
        #              'door_wood_knock', 'can_opening', 'crow', 'airplane',
        #              'sheep', 'mouse_click', 'pouring_water', 'church_bells',
        #              'frog', 'brushing_teeth', 'drinking_sipping']

        # esc50
        wave_file_lst = ['1-23996-B-35.wav', '1-26176-A-43.wav', '1-26188-A-30.wav', '1-27403-A-28.wav', '1-36400-A-23.wav',
                         '1-46744-A-36.wav', '1-51433-A-17.wav', '1-51805-G-33.wav', '1-53663-A-24.wav', '1-54747-A-46.wav']

        label_lst = ['washing_machine', 'car_horn', 'door_wood_knock', 'snoring', 'breathing',
                     'vacuum_cleaner', 'pouring_water', 'door_wood_creaks', 'coughing', 'church_bells']

        # wave_file_lst = ['1-29561-A-10.wav', '1-34119-A-1.wav', '1-34119-B-1.wav', '1-35687-A-38.wav', '1-39901-B-11.wav']
        # label_lst = ['rain', 'rooster', 'rooster1', 'clock_tick', 'sea_waves']

        # wave_file_lst = ['1-100032-A-0.wav', '1-110389-A-0.wav', '1-116765-A-41.wav', '1-17150-A-12.wav', '1-172649-A-40.wav',
        #                  '1-17367-A-10.wav', '1-187207-A-20.wav','1-21934-A-38.wav', '1-26143-A-21.wav', '1-26806-A-1.wav',
        #                  '1-28135-B-11.wav']
        # label_lst = ['dog', 'dog1', 'chainsaw', 'crackling_fire', 'helicopter',
        #              'rain', 'crying_baby', 'clock_tick', 'sneezing', 'rooster', 'sea_waves']

        # sound_path = os.path.join(data_dir, wave_file_lst[0])
        # sound, label = sound_feature_extract(sound_path, wave_file_lst[0])
        # fpam_flops, fpam_params = thop_params_calculation(FPAM_net, sound)
        # print('fpam_flops:', fpam_flops, 'fpam_params:', fpam_params)


        for i in range(len(wave_file_lst)):
            sound_path = os.path.join(data_dir, wave_file_lst[i])
            sound, label = sound_feature_extract(sound_path, wave_file_lst[i])
            _, atten_output = FPAM_net(sound)
            print('atten_output.shape:', atten_output.shape)    # 1, 1024, 26, 8

            # gcn_output, rows, columns = gcn_max_med_model(audio_output, resnet_output)
            # print('gcn_output.shape:', gcn_output.shape)
            # rows = rows.squeeze()
            # columns = columns.squeeze()

            Fig_name = os.path.join('atten_visualization', args.dataset_name, label_lst[i])
            audio_drawcam(sound_path, Fig_name, atten_output)


def audio_drawcam(soundpath, figname, atten_output):
    wav, sr = librosa.load(soundpath, sr=16000)
    timemax = librosa.get_duration(y=wav, sr=16000, S=None, n_fft=2048, center=True, filename=None)
    melspec = librosa.feature.melspectrogram(wav, sr, n_fft=1024, hop_length=512, n_mels=64)  # n_mels, frequency domain
    max_freq = librosa.mel_frequencies(n_mels=64)[-1]
    print("sr", sr, sr / 2)
    print("timemax", timemax)
    print("max_freq", max_freq)
    # convert to log scale
    logmelspec = librosa.power_to_db(melspec)
    logmelspec_h, logmelspec_w = logmelspec.shape       # 64, 157
    print("logmelspec.shape", logmelspec.shape, logmelspec.dtype, np.max(logmelspec), np.min(logmelspec))
    logmel_max = np.max(logmelspec)
    logmel_min = np.min(logmelspec)

    fig = plt.figure()
    librosa.display.specshow(logmelspec, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar()
    plt.savefig(figname + "_audio")
    plt.close()

    print("atten_output.size()", atten_output.size())   # 1, 1024, 26, 8
    atten_output = torch.nn.functional.interpolate(atten_output, (logmelspec_w, logmelspec_h),
                                                   mode='nearest')  # scale_factor=10

    atten_output = atten_output.squeeze(0).data.cpu().numpy()
    atten_output = np.sum(atten_output, axis=0)
    atten_output_cam = (atten_output - np.min(atten_output)) / (
                np.max(atten_output) - np.min(atten_output))  # Normalize between 0-1


    print('atten_output_cam.shape:',  atten_output_cam.shape)
    atten_output_cam = atten_output_cam * (logmel_max-logmel_min) + logmel_min
    atten_output_cam = np.rot90(atten_output_cam)
    print('atten_output_cam.shape:',  atten_output_cam.shape)

    fig = plt.figure()
    librosa.display.specshow(atten_output_cam, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar()
    plt.savefig(figname + "_audio_atten")
    plt.close()

    # atten_output_cam = np.array(atten_output_cam * 255., dtype=np.float32)  # Scale between 0-255 to visualize
    atten_output_cam = np.uint8(atten_output_cam * 255.)  # Scale between 0-255 to visualize
    atten_output_cam = np.rot90(atten_output_cam)

    atten_output_heatmap = cv2.applyColorMap(atten_output_cam, cv2.COLORMAP_JET)

    # cam is ok
    print("atten_output_cam.shape", atten_output_cam.shape, atten_output_cam.dtype)
    fig = plt.figure()
    # librosa.display.specshow(logmelspec, sr=sr, x_axis='time', y_axis='mel')
    # librosa.display.specshow(atten_output_cam.expand_dims(0), x_axis='time', y_axis='chroma')
    plt.imshow(atten_output_cam, interpolation='nearest', aspect='auto',
               extent=[0, 5, 0, 5 + (max_freq - 8192.) / (32. * 512 - 8192)])
    plt.colorbar()
    plt.xlabel('time')
    plt.ylabel('Hz')
    # plt.xlim([0,5])
    plt.xticks([0, 0.6, 1.2, 1.8, 2.4, 3, 3.6, 4.2, 4.8])
    # plt.ylim([0,5+(max_freq-8192.)/(32.*512-8192)])
    plt.yticks(range(0, 6), ['0', '512', '1024', '2048', '4096', '8192'])
    # plt.legend()
    #plt.savefig(figname + "_cam")
    plt.close()
    #cv2.imwrite(figname + "_cam_img.jpg", atten_output_cam)

    # heatmap seems not very good. the diagram is strange
    print("atten_output_heatmap.shape", atten_output_heatmap.shape, atten_output_heatmap.dtype)
    fig = plt.figure()
    # librosa.display.specshow(logmelspec, sr=sr, x_axis='time', y_axis='mel')
    # librosa.display.specshow(atten_output_heatmap.expand_dims(0), x_axis='time', y_axis='chroma')
    plt.imshow(atten_output_heatmap, interpolation='nearest', aspect='auto',
               extent=[0, 5, 0, 5 + (max_freq - 8192.) / (32. * 512 - 8192)])
    plt.colorbar()
    plt.xlabel('time')
    plt.ylabel('Hz')
    # plt.xlim()
    plt.xticks([0, 0.6, 1.2, 1.8, 2.4, 3, 3.6, 4.2, 4.8])
    # plt.ylim([0,5+(max_freq-8192.)/(32.*512-8192)])
    plt.yticks(range(0, 6), ['0', '512', '1024', '2048', '4096', '8192'])
    # plt.legend()
    # plt.savefig(figname + "_heatmap")
    plt.close()
    # cv2.imwrite(figname + "_heatmap_img.jpg", atten_output_heatmap)



def draw_image(args, FPAM_net, gcn_max_med_model):
    import random
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.use_deterministic_algorithms(True)

    # torch.set_deterministic(True)
    if (args.dataset_name == 'SUNRGBD'):
        dataset_name = 'Places365-7'
    else:
        dataset_name = args.dataset_name

    # load the pretrained-model
    best_model_name = os.path.join('./weights', dataset_name, args.model_name)
    print('best_model_name:', best_model_name)
    checkpoint = torch.load(best_model_name)
    print("=> loaded checkpoint '{}' (epoch {}) (accuracy {})"
          .format(args.resume, checkpoint['epoch'], checkpoint['best_prec1']))
    FPAM_net.load_state_dict(checkpoint['audio_net_state_dict'])
    gcn_max_med_model.load_state_dict(checkpoint['gcn_max_med_model_state_dict'])
    FPAM_net = FPAM_net.cuda()
    gcn_max_med_model = gcn_max_med_model.cuda()
    FPAM_net.eval()
    gcn_max_med_model.eval()
    # added
    for param in FPAM_net.parameters():  # freeze netT
        param.requires_grad = False
    for param in gcn_max_med_model.parameters():  # freeze netT
        param.requires_grad = False

    img_file_lst = ['val/corridor/Places365_val_00034934.jpg', 'val/corridor/Places365_val_00034820.jpg',
                    'val/corridor/Places365_val_00028101.jpg',
                    'val/bedroom/Places365_val_00015345.jpg', 'val/bedroom/Places365_val_00001822.jpg',
                    'val/bedroom/Places365_val_00009488.jpg',
                    'val/bedroom/Places365_val_00020357.jpg',
                    'val/office/Places365_val_00001805.jpg', 'val/office/Places365_val_00004918.jpg',
                    'val/office/Places365_val_00008146.jpg',
                    'val/dining_room/Places365_val_00017334.jpg', 'val/dining_room/Places365_val_00011164.jpg',
                    'val/dining_room/Places365_val_00000611.jpg',
                    'val/living_room/Places365_val_00035644.jpg', 'val/living_room/Places365_val_00008022.jpg',
                    'val/living_room/Places365_val_00018413.jpg',
                    'val/kitchen/Places365_val_00035756.jpg', 'val/kitchen/Places365_val_00007731.jpg',
                    'val/kitchen/Places365_val_00004746.jpg',
                    'val/bathroom/Places365_val_00029507.jpg', 'val/bathroom/Places365_val_00019364.jpg',
                    'val/bathroom/Places365_val_00028614.jpg']

    # img_file_lst = ['val/corridor/Places365_val_00034934.jpg', 'val/corridor/Places365_val_00034934.jpg']
    with torch.no_grad():
        for i in range(len(img_file_lst)):
            img_path = os.path.join(data_dir, img_file_lst[i])
            print('img_path:', img_path)
            img = Image.open(img_path)

            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            img_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

            img = img_transforms(img)
            print(img.size())
            img = img.unsqueeze(dim=0)
            img = img.cuda()
            print(img.size())

            img_output, resnet_output = FPAM_net(img)

            fpam_flops, fpam_params = thop_params_calculation(FPAM_net, img)
            print('fpam_flops:', fpam_flops, 'fpam_params:', fpam_params)
            print('img_output.shape:', img_output.shape, 'resnet_output.shape:', resnet_output.shape)

            gcn_output, rows, columns = gcn_max_med_model(img_output, resnet_output)

            # gcn_flosps, gcn_params = thop_params_calculation(gcn_max_med_model, img_output, resnet_output)

            from thop import profile
            gcn_flosps, gcn_params = profile(gcn_max_med_model, inputs=(img_output, resnet_output,))
            print('gcn_flosps:', gcn_flosps, 'gcn_params:', gcn_params)
            total_flops = fpam_flops + gcn_flosps
            total_params = fpam_params + gcn_params
            from thop import clever_format
            total_flops, total_params = clever_format([total_flops, total_params], "%.3f")
            print('total_flosps:', total_flops, 'total_params:', total_params)

            print('gcn_output.shape:', gcn_output.shape, gcn_output)

            rows = rows.squeeze()
            columns = columns.squeeze()
            print('rows:', rows.shape, len(rows))
            print('columns:', columns.shape, len(columns))

            save_path = os.path.join('./results', img_file_lst[i].split('/')[2])
            image_drawgraph(img_path, save_path, rows, columns)

            # Fig_name = os.path.join('STG', label_lst[i])
            # drawgraph(sound_path, Fig_name, rows, columns, feature_width=26, feature_height=8)
            # params = list(gcn_max_med_model.parameters())
            # weight_softmax = params[-2].cuda().data.cpu().numpy()
            # weight_softmax[weight_softmax < 0] = 0
            # print('weight size:', weight_softmax[0].shape)
            # CAMs = returnCAM(img_output[0], weight_softmax)
            # heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (26, 8)), cv2.COLORMAP_JET)
            # cv2.imwrite('./results/' + label_lst[i] + 'heatimg.png', heatmap)


def main(args, config):
    global best_prec1, discriminative_matrix
    print(args)

    audio_datasets = ['ESC10', 'ESC50', 'DCASE2019', 'US8K', 'DCASE2021']
    audio_visual_datasets = ['DCASE2021-Audio-Visual', 'ADVANCE']
    visual_datasets = ['Places365-7', 'Places365-14', 'MIT67', 'SUN_RGBD', 'SUN397', 'NYUData']

    # dataset selection for training
    dataset_selection = DatasetSelection()

    if args.dataset_name in audio_datasets:
        print('args.dataset_name:', args.dataset_name)
        data_dir, data_sample = dataset_selection.datasetSelection(args)

    elif args.dataset_name in audio_visual_datasets:
        print('args.dataset_name:', args.dataset_name)
        data_dir, data_sample = dataset_selection.datasetSelection(args)
        
    elif args.dataset_name == 'DCASE2021-Visual' or args.dataset_name == 'ADVANCE-Visual':
        print('args.dataset_name:', args.dataset_name)
        data_dir, data_sample = dataset_selection.datasetSelection(args)
        
    else:
        print('args.dataset_name:', args.dataset_name)
        data_dir = dataset_selection.datasetSelection(args)

    # traindir = os.path.join(data_dir, 'train')
    # valdir = os.path.join(data_dir, 'val')

    if args.dataset_name != 'DCASE2021-Visual' or args.dataset_name != 'ADVANCE-Visual':
        traindir = os.path.join(data_dir, 'train')
        valdir = os.path.join(data_dir, 'val')
        
    # Attention Model Initialization
    if (args.atten_type == 'baseline'):
        print('Backbone_ESC')
        FPAM_net = Backbone_ESC(args)
        
    elif (args.atten_type == 'fpam'):
        if args.dataset_name in audio_visual_datasets:
            audio_FPAM_net = FPAM_Audio_Visual(args, modality_type='audio')
            visual_FPAM_net = FPAM_Audio_Visual(args, modality_type='visual')
            FPAM_Fusion_net = FPAM_Fusion_Net(args)

            for param in audio_FPAM_net.parameters():
                param.requires_grad = True
            for param in visual_FPAM_net.parameters():
                param.requires_grad = True
            for param in FPAM_Fusion_net.parameters():
                param.requires_grad = True

            audio_FPAM_net = audio_FPAM_net.cuda()
            visual_FPAM_net = visual_FPAM_net.cuda()
            FPAM_Fusion_net = FPAM_Fusion_net.cuda()

        elif args.dataset_name in audio_datasets:
            FPAM_net = FPAM_Audio_Visual(args, modality_type='audio')
            for param in FPAM_net.parameters():
                param.requires_grad = True
            if args.status == 'train':
                FPAM_net = FPAM_net.cuda()

        # visual scene recognition
        else:
            FPAM_net = FPAM_Audio_Visual(args, modality_type='visual')
            for param in FPAM_net.parameters():
                param.requires_grad = True
            if args.status == 'train':
                FPAM_net = FPAM_net.cuda()



    cudnn.benchmark = True

    # create dataloader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    #'ESC10', 'ESC50', 'DCASE2019', 'US8K', 'DCASE2021'
    if args.dataset_name in audio_datasets:
        train_dataset = CVS_Audio(args, data_dir, data_sample, data_type='train')
        
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True,
                                  drop_last=True)

        val_dataset = CVS_Audio(args, data_dir, data_sample, data_type='test')
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
        
    # 'DCASE2021-Audio-Visual', 'ADVANCE'
    elif args.dataset_name in audio_visual_datasets:

        if args.dataset_name == 'ADVANCE':
            (train_sample, train_label, val_sample, val_label, test_sample, test_label) = data_sample

            train_dataset = CVSDataset(data_dir, train_sample, train_label,
                                       event_label_name='event_label_bayes')
            val_dataset = CVSDataset(data_dir, val_sample, val_label,
                                     event_label_name='event_label_bayes')
            test_dataset = CVSDataset(data_dir, test_sample, test_label,
                                      event_label_name='event_label_bayes')

            train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.workers)
            val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.workers)
            test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers)

        elif args.dataset_name == 'DCASE2021-Audio-Visual':
            train_dataset = CVS_Audio_Visual(args, data_dir, data_sample, data_type='train')
            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.workers, pin_memory=True,
                                      drop_last=True)

            val_dataset = CVS_Audio_Visual(args, data_dir, data_sample, data_type='test')
            val_loader = DataLoader(dataset=val_dataset,
                                    batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.workers, pin_memory=True)

    elif args.dataset_name == 'DCASE2021-Visual':

        train_dataset = CVS_Visual(args, data_dir, data_sample, data_type='train')
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True,
                                  drop_last=True)

        val_dataset = CVS_Visual(args, data_dir, data_sample, data_type='test')
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
    
    elif args.dataset_name == 'ADVANCE-Visual':
        (train_sample, train_label, val_sample, val_label, test_sample, test_label) = data_sample

        train_dataset = Visual_Dataset(data_dir, train_sample, train_label,
                                   event_label_name='event_label_bayes')
        val_dataset = Visual_Dataset(data_dir, val_sample, val_label,
                                 event_label_name='event_label_bayes')
        test_dataset = Visual_Dataset(data_dir, test_sample, test_label,
                                  event_label_name='event_label_bayes')

        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers)


    else:

        train_dataset = ImageFolderWithPaths(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

        val_dataset = ImageFolderWithPaths(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True,
            drop_last=True)

        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        if args.dataset_name == 'Places365-7':
            valdir = '/data/dataset/SUNRGBD/val/'
            SUN_RGBD_val_dataset = ImageFolderWithPaths(valdir,
                                                        transforms.Compose([
                                                            transforms.Resize(256),
                                                            transforms.CenterCrop(224),
                                                            transforms.ToTensor(),
                                                            normalize,
                                                        ]))

            SUN_RGBD_val_loader = torch.utils.data.DataLoader(
                dataset=SUN_RGBD_val_dataset,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True,
                drop_last=True)

    # define loss function (criterion) and pptimizer
    # mixup
    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy().cuda()
    # label smoothing
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING).cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda()

    print('criterion:', criterion)

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        # mixup_fn = Mixup(
        #     mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
        #     prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
        #     label_smoothing=config.MODEL  .LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=args.num_classes)

    # optimizer
    if args.dataset_name in audio_visual_datasets:
        FPAM_Fusion_net_params = list(FPAM_Fusion_net.parameters())
        audio_FPAM_net_params = list(audio_FPAM_net.parameters())
        visual_FPAM_net_params = list(visual_FPAM_net.parameters())

        optimizer = torch.optim.SGD([{'params': FPAM_Fusion_net_params},
                                     {'params':audio_FPAM_net_params},
                                     {'params':visual_FPAM_net_params}],
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    else:
        FPAM_module_params = list(FPAM_net.parameters())

        optimizer = torch.optim.SGD([{'params': FPAM_module_params}],
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    if config.AMP_OPT_LEVEL != "O0":
        if args.dataset_name in audio_visual_datasets:
            [audio_FPAM_net, visual_FPAM_net, FPAM_Fusion_net], optimizer = amp.initialize([audio_FPAM_net, visual_FPAM_net, FPAM_Fusion_net],
                                                   optimizer, opt_level=config.AMP_OPT_LEVEL)
        else:
            [FPAM_net], optimizer = amp.initialize([FPAM_net],
                                                    optimizer, opt_level=config.AMP_OPT_LEVEL)

    # distributed training
    # FPAM_net = torch.nn.parallel.DistributedDataParallel(FPAM_net, device_ids=[config.LOCAL_RANK],
    #                                                      broadcast_buffers=False, find_unused_parameters=True)
    #
    # gcn_max_med_model = torch.nn.parallel.DistributedDataParallel(gcn_max_med_model, device_ids=[config.LOCAL_RANK],
    #                                                               broadcast_buffers=False, find_unused_parameters=True)
    #
    # FPAM_net_without_ddp = FPAM_net.module
    # gcn_max_med_model_without_ddp = gcn_max_med_model.module

    # learning rate schedule
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))

    if args.status == 'test':
        # added
        # for param in FPAM_net.parameters():  # freeze netT
        #     param.requires_grad = False
        # for param in gcn_max_med_model.parameters():  # freeze netT
        #     param.requires_grad = False

        best_model_name = os.path.join('./weights', args.dataset_name, args.model_name)

        # load the pretrained-models
        print("=> loading checkpoint '{}'".format(best_model_name))
        checkpoint = torch.load(best_model_name)
        print("=> loaded checkpoint '{}' (epoch {}) (accuracy {})"
              .format(args.resume, checkpoint['epoch'], checkpoint['best_prec1']))

        # load the weights for model
        audio_net_state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['audio_net_state_dict'].items()}
        gcn_max_med_model_state_dict = {str.replace(k, 'module.', ''): v for k, v in
                                        checkpoint['gcn_max_med_model_state_dict'].items()}

        FPAM_net.load_state_dict(audio_net_state_dict)
        FPAM_net = FPAM_net.cuda()

        # FPAM_net.load_state_dict(checkpoint['audio_net_state_dict'])
        # gcn_max_med_model.load_state_dict(checkpoint['gcn_max_med_model_state_dict'])

        # evaluate on validation set
        test_losses, test_acc1, test_acc5, confusion_matrix = validate(args, FPAM_net, val_loader,
                                                                       criterion, 0)
        
        print('test_losses:', test_losses)
        print('test_acc1:', test_acc1)
        print('test_acc5:', test_acc5)

    if args.status == 'draw_image':
        draw_image(args, FPAM_net)

    elif args.status == 'draw_audio':
        draw_audio(args, FPAM_net)

    elif args.status == 'draw_atten':
        FPAM_net = FPAM_net.cuda()
        draw_atten(args, FPAM_net)

    elif args.status == 'train':
        accuracies_list = []
        max_accuracy = 0.0
        max_test_accuracy = 0.0

        conf_mat_lst = []

        # writer
        writer = SummaryWriter(comment=args.model_type+'_'+args.dataset_name)

        # from 1 to 60 epochs
        for epoch in range(args.start_epoch, args.epochs + 1):  # epoch from 1 to epochs

            # if epoch != 0 and epoch % 10 == 0:
            #     print("=> loading checkpoint '{}'".format(best_model_name))
            #     checkpoint = torch.load(best_model_name)
            #
            #     FPAM_net.load_state_dict(checkpoint['audio_net_state_dict'])
            #     gcn_max_med_model.load_state_dict(checkpoint['gcn_max_med_model_state_dict'])
            #     print("=> loaded checkpoint '{}' (epoch {})"
            #           .format(args.resume, checkpoint['epoch']))


            if args.dataset_name in audio_visual_datasets:
                # train for one epoch
                print('training')
                training_losses = train_one_epoch_audio_visual(args, FPAM_Fusion_net, audio_FPAM_net, visual_FPAM_net,
                                                               train_loader, optimizer, criterion, epoch, mixup_fn, lr_scheduler)
            else:
                # train for one epoch
                print('training')
                training_losses = train_one_epoch(args, FPAM_net,
                                                  train_loader, optimizer, criterion, epoch, mixup_fn, lr_scheduler)

            logger.info(f"Training loss of the network on the {len(train_loader)}:  {training_losses:.6f}")

            # evaluate on validation set
            if args.dataset_name in audio_visual_datasets:
                print('evaluation on validation set')
                val_losses, val_acc1, val_acc5, conf_mat = validate_audio_visual(args, FPAM_Fusion_net, audio_FPAM_net, visual_FPAM_net, val_loader, epoch)

            else:
                print('evaluation on validation set')
                val_losses, val_acc1, val_acc5, conf_mat = validate(args, FPAM_net, val_loader, epoch)
                if args.dataset_name in visual_datasets:
                    test_acc1 = val_acc1
                elif args.dataset_name == 'DCASE2021':
                    test_acc1 = val_acc1

            # remember best prec@1 and save checkpoint
            logger.info(f"Accuracy of the network on the {len(val_loader)} val images and val loss: {val_acc1:.2f}%  and {val_losses:.6f}")
            max_accuracy = max(max_accuracy, val_acc1)
            logger.info(f'Max val accuracy: {max_accuracy:.2f}%')
            best_prec1 = max_accuracy
            is_best = val_acc1 > best_prec1

            # evaluate on test set of advance
            if args.dataset_name == 'ADVANCE':
                print('evaluation on test set')
                test_losses, test_acc1, test_acc5, conf_mat = validate_audio_visual(args, FPAM_Fusion_net, audio_FPAM_net, visual_FPAM_net, test_loader, epoch)
                logger.info(
                    f"Accuracy of the network on the {len(test_loader)} test images and test loss: {test_acc1:.2f}% and {test_losses:.6f}")
                max_test_accuracy = max(max_test_accuracy, test_acc1)
                logger.info(f'Max test accuracy: {max_test_accuracy:.2f}%')
            
            elif args.dataset_name == 'ADVANCE-Visual':
                print('evaluation on test set')
                test_losses, test_acc1, test_acc5, conf_mat = validate(args, 
                                                                       FPAM_net,
                                                                       test_loader,
                                                                       epoch)
                logger.info(
                    f"Accuracy of the network on the {len(test_loader)} test images and test loss: {test_acc1:.2f}% and {test_losses:.6f}")
                max_test_accuracy = max(max_test_accuracy, test_acc1)
                logger.info(f'Max test accuracy: {max_test_accuracy:.2f}%')


            conf_mat_lst.append(conf_mat)
            print('confusion_matrix:', conf_mat.shape)
            print(conf_mat)


            if args.dataset_name == 'ESC10' or 'ESC50':
                conf_mat_filename = args.dataset_name + '_' + str(args.test_set_id) + '_conf_mat.npy'

            elif args.dataset_name == 'Places365-7' or 'Places365-14':
                conf_mat_filename = args.dataset_name + '_conf_mat.npy'

            print('conf_mat_filename:', conf_mat_filename)
            conf_mat_filename = os.path.join('logs', conf_mat_filename)
            np.save(conf_mat_filename, conf_mat)

            # remember best prec@1 and save checkpoint
            last_model_path, best_model_path = model_path_config(args, epoch)

            # TensorboardX writer
            lr = optimizer.param_groups[0]['lr']
            logger.info(f"LR of the network on the Epoch {epoch} is {lr:.9f}")
            writer.add_scalar('LR/Train', lr, epoch)

            if args.dataset_name == 'ADVANCE':
                writer.add_scalar('Acc1/Test', test_acc1, epoch)
                writer.add_scalar('Acc5/Test', test_acc5, epoch)
                writer.add_scalar('Loss/Test', test_losses, epoch)

            writer.add_scalar('Acc1/Train', val_acc1, epoch)
            writer.add_scalar('Acc5/Train', val_acc5, epoch)

            writer.add_scalar('Loss/Train', val_losses, epoch)

            if (epoch % 20 == 0):
                # save checkpoints
                if args.dataset_name in audio_visual_datasets:
                    if args.dataset_name == 'DCASE2021-Audio-Visual':
                        test_acc1 = val_acc1

                    accuracies_list.append("%.2f" % test_acc1.tolist())

                    save_checkpoint({
                        'epoch': epoch,
                        'arch': args.arch,
                        'test_acc1': test_acc1,
                        'best_prec1': best_prec1,
                        'audio_net_state_dict': FPAM_Fusion_net.state_dict(),
                        'optimizer': optimizer,
                    }, is_best, best_model_path, last_model_path)

                else:
                    save_checkpoint({
                        'epoch': epoch,
                        'arch': args.arch,
                        'test_acc1': test_acc1,
                        'best_prec1': best_prec1,
                        'audio_net_state_dict': FPAM_net.state_dict(),
                        'optimizer': optimizer,
                    }, is_best, best_model_path, last_model_path)

                    accuracies_list.append("%.2f" % test_acc1.tolist())


        # save args.dataset confusion_matrix
        conf_mat_file_name = args.dataset_name + '_' + str(args.test_set_id) + '_conf_mat_lst.npy'
        conf_mat_file_name = os.path.join('logs', conf_mat_file_name)
        conf_mat_array = np.array(conf_mat_lst)
        np.save(conf_mat_file_name, conf_mat_array)

def train_one_epoch_audio_visual(args, FPAM_Fusion_net, audio_FPAM_net, visual_FPAM_net, data_loader, optimizer, criterion, epoch, mixup_fn, lr_scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    norm_meter = AverageMeter()
    num_steps = len(data_loader)

    # record important metrics during training
    losses = AverageMeter()

    # switch to train mode
    audio_FPAM_net.train()
    visual_FPAM_net.train()
    FPAM_Fusion_net.train()
    optimizer.zero_grad()

    start = time.time()
    for idx, (audio, visual, labels, paths) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - start)
        # lr = lr_scheduler.get_lr()
        lr = optimizer.param_groups[0]['lr']
        # print('optimizer:', optimizer.param_groups[0]['lr'])

        # data to cuda
        visual = torch.autograd.Variable(visual).cuda()
        audio = torch.autograd.Variable(audio).cuda()
        labels = labels.cuda()

        # mixup training
        if mixup_fn is not None:
            visual, labels = mixup_fn(visual, labels)

        if (args.atten_type == 'baseline' or args.atten_type == 'fpam'):
            if args.atten_type == 'fpam':
                audio_fpam_out, _ = audio_FPAM_net(audio)
                # print('audio_fpam_out:', audio_fpam_out.shape)
                visual_fpam_out, _ = visual_FPAM_net(visual)
                # print('visual_fpam_out:', visual_fpam_out.shape)

                fpam_output = FPAM_Fusion_net(audio_fpam_out, visual_fpam_out)

            else:
                fpam_output = FPAM_Fusion_net(visual)
                # fpam_flops, fpam_params = thop_params_calculation(AFM, images)
                # print('fpam_flops:', fpam_flops, 'fpam_params:', fpam_params)

            if config.TRAIN.ACCUMULATION_STEPS > 1:
                loss = criterion(fpam_output, labels)
                loss = loss / config.TRAIN.ACCUMULATION_STEPS

                # mixed precision training
                if config.AMP_OPT_LEVEL != "O0":
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(amp.master_params(optimizer))
                else:
                    loss.backward()
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(AFM.parameters(), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(AFM.parameters())

                if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step_update(epoch * num_steps + idx)
            else:
                loss = criterion(fpam_output, labels)
                optimizer.zero_grad()

                # mixed precision training
                if config.AMP_OPT_LEVEL != "O0":
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()

                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(amp.master_params(optimizer))
                else:
                    loss.backward()
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(AFM.parameters(), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(AFM.parameters())

                optimizer.step()
                lr_scheduler.step_update(epoch * num_steps + idx)

        losses.update(loss, visual.size(0))
        norm_meter.update(grad_norm)

        # measure elapsed time
        batch_time.update(time.time() - start)

        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Learning rate {lr:.4f}\t'.format(
                epoch, idx, len(data_loader), batch_time=batch_time,
                data_time=data_time, lr=lr, loss=losses))

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    return losses.avg



def train_one_epoch(args, AFM, data_loader, optimizer, criterion, epoch, mixup_fn, lr_scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    norm_meter = AverageMeter()
    num_steps = len(data_loader)

    # record important metrics during training
    losses = AverageMeter()

    # switch to train mode
    AFM.train()
    optimizer.zero_grad()

    start = time.time()
    for idx, (images, labels, paths) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - start)
        # lr = lr_scheduler.get_lr()
        lr = optimizer.param_groups[0]['lr']
        # print('optimizer:', optimizer.param_groups[0]['lr'])

        # data to cuda
        images = torch.autograd.Variable(images).cuda()
        labels = labels.cuda()

        # mixup training
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        if (args.atten_type == 'baseline' or args.atten_type == 'fpam'):
            if args.atten_type == 'fpam':
                fpam_output, _ = AFM(images)
            else:
                fpam_output = AFM(images)
                # fpam_flops, fpam_params = thop_params_calculation(AFM, images)
                # print('fpam_flops:', fpam_flops, 'fpam_params:', fpam_params)

            if config.TRAIN.ACCUMULATION_STEPS > 1:
                loss = criterion(fpam_output, labels)
                loss = loss / config.TRAIN.ACCUMULATION_STEPS

                # mixed precision training
                if config.AMP_OPT_LEVEL != "O0":
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(amp.master_params(optimizer))
                else:
                    loss.backward()
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(AFM.parameters(), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(AFM.parameters())

                if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step_update(epoch * num_steps + idx)
            else:
                loss = criterion(fpam_output, labels)
                optimizer.zero_grad()

                # mixed precision training
                if config.AMP_OPT_LEVEL != "O0":
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()

                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(amp.master_params(optimizer))
                else:
                    loss.backward()
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(AFM.parameters(), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(AFM.parameters())

                optimizer.step()
                lr_scheduler.step_update(epoch * num_steps + idx)

        losses.update(loss, images.size(0))
        norm_meter.update(grad_norm)

        # measure elapsed time
        batch_time.update(time.time() - start)

        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Learning rate {lr:.4f}\t'.format(
                epoch, idx, len(data_loader), batch_time=batch_time,
                data_time=data_time, lr=lr, loss=losses))

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    return losses.avg

def validate_audio_visual(args, FPAM_Fusion_net, audio_FPAM_net, visual_FPAM_net, data_loader, epoch):


    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    # switch to evaluate mode
    criterion = torch.nn.CrossEntropyLoss()

    FPAM_Fusion_net.eval()
    audio_FPAM_net.eval()
    visual_FPAM_net.eval()
    
    # recall prediction and labels

    conf_matrix = torch.zeros(args.num_classes, args.num_classes)
    # print('conf_matrix:', conf_matrix.shape)

    end = time.time()
    with torch.no_grad():
        fpam_output_lst = []
        labels_lst = []
        all_pred = []
        all_true = []

        for i, (audio, images, labels, paths) in enumerate(data_loader):

            # data to cuda
            images = torch.autograd.Variable(images).cuda()
            audio = torch.autograd.Variable(audio).cuda()
            labels = labels.cuda()

            if (args.atten_type == 'baseline' or args.atten_type == 'fpam'):
                if args.atten_type == 'fpam':
                    audio_fpam_out, _ = audio_FPAM_net(audio)
                    visual_fpam_out, _ = visual_FPAM_net(images)
                    fpam_output = FPAM_Fusion_net(audio_fpam_out, visual_fpam_out)
                else:
                    fpam_output = AFM(images)

                # compute gradient and loss for SGD step
                loss = criterion(fpam_output, labels)
                
                
                # measure accuracy and record loss
                prec1, prec5 = accuracy(fpam_output, labels, topk=(1, 5))

                prediction = torch.max(fpam_output, 1)[1]
                # print('prediction:', prediction)
                # print('labels:', labels)

                conf_matrix = confusion_matrix_cal(args, prediction, labels=labels, conf_matrix=conf_matrix)

                all_pred.extend(prediction)
                all_true.extend(labels)

                labels_lst.append(labels)
                fpam_output_lst.append(fpam_output)

            losses.update(loss, images.size(0))
            top1.update(prec1, images.size(0))
            top5.update(prec5, images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(data_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        
        all_true = torch.tensor(all_true).cpu().data.numpy()
        all_pred = torch.tensor(all_pred).cpu().data.numpy()

        performance_matrix(all_true, all_pred)

        # fpam_outputs = fpam_output_lst[0]
        # targets = labels_lst[0]
        # for i in range(len(fpam_output_lst) - 1):
        #     fpam_outputs = torch.cat((fpam_outputs, fpam_output_lst[i + 1]), dim=0)
        #     targets = torch.cat((targets, labels_lst[i + 1]), dim=0)
            

        # fpam_outputs = fpam_outputs.cpu().data.numpy()
        # targets = targets.cpu().data.numpy()
        #
        # data_tsne = TSNE(n_components=2, random_state=33).fit_transform(fpam_outputs)
        # fig = plot_embedding(args,
        #                      data_tsne,
        #                     targets,
        #                      epoch,
        #                     't-SNE visualization of latent features (epoch {})'.format(epoch))
        # plt.close(fig)




    return losses.avg, top1.avg, top5.avg, conf_matrix.numpy()

def validate(args, FPAM, data_loader, epoch):


    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    criterion = torch.nn.CrossEntropyLoss()
    FPAM.eval()
    conf_matrix = torch.zeros(args.num_classes, args.num_classes)

    end = time.time()
    with torch.no_grad():
        fpam_output_lst = []
        labels_lst = []
        all_pred = []
        all_true = []
        
        for i, (images, labels, paths) in enumerate(data_loader):

            # data to cuda
            images = torch.autograd.Variable(images).cuda()
            labels = labels.cuda()

            if (args.atten_type == 'baseline' or args.atten_type == 'fpam'):
                if args.atten_type == 'fpam':
                    fpam_output, _ = FPAM(images)
                else:
                    fpam_output = FPAM(images)

                # compute gradient and loss for SGD step
                loss = criterion(fpam_output, labels)
                # measure accuracy and record loss
                prec1, prec5 = accuracy(fpam_output, labels, topk=(1, 5))

                prediction = torch.max(fpam_output, 1)[1]
                all_pred.extend(prediction)
                all_true.extend(labels)

                conf_matrix = confusion_matrix_cal(args, prediction, labels=labels, conf_matrix=conf_matrix)
                fpam_output_lst.append(fpam_output)
                labels_lst.append(labels)

            losses.update(loss, images.size(0))
            top1.update(prec1, images.size(0))
            top5.update(prec5, images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(data_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        
        all_true = torch.tensor(all_true).cpu().data.numpy()
        all_pred = torch.tensor(all_pred).cpu().data.numpy()

        performance_matrix(all_true, all_pred)

        # fpam_outputs = fpam_output_lst[0]
        # targets = labels_lst[0]
        # for i in range(len(fpam_output_lst) - 1):
        #     fpam_outputs = torch.cat((fpam_outputs, fpam_output_lst[i + 1]), dim=0)
        #     targets = torch.cat((targets, labels_lst[i + 1]), dim=0)
        #
        # fpam_outputs = fpam_outputs.cpu().data.numpy()
        # targets = targets.cpu().data.numpy()
        # print('fpam_outputs.shape:', fpam_outputs.shape)
        # print('targets.shape:', targets.shape)
        #
        # data_tsne = TSNE(n_components=2, random_state=33).fit_transform(fpam_outputs)
        # fig = plot_embedding(data_tsne,
        #                     targets,
        #                      epoch,
        #                     't-SNE visualization of latent features (epoch {})'.format(epoch))
        # plt.close(fig)




    return losses.avg, top1.avg, top5.avg, conf_matrix.numpy()


def plot_embedding(args, data, label, epoch, title='t-SNE visualization of latent features'):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.scatter(data[i, 0], data[i, 1],
                 color=plt.cm.Set1(label[i] / 10.))

        # plt.text(data[i, 0], data[i, 1], str(label[i]),
        #          color=plt.cm.Set1(label[i] / 10.),
        #          fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    # plt.legend()
    plt.title(title)
    tSNE_filename = os.path.join('images', args.dataset_name, 'visual_tsne_{}.png'.format(epoch))
    plt.savefig(tSNE_filename, dpi=150)

    return fig

def save_checkpoint(state, is_best, best_model_name, latest_model_name):
    torch.save(state, latest_model_name)
    if is_best:
        shutil.copyfile(latest_model_name, best_model_name)


def confusion_matrix_cal(args, preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        if (p >= args.num_classes or t >= args.num_classes):
            print('p:', p, 't:', t)
            continue
        else:
            conf_matrix[t, p] += 1
    return conf_matrix


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


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lri))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def my_forward(model, x):
    mo = nn.Sequential(*list(model.children())[:-1])
    feature = mo(x)
    #    print(feature.size())
    feature = feature.view(x.size(0), -1)
    output = model.fc(feature)
    return feature


if __name__ == '__main__':
    args = arguments_parse.argsParser()
    print('args:', args)
    config = get_config(args)

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED  # + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0

    # linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0
    # linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0
    # linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0

    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS

    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    print('config.OUTPUT:', config.OUTPUT)
    if not os.path.isdir(config.OUTPUT):
        os.makedirs(config.OUTPUT, exist_ok=True)

    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.TYPE}")

    # if dist.get_rank() == 0:

    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(args, config)


