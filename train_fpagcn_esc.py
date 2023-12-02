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
from model.network import GCN_audio_top_med_fea, GCN_max_med_fusion, AGCN_max_med_fusion, GraphAttentionModule
from model.network import GCN_audio_top_med
from model.network import AFM_Net
from model.network import Backbone_ESC

from model.CVS_dataset import CVS_Audio, CVS_Visual, CVS_Audio_Visual
from model.CVS_dataset import sound_inference

from utils.arguments import arguments_parse
from utils.performance_metrics import obtain_inference_time_fps
from utils.performance_metrics import performance_matrix
from utils.performance_metrics import plot_precision_recall_curve
from utils.performance_metrics import thop_params_calculation
from utils.performance_metrics import confusion_matrix_cal
from utils.performance_metrics import accuracy

from data.dataset import ImageFolderWithPaths, DatasetSelection, ConcatDataset
from model.CVS_dataset import CVSDataset, Visual_Dataset, Audio_Dataset

import sys

# t-SNE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

sys.path.append("./resnet-audio/")
sys.path.append("./resnet_audio/")
sys.path.append("./utils/")

best_prec1 = 0

import librosa
import matplotlib.patches as patches
import numpy as np
import cv2
from PIL import Image

from config import get_config
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data import Mixup
from logger import create_logger
from lr_scheduler import build_scheduler

import torch.distributed as dist
import datetime
import timeit


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


def model_load_state_dict(args, model_type, model):
    model_state_dict = model.state_dict()
    new_state_dict = {}

    if model_type == 'audio':
        state = torch.load(args.audio_pretrain_weights)['audio_net_state_dict']
        print('Loading the pretrained Audio Backbone model from the weights {%s}!' % args.audio_pretrain_weights)  # AUDIOSET PRETRAINED MODEL

    elif model_type == 'visual':
        state = torch.load(args.visual_pretrain_weights)['audio_net_state_dict']
        print('Loading the pretrained Visual Backbone model from the weights {%s}!' % args.visual_pretrain_weights)  # AUDIOSET PRETRAINED MODEL
        
    elif model_type == 'audio_gcn':
        state = torch.load(args.audio_pretrain_weights)['gcn_max_med_model_state_dict']
        print('Loading the pretrained Audio GCN model from the weights {%s}!' % args.visual_pretrain_weights)  # AUDIOSET PRETRAINED MODEL

    elif model_type == 'visual_gcn':
        state = torch.load(args.visual_pretrain_weights)['gcn_max_med_model_state_dict']
        print('Loading the pretrained Visual GCN model from the weights {%s}!' % args.visual_pretrain_weights)  # AUDIOSET PRETRAINED MODEL
        

    for key, value in state.items():
        # print('key:', key)
        new_state_dict.update({key: value})

    # for key, value in model_state_dict.items():
    #     print('model key:', key)
  
    model.load_state_dict(model_state_dict)
    


def main(args, config):
    global best_prec1, discriminative_matrix
    print(args)

    audio_datasets = ['ESC10', 'ESC50', 'US8K', 'HomeAutomation', 'DCASE2019', 'DCASE2021']
    audio_visual_datasets = ['DCASE2021-Audio-Visual', 'ADVANCE']

    # dataset selection for training
    dataset_selection = DatasetSelection()

    if args.dataset_name in audio_datasets:
        data_dir, data_sample = dataset_selection.datasetSelection(args)

    elif args.dataset_name in audio_visual_datasets:
        data_dir, data_sample = dataset_selection.datasetSelection(args)

    elif args.dataset_name == 'DCASE2021-Visual' or args.dataset_name == 'ADVANCE-Visual' or args.dataset_name == 'ADVANCE-Audio':
        data_dir, data_sample = dataset_selection.datasetSelection(args)

    else:
        data_dir = dataset_selection.datasetSelection(args)

    print('data_dir:', data_dir)

    if args.dataset_name != 'DCASE2019' and args.dataset_name != 'DCASE2021' and args.dataset_name != 'DCASE2021-Visual' and args.dataset_name not in audio_visual_datasets and \
            args.dataset_name != 'ADVANCE-Audio' and args.dataset_name != 'ADVANCE-Visual':
        
        print('args.dataset_name:', args.dataset_name)
        print('data_dir:', data_dir)
        traindir = os.path.join(data_dir, 'train')
        valdir = os.path.join(data_dir, 'val')


    # Attention Model Initialization
    if args.dataset_name in audio_visual_datasets:
        Audio_FPAM_net = AFM_Net(args, modality_type='audio')
        Visual_FPAM_net = AFM_Net(args, modality_type='visual')

    elif args.dataset_name == 'SUN_RGBD_Full':
        Depth_FPAM_net = AFM_Net(args, modality_type='visual')
        Visual_FPAM_net = AFM_Net(args, modality_type='visual')
        Audio_FPAM_net = Depth_FPAM_net

    elif (args.atten_type == 'afm'):
        if args.dataset_name == 'Places365-7' or args.dataset_name == 'Places365-14' or args.dataset_name == 'SUN_RGBD' or args.dataset_name == 'NYUdata' or args.dataset_name == 'ADVANCE-Visual' or args.dataset_name == 'DCASE2021-Visual' or args.dataset_name == 'SUN_RGBD_Depth':
            FPAM_net = AFM_Net(args, modality_type='visual')
        else:
            FPAM_net = AFM_Net(args, modality_type='audio')
        
    elif (args.atten_type == 'baseline'):
        FPAM_net = Backbone_ESC(args, modality_type='audio')
        
    # single model
    if (args.fusion == False):
        gcn_max_med_model = GCN_audio_top_med(args)

    elif (args.dataset_name in audio_visual_datasets or args.dataset_name == 'SUN_RGBD_Full'):
        Audio_gcn_max_med_model = AGCN_max_med_fusion(args)
        Visual_gcn_max_med_model = AGCN_max_med_fusion(args)
        GAT_model = GraphAttentionModule(args)

        if args.dataset_name != 'SUN_RGBD_Full':
            model_load_state_dict(args, 'audio', Audio_FPAM_net)
            model_load_state_dict(args, 'visual', Visual_FPAM_net)
            model_load_state_dict(args, 'audio_gcn', Audio_gcn_max_med_model)
            model_load_state_dict(args, 'visual_gcn', Visual_gcn_max_med_model)

    # fusion model
    else:
        gcn_max_med_model = GCN_max_med_fusion(args)
        
        
    # for param in FPAM_net.parameters():
    #     print('param.requires_grad:', param.requires_grad)
    # for param in gcn_max_med_model.parameters():
    #     print('param.requires_grad:', param.requires_grad)

    # require gradient
    if args.dataset_name in audio_visual_datasets or args.dataset_name == 'SUN_RGBD_Full':
        for param in Audio_FPAM_net.parameters():
            param.requires_grad = True
        for param in Visual_FPAM_net.parameters():
            param.requires_grad = True
        for param in Audio_gcn_max_med_model.parameters():
            param.requires_grad = True
        for param in Visual_gcn_max_med_model.parameters():
            param.requires_grad = True
        for param in GAT_model.parameters():
            param.requires_grad = True

        Audio_FPAM_net = Audio_FPAM_net.cuda()
        Visual_FPAM_net = Visual_FPAM_net.cuda()
        Audio_gcn_max_med_model = Audio_gcn_max_med_model.cuda()
        Visual_gcn_max_med_model = Visual_gcn_max_med_model.cuda()
        GAT_model = GAT_model.cuda()

    else:
        for param in FPAM_net.parameters():
            param.requires_grad = True
        for param in gcn_max_med_model.parameters():
            param.requires_grad = True

        FPAM_net = FPAM_net.cuda()
        gcn_max_med_model = gcn_max_med_model.cuda()

    # params_calculation(FPAM_net, gcn_max_med_model)


    cudnn.benchmark = True
    # create dataloader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

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
            print('args.dataset_name:', args.dataset_name)
            train_dataset = CVS_Audio_Visual(args, data_dir, data_sample, data_type='train')
            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.workers, pin_memory=True,
                                      drop_last=True)

            val_dataset = CVS_Audio_Visual(args, data_dir, data_sample, data_type='test')
            val_loader = DataLoader(dataset=val_dataset,
                                    batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.workers, pin_memory=True)
            
    elif args.dataset_name == 'ADVANCE-Visual':
        print('ADVANCE-Visual args.dataset_name dataloader:', args.dataset_name)

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

    elif args.dataset_name == 'ADVANCE-Audio':
        print('ADVANCE-Audio args.dataset_name dataloader:', args.dataset_name)

        (train_sample, train_label, val_sample, val_label, test_sample, test_label) = data_sample

        train_dataset = Audio_Dataset(data_dir, train_sample, train_label,
                                      event_label_name='event_label_bayes')
        val_dataset = Audio_Dataset(data_dir, val_sample, val_label,
                                    event_label_name='event_label_bayes')
        test_dataset = Audio_Dataset(data_dir, test_sample, test_label,
                                     event_label_name='event_label_bayes')

        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers)

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


    elif args.dataset_name == 'SUN_RGBD_Full':
        train_dataset = ImageFolderWithPaths(traindir,
            transforms.Compose
            ([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        )

        depth_train_dataset = ImageFolderWithPaths('/lintianlin_group_v100/lgzhou/dataset/SUN_RGBD_Full/depth',
                        transforms.Compose
                            ([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                        ])
        )

        train_dataset = ConcatDataset(train_dataset, depth_train_dataset)


        val_dataset = ImageFolderWithPaths(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
        
        depth_val_dataset = ImageFolderWithPaths('/lintianlin_group_v100/lgzhou/dataset/SUN_RGBD_Full/depth/val', transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
        
        val_dataset = ConcatDataset(val_dataset, depth_val_dataset)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True,
            drop_last=True)

        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    # visual dataset
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
    if args.dataset_name in audio_visual_datasets or args.dataset_name == 'SUN_RGBD_Full':
        Audio_FPAM_module_params = list(Audio_FPAM_net.parameters())
        Visual_FPAM_module_params = list(Visual_FPAM_net.parameters())

        Audio_gcn_max_med_module_params = list(Audio_gcn_max_med_model.parameters())
        Visual_gcn_max_med_module_params = list(Visual_gcn_max_med_model.parameters())

        GAT_module_params = list(GAT_model.parameters())


        print('args.lr:', args.lr)
        optimizer = torch.optim.SGD([{'params': Audio_FPAM_module_params},
                                     {'params': Visual_FPAM_module_params},
                                     {'params': Audio_gcn_max_med_module_params},
                                     {'params': Visual_gcn_max_med_module_params},
                                     {'params': GAT_module_params}],
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        if config.AMP_OPT_LEVEL != "O0":
            [Audio_FPAM_net, Visual_FPAM_net, Audio_gcn_max_med_model, Visual_gcn_max_med_model, GAT_model], optimizer = amp.initialize([Audio_FPAM_net, Visual_FPAM_net, Audio_gcn_max_med_model, Visual_gcn_max_med_model, GAT_model],
                                                                                                                             optimizer, opt_level=config.AMP_OPT_LEVEL)

    else:

        FPAM_module_params = list(FPAM_net.parameters())
        gcn_max_med_module_params = list(gcn_max_med_model.parameters())

        print('args.lr:', args.lr)
        optimizer = torch.optim.SGD([{'params': FPAM_module_params},
                                     {'params': gcn_max_med_module_params}],
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        if config.AMP_OPT_LEVEL != "O0":
            [FPAM_net, gcn_max_med_model], optimizer = amp.initialize([FPAM_net, gcn_max_med_model],
                                                                      optimizer, opt_level=config.AMP_OPT_LEVEL)

    # distributed training
    if args.dataset_name in audio_visual_datasets or args.dataset_name == 'SUN_RGBD_Full':
        Audio_FPAM_net = torch.nn.parallel.DistributedDataParallel(Audio_FPAM_net,
                                                                   device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=True)
        Visual_FPAM_net = torch.nn.parallel.DistributedDataParallel(Visual_FPAM_net,
                                                                   device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=True)

        Audio_gcn_max_med_model = torch.nn.parallel.DistributedDataParallel(Audio_gcn_max_med_model,
                                                                      device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=True)

        Visual_gcn_max_med_model = torch.nn.parallel.DistributedDataParallel(Visual_gcn_max_med_model,
                                                                      device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=True)

        GAT_model = torch.nn.parallel.DistributedDataParallel(GAT_model,
                                                              device_ids=[config.LOCAL_RANK],
                                                              broadcast_buffers=False,
                                                              find_unused_parameters=True)

    else:
        FPAM_net = torch.nn.parallel.DistributedDataParallel(FPAM_net, device_ids=[config.LOCAL_RANK],
                                                             broadcast_buffers=False, find_unused_parameters=True)

        gcn_max_med_model = torch.nn.parallel.DistributedDataParallel(gcn_max_med_model, device_ids=[config.LOCAL_RANK],
                                                                      broadcast_buffers=False, find_unused_parameters=True)

    # distributed GPUs training
    # for model saving
    if args.dataset_name in audio_visual_datasets or args.dataset_name == 'SUN_RGBD_Full':
        Audio_FPAM_net_without_ddp = Audio_FPAM_net.module
        Visual_FPAM_net_without_ddp = Visual_FPAM_net.module
        Audio_gcn_max_med_model_without_ddp = Audio_gcn_max_med_model.module
        Visual_gcn_max_med_model_without_ddp = Visual_gcn_max_med_model.module
        GAT_model_without_ddp = GAT_model.module

    else:
        FPAM_net_without_ddp = FPAM_net.module
        gcn_max_med_model_without_ddp = gcn_max_med_model.module

    # learning rate schedule
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))

    if args.status == 'test':
        # added
        # for param in FPAM_net.parameters():  # freeze netT
        #     param.requires_grad = False
        # for param in gcn_max_med_model.parameters():  # freeze netT
        #     param.requires_grad = False

        if (args.dataset_name == 'SUNRGBD'):
            dataset_name = 'Places365-7'
        else:
            dataset_name = args.dataset_name

        best_model_name = os.path.join('./weights', dataset_name, args.model_name)

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
        gcn_max_med_model.load_state_dict(gcn_max_med_model_state_dict)

        FPAM_net = FPAM_net.cuda()
        gcn_max_med_model = gcn_max_med_model.cuda()

        # FPAM_net.load_state_dict(checkpoint['audio_net_state_dict'])
        # gcn_max_med_model.load_state_dict(checkpoint['gcn_max_med_model_state_dict'])

        # evaluate on validation set
        test_losses, test_acc1, test_acc5, confusion_matrix = validate(args, FPAM_net, gcn_max_med_model, val_loader,
                                                                       criterion, 0)
        print('test_losses:', test_losses)
        print('test_acc1:', test_acc1)
        print('test_acc5:', test_acc5)

    if args.status == 'draw_image':
        draw_image(args, FPAM_net, gcn_max_med_model)

    elif args.status == 'draw_audio':
        draw_audio(args, FPAM_net, gcn_max_med_model)

    elif args.status == 'train':
        accuracies_list = []
        max_accuracy = 0.0
        max_val_f1_score = 0.0

        places_confusion_matrix_lst = []
        sun_rgbd_confusion_matrix_lst = []
        # writer
        writer = SummaryWriter(comment=args.model_type+'_'+args.dataset_name)

        # from 0 to 300 epoches
        # for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        for epoch in range(args.start_epoch, args.epochs + 1):  # epoch from 1 to epochs

            # if epoch != 0 and epoch % 10 == 0:
            #     print("=> loading checkpoint '{}'".format(best_model_name))
            #     checkpoint = torch.load(best_model_name)
            #
            #     FPAM_net.load_state_dict(checkpoint['audio_net_state_dict'])
            #     gcn_max_med_model.load_state_dict(checkpoint['gcn_max_med_model_state_dict'])
            #     print("=> loaded checkpoint '{}' (epoch {})"
            #           .format(args.resume, checkpoint['epoch']))

            # train for one epoch
            if args.dataset_name in audio_visual_datasets or args.dataset_name == 'SUN_RGBD_Full':
                print('training')
                train_losses, train_acc1, train_acc5 = train_one_epoch_audio_visual(args,
                                                                                    Audio_FPAM_net, Visual_FPAM_net,
                                                                                    Audio_gcn_max_med_model, Visual_gcn_max_med_model, GAT_model,
                                                                                    train_loader, optimizer, criterion, epoch, mixup_fn, lr_scheduler)


                print('evaluation on validation set')
                test_losses, val_acc1, val_acc5, places_confusion_matrix, f1_score = validate_audio_visual(args,
                                                                                                           Audio_FPAM_net, Visual_FPAM_net,
                                                                                                           Audio_gcn_max_med_model, Visual_gcn_max_med_model, GAT_model,
                                                                                                           val_loader, epoch)

                max_val_f1_score = max(max_val_f1_score, f1_score)


                if args.dataset_name == 'ADVANCE':
                    # evaluate on validation set
                    print('evaluation on test set')
                    test_losses, test_acc1, test_acc5, conf_mat, f1_score = validate_audio_visual(args,
                                                                                                  Audio_FPAM_net, Visual_FPAM_net,
                                                                                                  Audio_gcn_max_med_model, Visual_gcn_max_med_model, GAT_model,
                                                                                                  test_loader, epoch)
                else:
                    test_acc1 = val_acc1
                    test_acc5 = val_acc5



            elif args.dataset_name == 'ADVANCE-Audio' or args.dataset_name == 'ADVANCE-Visual':
                
                print('training')
                train_losses, train_acc1, train_acc5 = train_one_epoch(args, FPAM_net, gcn_max_med_model,
                                                                       train_loader, optimizer, criterion, epoch,
                                                                       mixup_fn, lr_scheduler)

                print('evaluation on validation set')
                val_losses, val_acc1, val_acc5, places_confusion_matrix, f1_score = validate(args,
                                                                                             FPAM_net, 
                                                                                             gcn_max_med_model, 
                                                                                             val_loader,
                                                                                             criterion,
                                                                                             epoch)
                max_val_f1_score = max(max_val_f1_score, f1_score)

                
                print('evaluation on test set')
                test_losses, test_acc1, test_acc5, places_confusion_matrix, f1_score = validate(args, 
                                                                                                FPAM_net,
                                                                                                gcn_max_med_model,
                                                                                                test_loader, 
                                                                                                criterion,
                                                                                                epoch)


            else:
                print('training')
                train_losses, train_acc1, train_acc5 = train_one_epoch(args, FPAM_net, gcn_max_med_model,
                                                                       train_loader, optimizer, criterion, epoch, mixup_fn, lr_scheduler)
                print('testing')
                test_losses, test_acc1, test_acc5, places_confusion_matrix, f1_score = validate(args, FPAM_net, gcn_max_med_model,
                                                                                                val_loader, criterion, epoch)
                
                val_acc1 = test_acc1


            # evaluate on validation set


            logger.info(f"Accuracy of the network on the {len(val_loader)} test images: {test_acc1:.1f}%")
            max_accuracy = max(max_accuracy, val_acc1)
            logger.info(f'Max val accuracy: {max_accuracy:.2f}%')
            logger.info(f'Max val f1_score: {max_val_f1_score:.3f}%')

            places_confusion_matrix_lst.append(places_confusion_matrix)
            print('dataset:', args.dataset_name)
            print('confusion_matrix:', places_confusion_matrix.shape)
            print(places_confusion_matrix)

            if args.dataset_name == 'Places365-7' or 'Places365-14':
                places_name = args.dataset_name + str(args.nodes_num) + '_confusion.npy'

            elif args.dataset_name == 'ESC10' or 'ESC50':
                places_name = args.dataset_name + '_' + str(args.test_set_id) + '_' + str(
                    args.nodes_num) + '_confusion.npy'

            print('places_name:', places_name)
            np.save(places_name, places_confusion_matrix)

            # if args.dataset_name == 'Places365-7':
            #     print('Testing the SUN RGBD dataset!')
            #     test_losses, sun_rgbd_test_acc1, sun_rgbd_test_acc5, sun_rgbd_confusion_matrix, f1_score = validate(args,
            #                                                                                               FPAM_net,
            #                                                                                               gcn_max_med_model,
            #                                                                                               SUN_RGBD_val_loader,
            #                                                                                               criterion,
            #                                                                                               epoch)
            #     print('SUN_RGBD test_acc1:', sun_rgbd_test_acc1, 'SUN_RGBD test_acc5:', sun_rgbd_test_acc5)
            #     print('dataset: SUN_RGBD')
            #     print('sun_rgbd_confusion_matrix:', sun_rgbd_confusion_matrix.shape)
            #     print(sun_rgbd_confusion_matrix)
            #     sun_rgbd_confusion_matrix_lst.append(sun_rgbd_confusion_matrix)
            #     suns_name = 'suns_' + str(args.nodes_num) + '_confusion.npy'
            #     np.save(suns_name, sun_rgbd_confusion_matrix)

            # remember best prec@1 and save checkpoint
            last_model_path, best_model_path = model_path_config(args, epoch)
            is_best = test_acc1 > best_prec1
            best_prec1 = max(test_acc1, best_prec1)
            logger.info(f'Max accuracy: {best_prec1:.2f}%')

            # distributed GPUs training
            if (epoch % 20 == 0):
                # save checkpoints
                if args.dataset_name in audio_visual_datasets:
                    save_checkpoint({
                        'epoch': epoch,
                        'arch': args.arch,
                        'test_acc1': test_acc1,
                        'best_prec1': best_prec1,
                        'audio_net_state_dict': Audio_FPAM_net_without_ddp.state_dict(),
                        'audio_gcn_max_med_model_state_dict': Audio_gcn_max_med_model_without_ddp.state_dict(),
                        'visual_fpam_net_state_dict': Visual_FPAM_net_without_ddp.state_dict(),
                        'visual_gcn_max_med_model_state_dict': Visual_gcn_max_med_model_without_ddp.state_dict(),
                        'GAT_model_state_dict': GAT_model_without_ddp.state_dict(),
                        'optimizer': optimizer,
                    }, is_best, best_model_path, last_model_path)

                else:
                    save_checkpoint({
                        'epoch': epoch,
                        'arch': args.arch,
                        'test_acc1': test_acc1,
                        'best_prec1': best_prec1,
                        'audio_net_state_dict': FPAM_net_without_ddp.state_dict(),
                        'gcn_max_med_model_state_dict': gcn_max_med_model_without_ddp.state_dict(),
                        'optimizer': optimizer,
                    }, is_best, best_model_path, last_model_path)


            # TensorboardX writer
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('LR/Train', lr, epoch)
            writer.add_scalar('Acc1/Test', test_acc1, epoch)
            writer.add_scalar('Acc5/Test', test_acc5, epoch)
            
            writer.add_scalar('Acc1/Train', train_acc1, epoch)
            writer.add_scalar('Acc5/Train', train_acc5, epoch)

            writer.add_scalar('Loss/Train', train_losses, epoch)
            writer.add_scalar('Loss/Test', test_losses, epoch)

            accuracies_list.append("%.2f" % test_acc1.tolist())

        if args.dataset_name == 'Places365-7':
            places_name = args.dataset_name + str(args.nodes_num) + '_confusion.npy'
            # save sun_rgbd confusion_matrix
            suns_name = 'suns_' + str(args.nodes_num) + '_confusion.npy'
            np.save(suns_name, sun_rgbd_confusion_matrix_lst)

        elif args.dataset_name == 'Places365-14':
            places_name = args.dataset_name + str(args.nodes_num) + '_confusion.npy'

        elif args.dataset_name == 'ESC10' or 'ESC50':
            places_name = args.dataset_name + '_' + str(args.test_set_id) + '_' + str(args.nodes_num) + '_confusion.npy'

        # save args.dataset confusion_matrix
        places_confusion_matrix_array = np.array(places_confusion_matrix_lst)
        np.save(places_name, places_confusion_matrix_array)


def train_one_epoch_audio_visual(args, Audio_AFM, Visual_AFM, Audio_gcn_max_med_model, Visual_gcn_max_med_model, GAT_model, data_loader, optimizer, criterion, epoch, mixup_fn, lr_scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    norm_meter = AverageMeter()
    num_steps = len(data_loader)

    # switch to train mode
    top1 = AverageMeter()
    top5 = AverageMeter()

    Audio_AFM.train()
    Visual_AFM.train()
    Audio_gcn_max_med_model.train()
    Visual_gcn_max_med_model.train()
    GAT_model.train()
    
    # measure the inference time and fps
    total_time = 0.0

    optimizer.zero_grad()

    start = time.time()
    
    # SUN RGBD-Full
    # for idx, (visual, depth) in enumerate(data_loader):
    # 
    #     audio = depth[0]
    #     labels = visual[1]
    #     visual = visual[0]

        # print('visual.shape:', visual.shape)
        # print('visual.shape:', visual.shape)
        # print('audio.shape:', audio.shape)
        # 
        # print('labels 1.shape:', labels)
        # print('labels 2.shape:', depth[1].shape, depth[1])

        # print('audio 0:', audio[0].shape, audio[0])
        # print('visual 0:', visual[0].shape, visual[0])
        # print('audio 1:', audio[1].shape, audio[1])
        # print('audio 1:', visual[1].shape, visual[1])

    for idx, (audio, visual, labels, paths) in enumerate(data_loader):

        # measure data loading time
        data_time.update(time.time() - start)
        # lr = lr_scheduler.get_lr()
        lr = optimizer.param_groups[0]['lr']
        # print('optimizer:', optimizer.param_groups[0]['lr'])

        # data to cuda
        images = torch.autograd.Variable(visual).cuda()
        audio = torch.autograd.Variable(audio).cuda()

        labels = labels.cuda()

        if mixup_fn is not None:
            images, visual_labels = mixup_fn(images, labels)
            audio, audio_labels = mixup_fn(images, labels)

        if (args.fusion == False):
            # compute output
            img_output, _ = AFM(images)
            gcn_output, rows, columns = gcn_max_med_model(img_output)

            loss = criterion(gcn_output, labels)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()  # optimizing
            # measure accuracy and record loss
            lr_scheduler.step_update(epoch * num_steps + idx)

        else:
            # compute output
            start_time = timeit.default_timer()

            visual_fpam_output, visual_resnet_output = Visual_AFM(images)
       

            visual_max_graph, visual_med_graph, rows, columns = Visual_gcn_max_med_model(visual_fpam_output, visual_resnet_output)
            print('visual_max_graph:', visual_max_graph.shape)
            print('visual_med_graph:', visual_med_graph.shape)
            print('visual_resnet_output:', visual_resnet_output.shape)

            audio_fpam_output, audio_resnet_output = Audio_AFM(audio)

            
            audio_max_graph, audio_med_graph, rows, columns = Audio_gcn_max_med_model(audio_fpam_output, audio_resnet_output)
            print('audio_max_graph:', audio_max_graph.shape)
            print('audio_med_graph:', audio_med_graph.shape)
            print('audio_resnet_output:', audio_resnet_output.shape)

            graph_atten_output = GAT_model(audio_max_graph, audio_med_graph, audio_resnet_output, visual_max_graph, visual_med_graph, visual_resnet_output)
            print('graph_atten_output:', graph_atten_output.shape)

            end_time = timeit.default_timer()
            total_time += end_time - start_time

            prec1, prec5 = accuracy(graph_atten_output, labels, topk=(1, 5))

            if config.TRAIN.ACCUMULATION_STEPS > 1:

                loss = criterion(graph_atten_output, labels)
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
                        grad_norm = torch.nn.utils.clip_grad_norm_(Visual_AFM.parameters(), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(Visual_AFM.parameters())

                if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step_update(epoch * num_steps + idx)
            else:
                loss = criterion(graph_atten_output, labels)
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
                        grad_norm = torch.nn.utils.clip_grad_norm_(Visual_AFM.parameters(), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(Visual_AFM.parameters())

                optimizer.step()
                lr_scheduler.step_update(epoch * num_steps + idx)

            torch.cuda.synchronize()

            # optimizer.zero_grad()
            # #loss.backward(retain_graph=True)    # loss propagation
            # loss.backward()    # loss propagation
            # optimizer.step()                    # optimizing
            # lr_scheduler.step_update(epoch * num_steps + idx)
            # norm_meter.update(grad_norm)

        losses.update(loss, images.size(0))
        top1.update(prec1, images.size(0))
        top5.update(prec5, images.size(0))

        # for name, param in Audio_AFM.named_parameters():
        #     if param.grad is None:
        #         print('Audio_AFM:', name)
        # for name, param in Visual_AFM.named_parameters():
        #     if param.grad is None:
        #         print('Visual_AFM:', name)
        # for name, param in Audio_gcn_max_med_model.named_parameters():
        #     if param.grad is None:
        #         print('Audio_gcn_max_med_model:', name)
        # for name, param in Visual_gcn_max_med_model.named_parameters():
        #     if param.grad is None:
        #         print('Visual_gcn_max_med_model:', name)
        #
        # for name, param in GAT_model.named_parameters():
        #     if param.grad is None:
        #         print('GAT_model:', name)

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
    # measure the inference time and fps
    obtain_inference_time_fps(args, total_time, type='train')

    return losses.avg, top1.avg, top5.avg


def train_one_epoch(args, AFM, gcn_max_med_model, data_loader, optimizer, criterion, epoch, mixup_fn, lr_scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    norm_meter = AverageMeter()
    num_steps = len(data_loader)
    # switch to train mode
    top1 = AverageMeter()
    top5 = AverageMeter()

    AFM.train()
    gcn_max_med_model.train()
    optimizer.zero_grad()

    # measure the inference time and fps
    total_time = 0.0

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

        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        if (args.fusion == False):
            # compute output
            img_output, _ = AFM(images)
            gcn_output, rows, columns = gcn_max_med_model(img_output)

            loss = criterion(gcn_output, labels)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()  # optimizing
            # measure accuracy and record loss
            lr_scheduler.step_update(epoch * num_steps + idx)

        elif (args.atten_type == 'baseline' or args.atten_type == 'fpam_esc'):
            if args.atten_type == 'fpam_esc':
                fpam_output, _ = AFM(images)
            else:
                fpam_output = AFM(images)

            # compute gradient and loss for SGD step
            loss = criterion(fpam_output, labels)
            prec1, prec5 = accuracy(fpam_output, labels, topk=(1, 5))

            # optimizer.zero_grad()
            # loss.backward(retain_graph=True)
            # optimizer.step()  # optimizing
            # # measure accuracy and record loss
            # lr_scheduler.step_update(epoch * num_steps + idx)

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

            torch.cuda.synchronize()

        else:
            # compute output
            start_time = timeit.default_timer()
            fpam_output, resnet_output = AFM(images)
            gcn_output, rows, columns = gcn_max_med_model(fpam_output, resnet_output)
            end_time = timeit.default_timer()
            total_time += end_time - start_time

            # print('gcn_output mean:', torch.mean(gcn_output), torch.var(gcn_output))
            # print("gcn_output:",gcn_output.shape)
            prec1, prec5 = accuracy(gcn_output, labels, topk=(1, 5))

            # loss = criterion(gcn_output, labels)

            if config.TRAIN.ACCUMULATION_STEPS > 1:
                loss = criterion(gcn_output, labels)
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
                loss = criterion(gcn_output, labels)
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

            torch.cuda.synchronize()

            # optimizer.zero_grad()
            # #loss.backward(retain_graph=True)    # loss propagation
            # loss.backward()    # loss propagation
            # optimizer.step()                    # optimizing
            # lr_scheduler.step_update(epoch * num_steps + idx)
            # norm_meter.update(grad_norm)

        losses.update(loss, images.size(0))
        top1.update(prec1, images.size(0))
        top5.update(prec5, images.size(0))
        
        norm_meter.update(grad_norm)

        # measure elapsed time
        batch_time.update(time.time() - start)

        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Learning rate {lr:.4f}\t'.format(
                epoch, idx, len(data_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                lr=lr,))
    

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    # measure the inference time and fps
    obtain_inference_time_fps(args, total_time, type='train')

    return losses.avg, top1.avg, top5.avg

    
def validate_audio_visual(args, Audio_AFM, Visual_AFM, Audio_gcn_max_med_model, Visual_gcn_max_med_model, GAT_model,
                          data_loader, epoch):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    criterion = torch.nn.CrossEntropyLoss()

    Audio_AFM.eval()
    Visual_AFM.eval()
    Audio_gcn_max_med_model.eval()
    Visual_gcn_max_med_model.eval()
    GAT_model.eval()

    # measure inference time and fps
    total_time = 0.0

    conf_matrix = torch.zeros(args.num_classes, args.num_classes)

    end = time.time()
    with torch.no_grad():
        gcn_output_lst = []
        labels_lst = []
        all_pred = []
        all_true = []
        all_scores = []

        # SUN RGBD-Full
        for idx, (visual, depth) in enumerate(data_loader):
            audio = depth[0]
            labels = visual[1]
            images = visual[0]
        
        #for i, (audio, images, labels, paths) in enumerate(data_loader):

            # data to cuda
            images = torch.autograd.Variable(images).cuda()
            audio = torch.autograd.Variable(audio).cuda()
            labels = labels.cuda()

            if (args.fusion == False):
                fpam_output, resnet_output = AFM(images)
                gcn_output, rows, columns = gcn_max_med_model(fpam_output)

                loss = criterion(gcn_output, labels)
                # measure accuracy and record loss
                prec1, prec5 = accuracy(gcn_output, labels, topk=(1, 5))

            else:
                # compute output
                start_time = timeit.default_timer()

                visual_fpam_output, visual_resnet_output = Visual_AFM(images)
                visual_max_graph, visual_med_graph, rows, columns = Visual_gcn_max_med_model(visual_fpam_output,
                                                                                             visual_resnet_output)

                audio_fpam_output, audio_resnet_output = Audio_AFM(audio)
                audio_max_graph, audio_med_graph, rows, columns = Audio_gcn_max_med_model(audio_fpam_output,
                                                                                          audio_resnet_output)

                graph_atten_output = GAT_model(audio_max_graph, audio_med_graph, audio_resnet_output, visual_max_graph,
                                               visual_med_graph, visual_resnet_output)
                end_time = timeit.default_timer()
                total_time += end_time - start_time

                # compute gradient and loss for SGD step
                loss = criterion(graph_atten_output, labels)

                prediction = torch.max(graph_atten_output, 1)[1]

                conf_matrix = confusion_matrix_cal(prediction, labels=labels, conf_matrix=conf_matrix)

                all_pred.extend(prediction)
                all_true.extend(labels)
                all_scores.append(graph_atten_output)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(graph_atten_output, labels, topk=(1, 5))

                gcn_output_lst.append(graph_atten_output)
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
        f1_score = performance_matrix(all_true, all_pred)

        # measure the inference time and fps
        obtain_inference_time_fps(args, total_time, type='test')

        # plot the precision recall curve
        plot_precision_recall_curve(args, all_scores, all_true, epoch)

        # gcn_outputs = gcn_output_lst[0]
        # targets = labels_lst[0]
        # for i in range(len(gcn_output_lst) - 1):
        #     gcn_outputs = torch.cat((gcn_outputs, gcn_output_lst[i + 1]), dim=0)
        #     targets = torch.cat((targets, labels_lst[i + 1]), dim=0)
        #
        # gcn_outputs = gcn_outputs.cpu().data.numpy()
        # targets = targets.cpu().data.numpy()
        # print('gcn_outputs.shape:', gcn_outputs.shape)
        # print('targets.shape:', targets.shape)
        #
        # data_tsne = TSNE(n_components=2, random_state=33).fit_transform(gcn_outputs)
        # fig = plot_embedding(data_tsne,
        #                     targets,
        #                      epoch,
        #                     't-SNE visualization of latent features (epoch {})'.format(epoch))
        # plt.close(fig)

        # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], color=plt.cm.Set1(targets / 10.))
        # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=targets, alpha=0.6)

    return losses.avg, top1.avg, top5.avg, conf_matrix.numpy(), f1_score*100.0




def validate(args, AFM, gcn_max_med_model, data_loader, criterion, epoch):


    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    criterion = torch.nn.CrossEntropyLoss()
    AFM.eval()
    gcn_max_med_model.eval()
    conf_matrix = torch.zeros(args.num_classes, args.num_classes)

    # measure inference time and fps
    total_time = 0.0

    end = time.time()
    with torch.no_grad():
        gcn_output_lst = []
        labels_lst = []
        all_pred = []
        all_true = []
        all_scores = []
        
        for i, (images, labels, paths) in enumerate(data_loader):

            # data to cuda
            images = torch.autograd.Variable(images).cuda()
            labels = labels.cuda()

            if (args.fusion == False):
                fpam_output, resnet_output = AFM(images)
                gcn_output, rows, columns = gcn_max_med_model(fpam_output)

                loss = criterion(gcn_output, labels)
                # measure accuracy and record loss
                prec1, prec5 = accuracy(gcn_output, labels, topk=(1, 5))

            elif (args.atten_type == 'baseline' or args.atten_type == 'fpam_esc'):
                if args.atten_type == 'fpam_esc':
                    fpam_output, _ = AFM(images)
                else:
                    fpam_output = AFM(images)

                # compute gradient and loss for SGD step
                loss = criterion(fpam_output, labels)
                # measure accuracy and record loss
                prec1, prec5 = accuracy(fpam_output, labels, topk=(1, 5))

                prediction = torch.max(fpam_output, 1)[1]
                conf_matrix = confusion_matrix_cal(prediction, labels=labels, conf_matrix=conf_matrix)
                gcn_output_lst.append(fpam_output)
                labels_lst.append(labels)

            else:

                start_time = timeit.default_timer()
                fpam_output, resnet_output = AFM(images)
                gcn_output, rows, columns = gcn_max_med_model(fpam_output, resnet_output)
                end_time = timeit.default_timer()
                total_time += end_time - start_time

                # compute gradient and loss for SGD step
                loss = criterion(gcn_output, labels)

                prediction = torch.max(gcn_output, 1)[1]
                # print('prediction:', prediction.shape, prediction)


                conf_matrix = confusion_matrix_cal(prediction, labels=labels, conf_matrix=conf_matrix)

                # add the results to torch for visualization
                all_pred.extend(prediction)
                all_true.extend(labels)
                all_scores.append(gcn_output)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(gcn_output, labels, topk=(1, 5))


                gcn_output_lst.append(gcn_output)
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

        # precision recall f1-score
        f1_score = performance_matrix(all_true, all_pred)

        # measure the inference time and fps
        obtain_inference_time_fps(args, total_time, type='test')

        # precision recall curve
        plot_precision_recall_curve(args, all_scores, all_true, epoch)
        
        # scores = all_scores[0]
        # for i in range(len(all_scores)-1):
        #     scores = torch.cat((scores, all_scores[i + 1]), dim=0)
        # 
        # all_scores = torch.tensor(scores).cpu().data.numpy()
        # print('all_scores.shape:', all_scores.shape)
        # 
        # all_true = label_binarize(all_true, classes=[*range(args.num_classes)])
        # 
        # precision = dict()
        # recall = dict()
        # for i in range(args.num_classes):
        #     precision[i], recall[i], _ = metrics.precision_recall_curve(all_true[:, i], all_scores[:, i])
        #     plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
        # 
        # plt.xlabel("Recall")
        # plt.ylabel("Precision")
        # plt.legend(loc="best")
        # plt.title("Precision vs. Recall Curve")
        # plt.savefig("visualization/pr_curve/{}/epoch_{}.jpg".format(args.dataset_name, epoch))

        # gcn_outputs = gcn_output_lst[0]
        # targets = labels_lst[0]
        # for i in range(len(gcn_output_lst) - 1):
        #     gcn_outputs = torch.cat((gcn_outputs, gcn_output_lst[i + 1]), dim=0)
        #     targets = torch.cat((targets, labels_lst[i + 1]), dim=0)
        # 
        # gcn_outputs = gcn_outputs.cpu().data.numpy()
        # targets = targets.cpu().data.numpy()
        # print('gcn_outputs.shape:', gcn_outputs.shape)
        # print('targets.shape:', targets.shape)
        # 
        # data_tsne = TSNE(n_components=2, random_state=33).fit_transform(gcn_outputs)
        # fig = plot_embedding(data_tsne,
        #                     targets,
        #                      epoch,
        #                     't-SNE visualization of latent features (epoch {})'.format(epoch))
        # plt.close(fig)

        # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], color=plt.cm.Set1(targets / 10.))
        # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=targets, alpha=0.6)


    return losses.avg, top1.avg, top5.avg, conf_matrix.numpy(), f1_score*100.0


def plot_embedding(data, label, epoch, title='t-SNE visualization of latent features'):
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
    tSNE_filename = 'images/visual_tsne_{}.png'.format(epoch)
    plt.savefig(tSNE_filename, dpi=150)

    return fig

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


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lri))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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

    # distributed training
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    print('rank:', rank)
    print('config.LOCAL_RANK:', config.LOCAL_RANK)



    torch.cuda.set_device(config.LOCAL_RANK)

    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED  # + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # distributed training
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0

    # single GPU training
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


