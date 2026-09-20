# Developed by Liguang Zhou

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import librosa
import cv2


def image_drawgraph(imgpath, savepath, rows, columns):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    margin = .1
    pic_elements = []
    img = Image.open(imgpath)
    # centorcrop
    img = img.resize([256,256])
    img = crop_center(np.array(img),224,224)
    img = Image.fromarray(img)

    # plt.rcParams['figure.figsize'] = (15.0,15.0)
    # plt.rcParams['savefig.dpi'] = 300               # 图片像素
    # plt.rcParams['figure.dpi'] = 300                # 分辨率

    plt.imshow(img)

    print('len(rows):', len(rows))
    show_color=False
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
            ax.add_patch(plt.Circle((rows[i]*8, columns[i]*8), 4, color=clr,alpha=0.8))
            #check reliablity
            # ax.annotate(str(rows[i]*8)+","+str(columns[i]*8), (rows[i]*8, columns[i]*8), color=clr, weight='bold', ha='center', va='center', size=4)
            # plt.annotate("", (rows[i]*8, columns[i]*8),bbox={"boxstyle": "circle", "color": clr, "pad": 0.0002, "alpha": al})
        else:
            if i<len(rows)//2:
                num=i
            else:
                num=i-len(rows)//2
            # ax.annotate(str(num + 1), (rows[i]*8, columns[i]*8), color=clr, weight='bold', ha='center', va='center', size=7)
            # ax.add_patch()
            ax.add_patch(plt.Circle((rows[i]*8, columns[i]*8), 4, color=clr,alpha=0.8))
            if num+1>=10:
                plt.text(rows[i]*8-3, columns[i]*8+1, str(num + 1), color='black',fontsize=3)
            else:
                plt.text(rows[i]*8-2, columns[i]*8+1, str(num + 1), color='black',fontsize=3)
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

    # 去除坐标系
    plt.axis('off')
    # 去除图像周围的白边
    width = 480
    height = 480
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig(savepath, dpi=300)
    plt.clf()



def audio_drawgraph(soundpath, figname, rows, columns, feature_width=26, feature_height=8):
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
