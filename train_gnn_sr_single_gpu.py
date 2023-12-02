# Developed by Liguang Zhou

import os
import shutil
import time

import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from model.network import GCN_audio_top_med_fea, GCN_max_med_fusion
from model.network import GCN_audio_top_med
from model.network import AFM, FPAM

from arguments import arguments_parse
from dataset import ImageFolderWithPaths, DatasetSelection

import sys
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



def rec_cal(index_width, index_height, feature_width, feature_height, timemax):
    feature_height_max = 8192
    min_f0, max_f0 = (index_width) * (timemax / feature_width), (index_width + 1) * (timemax / feature_width)
    min_f1, max_f1 = (index_height) * (feature_height_max / feature_height), (index_height + 1) * (
                feature_height_max / feature_height)
    width = max_f0 - min_f0
    height = max_f1 - min_f1

    return min_f0, min_f1, width, height

def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return img[starty:starty+cropy, startx:startx+cropx, :]
    

def img_drawgraph(imgpath, savepath, rows, columns):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    margin = .1
    pic_elements = []
    img = Image.open(imgpath)
    # centorcrop
    img = img.resize([256,256])
    img = crop_center(np.array(img),224,224)
    img = Image.fromarray(img)


    plt.imshow(img)

    print('len(rows):', len(rows))
    show_color=True
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
        if i<len(rows)//2:
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
            ax.annotate(str(num + 1), (rows[i]*8, columns[i]*8), color=clr, weight='bold', ha='center', va='center', size=7)
            # ax.add_patch()
            # ax.annotate(str(num + 1), (rows[i]*8, columns[i]*8),fontsize=1, color=clr) #bbox={"boxstyle": "circle", "color": clr, "pad": 0.001, "alpha": al}
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

    plt.savefig(savepath)

def main():
    global args, best_prec1, discriminative_matrix
    args = arguments_parse.argsParser()
    print(args)
    
    # dataset selection for training
    dataset_selection = DatasetSelection(args.dataset_name)
    data_dir = dataset_selection.datasetSelection()
    # discriminative_matrix = dataset_selection.discriminative_matrix_estimation()

    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')

    # Attention Model Initialization
    if (args.atten_type == 'afm'):
        FPAM_net = AFM(args)

    elif(args.atten_type == 'fpam'):
        FPAM_net = FPAM(args)

    FPAM_net = FPAM_net.cuda()

    # single model
    if (args.fusion == False):
        gcn_max_med_model = GCN_audio_top_med(args).cuda()
    # fusion model
    else:
        gcn_max_med_model = GCN_max_med_fusion(args).cuda()

    if(args.status == 'train'):
        # Configure gradient
        for param in FPAM_net.parameters():
            param.requires_grad = True
        for param in gcn_max_med_model.parameters():
            param.requires_grad = True

    else:
        # Configure gradient
        for param in FPAM_net.parameters():
            param.requires_grad = False
        for param in gcn_max_med_model.parameters():
            param.requires_grad = False


    cudnn.benchmark = True

    # create dataloader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = ImageFolderWithPaths(traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = ImageFolderWithPaths(valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.bs, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.bs, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()


    if args.status == 'test':
        # replacement
        # image_gcn_med_7f_pt_pafm_5_20_epoch_55.pth.tar, 91.1%
        # image_gcn_med_7f_pt_pafm_5_20_epoch_50.pth.tar, 90.7%
        # image_gcn_med_7f_pt_pafm_5_20_epoch_10.pth.tar, 88.3%
        # image_gcn_med_7f_pt_pafm_5_20_epoch_5.pth.tar, 82%
        # image_gcn_med_7f_pt_pafm_5_20_epoch_0.pth.tar, 76.4%
        if (args.dataset_name == 'SUNRGBD'):
            dataset_name = 'Places365-7'
        else:
            dataset_name = args.dataset_name

        best_model_name = os.path.join('./weights', dataset_name, args.model_name)

        # load the pretrained-models
        print("=> loading checkpoint '{}'".format(best_model_name))
        checkpoint = torch.load(best_model_name)

        # load the weights for model
        FPAM_net.load_state_dict(checkpoint['audio_net_state_dict'])
        gcn_max_med_model.load_state_dict(checkpoint['gcn_max_med_model_state_dict'])

        print("=> loaded checkpoint '{}' (epoch {}) (accuracy {})"
              .format(args.resume, checkpoint['epoch'], checkpoint['best_prec1']))

        # evaluate on validation set
        test_losses, test_acc1, test_acc5 = validate(args, FPAM_net, gcn_max_med_model, val_loader, criterion, 0)
        print('test_losses:', test_losses)
        print('test_acc1:', test_acc1)
        print('test_acc5:', test_acc5)

    if args.status == 'draw':
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

        best_model_name = os.path.join('./weights', dataset_name, args.model_name)
        print('best_model_name:', best_model_name)

        # load the pretrained-models
        print("=> loading checkpoint '{}'".format(best_model_name))
        checkpoint = torch.load(best_model_name)

        FPAM_net.load_state_dict(checkpoint['audio_net_state_dict'])
        gcn_max_med_model.load_state_dict(checkpoint['gcn_max_med_model_state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) (accuracy {})"
              .format(args.resume, checkpoint['epoch'], checkpoint['best_prec1']))

        FPAM_net.eval()
        gcn_max_med_model.eval()
        # added
        for param in FPAM_net.parameters():          # freeze netT
            param.requires_grad = False
        for param in gcn_max_med_model.parameters():  # freeze netT
            param.requires_grad = False


        img_file_lst = ['val/corridor/Places365_val_00034934.jpg', 'val/corridor/Places365_val_00034820.jpg', 'val/corridor/Places365_val_00028101.jpg',
                        'val/bedroom/Places365_val_00015345.jpg', 'val/bedroom/Places365_val_00001822.jpg', 'val/bedroom/Places365_val_00009488.jpg',
                        'val/bedroom/Places365_val_00020357.jpg',
                        'val/office/Places365_val_00001805.jpg', 'val/office/Places365_val_00004918.jpg', 'val/office/Places365_val_00008146.jpg',
                        'val/dining_room/Places365_val_00017334.jpg', 'val/dining_room/Places365_val_00011164.jpg', 'val/dining_room/Places365_val_00000611.jpg',
                        'val/living_room/Places365_val_00035644.jpg', 'val/living_room/Places365_val_00008022.jpg', 'val/living_room/Places365_val_00018413.jpg',
                        'val/kitchen/Places365_val_00035756.jpg', 'val/kitchen/Places365_val_00007731.jpg', 'val/kitchen/Places365_val_00004746.jpg',
                        'val/bathroom/Places365_val_00029507.jpg', 'val/bathroom/Places365_val_00019364.jpg', 'val/bathroom/Places365_val_00028614.jpg']

        # img_file_lst = ['val/corridor/Places365_val_00034934.jpg', 'val/corridor/Places365_val_00034934.jpg']
        with torch.no_grad():
            for i in range(len(img_file_lst)):
                FPAM_net.eval()
                gcn_max_med_model.eval()
                # #added
                # for param in FPAM_net.parameters(): #freeze netT
                #     param.requires_grad = False
                # for param in gcn_max_med_model.parameters(): #freeze netT
                #     param.requires_grad = False


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
                img = img.unsqueeze(dim=0).cuda()
                print(img.size())

                # sound, label = sound_feature_extract(sound_path, wave_file_lst[i])
                img_output, resnet_output = FPAM_net(img)
                print('img_output.shape:', img_output.shape, 'resnet_output.shape:', resnet_output.shape)
                gcn_output, rows, columns = gcn_max_med_model(img_output)
                print('gcn_output.shape:', gcn_output.shape, gcn_output)

                rows = rows.squeeze()
                columns = columns.squeeze()
                print('rows:', rows.shape, len(rows))
                print('columns:', columns.shape, len(columns))

                save_path = os.path.join('./results', img_file_lst[i].split('/')[2])
                img_drawgraph(img_path, save_path, rows, columns)



                # Fig_name = os.path.join('STG', label_lst[i])
                # drawgraph(sound_path, Fig_name, rows, columns, feature_width=26, feature_height=8)
                # params = list(gcn_max_med_model.parameters())
                # weight_softmax = params[-2].cuda().data.cpu().numpy()
                # weight_softmax[weight_softmax < 0] = 0
                # print('weight size:', weight_softmax[0].shape)
                # CAMs = returnCAM(img_output[0], weight_softmax)
                # heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (26, 8)), cv2.COLORMAP_JET)
                # cv2.imwrite('./results/' + label_lst[i] + 'heatimg.png', heatmap)

    elif args.status == 'train':
        accuracies_list = []

        # writer
        writer = SummaryWriter(comment=args.model_type)

        # optimizer
        audio_net_params = list(FPAM_net.parameters())
        gcn_max_med_model_params = list(gcn_max_med_model.parameters())

        optimizer = torch.optim.SGD([{'params': audio_net_params},
                                     {'params': gcn_max_med_model_params, 'lr': 0.01}],
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        for epoch in range(args.start_epoch, args.epochs):

            adjust_learning_rate(optimizer, epoch)

            # if epoch != 0 and epoch % 10 == 0:
            #     print("=> loading checkpoint '{}'".format(best_model_name))
            #     checkpoint = torch.load(best_model_name)
            #
            #     FPAM_net.load_state_dict(checkpoint['audio_net_state_dict'])
            #     gcn_max_med_model.load_state_dict(checkpoint['gcn_max_med_model_state_dict'])
            #     print("=> loaded checkpoint '{}' (epoch {})"
            #           .format(args.resume, checkpoint['epoch']))


            # train for one epoch
            print('training')
            train_losses, train_acc1, train_acc5 = train(args, FPAM_net, gcn_max_med_model,
                                                         train_loader, optimizer, criterion, epoch)

            # evaluate on validation set
            print('testing!')
            test_losses, test_acc1, test_acc5 = validate(args, FPAM_net, gcn_max_med_model,
                                                         val_loader, criterion, epoch)

            # best model name and latest model name
            model_dir = os.path.join('./weights', args.dataset_name)
            # if dir is not exist, make one
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)

            last_model_name = args.model_type + '_' + str(args.atten_type) + '_' + str(args.experiment_id) + '_' + str(args.nodes_num) + \
                              '_epoch_' + str(epoch+1) + '.pth.tar'

            best_model_name = args.model_type + '_' + str(args.atten_type) + '_' + str(args.experiment_id) + '_' + str(args.nodes_num) + \
                              '_epoch_' + str(epoch+1) + '_best' + '.pth.tar'

            last_model_path = os.path.join(model_dir, last_model_name)
            best_model_path = os.path.join(model_dir, best_model_name)

            # remember best prec@1 and save checkpoint
            is_best = test_acc1 > best_prec1
            best_prec1 = max(test_acc1, best_prec1)
            print("The best test accuracy obtained during training is = {}".format(best_prec1))

            if (epoch > 0 and epoch % 2 == 0):
                # save checkpoints
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'best_prec1': best_prec1,
                    'audio_net_state_dict': FPAM_net.state_dict(),
                    'gcn_max_med_model_state_dict':gcn_max_med_model.state_dict(),
                }, is_best, best_model_path, last_model_path)

            # TensorboardX writer
            writer.add_scalar('LR/Train', args.lr, epoch)
            writer.add_scalar('Acc1/Train', train_acc1, epoch)
            writer.add_scalar('Acc1/Test', test_acc1, epoch)
            writer.add_scalar('Acc5/Train', train_acc5, epoch)
            writer.add_scalar('Acc5/Test', test_acc5, epoch)
            writer.add_scalar('Loss/Train', train_losses, epoch)
            writer.add_scalar('Loss/Test', test_losses, epoch)

            accuracies_list.append("%.2f"%test_acc1.tolist())



def train(args, AFM, gcn_max_med_model, data_loader, optimizer, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    AFM.train()
    gcn_max_med_model.train()

    end = time.time()
    for i, (image, label, path) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # data to cuda
        image = torch.autograd.Variable(image).cuda()
        label = label.cuda()

        if(args.fusion == False):
            # compute output
            img_output, _ = AFM(image)
            gcn_output, rows, columns = gcn_max_med_model(img_output)
            loss = criterion(gcn_output, label)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward(retain_graph=True)    # loss propagation
            optimizer.step()                    # optimizing

            # measure accuracy and record loss
            prec1, prec5 = accuracy(gcn_output.data, label, topk=(1, 5))

        else:
            # compute output
            img_output, resnet_output = AFM(image)
            gcn_output, rows, columns = gcn_max_med_model(img_output, resnet_output)

            loss = criterion(gcn_output, label)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward(retain_graph=True)    # loss propagation
            optimizer.step()                    # optimizing

            # measure accuracy and record loss
            prec1, prec5 = accuracy(gcn_output, label, topk=(1, 5))


        losses.update(loss, image.size(0))
        top1.update(prec1, image.size(0))
        top5.update(prec5, image.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch+1, i, len(data_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

def validate(args, AFM, gcn_max_med_model, data_loader, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    AFM.eval()
    gcn_max_med_model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (image, label, path) in enumerate(data_loader):

            # data to cuda
            image = torch.autograd.Variable(image).cuda()
            label = label.cuda()

            if (args.fusion == False):
                audio_output, _ = AFM(image)
                gcn_output, rows, columns = gcn_max_med_model(audio_output)

                loss = criterion(gcn_output, label)
                # measure accuracy and record loss
                prec1, prec5 = accuracy(gcn_output, label, topk=(1, 5))

            else:
                audio_output, resnet_output = AFM(image)

                gcn_output, rows, columns = gcn_max_med_model(audio_output, resnet_output)
                # print('gcn_output.shape:', gcn_output.shape, gcn_output)

                # compute gradient and loss for SGD step
                loss = criterion(gcn_output, label)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(gcn_output, label, topk=(1, 5))


            losses.update(loss, image.size(0))
            top1.update(prec1, image.size(0))
            top5.update(prec5, image.size(0))
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
                    epoch+1, i, len(data_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // 20))
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
    output= model.fc(feature)
    return feature

if __name__ == '__main__':
    main()
