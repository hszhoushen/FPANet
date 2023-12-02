import sys

sys.path.append('../')

# sys.path.append("../data/")


import os
print('print sys.path:', sys.path)
print('get current working directory', os.getcwd())

import numpy as np
import torch
import time
import shutil

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
from tensorboardX import SummaryWriter


from model.CVS_dataset import CVS_Audio
from model.network import GCN_audio_top_med_fea
from model.network import GCN_audio_top_med
from model.network import Audio_Fusion_Classifier

from model.network import AFM
from arguments import arguments_parse
from arguments import dataset_selection


best_prec1 = 0
use_cuda = torch.cuda.is_available()


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


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
    audio_dataset = CVS_Audio(args, data_dir, data_sample, data_type='train')
    audio_dataloader = DataLoader(dataset=audio_dataset, batch_size=args.bs, shuffle=True,
                                  num_workers=args.num_threads)

    # test set
    test_audio_dataset = CVS_Audio(args, data_dir, data_sample, data_type='test')
    # for idx, (data, image) in enumerate(test_audio_dataset):
    #     print(idx)
    test_audio_dataloader = DataLoader(dataset=test_audio_dataset, batch_size=args.bs, shuffle=True,
                                       num_workers=args.num_threads)

    # create the model for classification
    # audio_net = resnet50(num_classes=527)
    # state = torch.load(args.audio_net_weights)['model']
    # audio_net.load_state_dict(state)
    # audio_net.fc = nn.Linear(2048, args.num_classes)
    # audio_net = audio_net.cuda()

    # Attention Model Initialization
    AFM_net = AFM(args)
    AFM_net = AFM_net.cuda()

    # audio gcn model
    K = args.nodes_num

    # single model
    if (args.fusion == False):
        audio_gcn_model = GCN_audio_top_med(args).cuda()
        audio_classifier = Audio_Fusion_Classifier(args.num_classes).cuda()
    # fusion model
    else:
        audio_gcn_model = GCN_audio_top_med_fea(args).cuda()
        audio_classifier = Audio_Fusion_Classifier(args.num_classes).cuda()


    # Configure gradient
    for param in AFM_net.parameters():
        param.requires_grad = True
    for param in audio_gcn_model.parameters():
        param.requires_grad = True
    for param in audio_classifier.parameters():
        param.requires_grad = True


    # optimizer
    audio_net_params = list(AFM_net.parameters())
    audio_gcn_model_params = list(audio_gcn_model.parameters())
    audio_classifier_params = list(audio_classifier.parameters())

    optimizer = torch.optim.SGD([{'params':audio_net_params},
                                 {'params':audio_gcn_model_params, 'lr':0.01},
                                 {'params':audio_classifier_params, 'lr':0.01}],
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # writer
    writer = SummaryWriter(comment=args.model_type)


    for epoch in range(args.start_epoch, args.epochs):

        # adjust learning rate before training
        adjust_learning_rate(args, optimizer, epoch)

        # network training
        train_losses, train_acc1, train_acc5 = net_train(args, AFM_net, audio_gcn_model, audio_classifier,
                                                         audio_dataloader, optimizer, epoch)

        # test test
        test_losses, test_acc1, test_acc5 = net_validate(args, AFM_net, audio_gcn_model, audio_classifier,
                                                         test_audio_dataloader, optimizer, epoch)

        # best model name and latest model name
        model_dir = os.path.join('../weights', args.dataset_name)
        last_model_name = args.model_type + '_' + str(args.test_set_id) + '_' + str(K) + \
                          '_epoch_' + str(epoch) + '.pth.tar'

        best_model_name = args.model_type + '_' + str(args.test_set_id) + '_' + str(K) + \
                          '_epoch_' + str(epoch) + '_best' + '.pth.tar'

        last_model_path = os.path.join(model_dir, last_model_name)
        best_model_path = os.path.join(model_dir, best_model_name)

        # remember best prec@1 and save checkpoint
        is_best = test_acc1 > best_prec1
        best_prec1 = max(test_acc1, best_prec1)
        print("The best test accuracy obtained during training is = {}".format(best_prec1))

        if (epoch > 10 and epoch % 20 == 0):
            # save checkpoints
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'audio_net_state_dict': AFM_net.state_dict(),
                'audio_gcn_model_state_dict':audio_gcn_model.state_dict(),
                'audio_classifier_state_dict': audio_classifier.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, best_model_path, last_model_path)

        # TensorboardX writer
        writer.add_scalar('LR/Train', args.lr, epoch)
        writer.add_scalar('Acc1/Train', train_acc1, epoch)
        writer.add_scalar('Acc1/Test', test_acc1, epoch)
        writer.add_scalar('Acc5/Train', train_acc5, epoch)
        writer.add_scalar('Acc5/Test', test_acc5, epoch)
        writer.add_scalar('Loss/Train', train_losses, epoch)
        writer.add_scalar('Loss/Test', test_losses, epoch)

    # close the SummaryWriter
    writer.close()

def net_train(args, AFM, audio_gcn_model, audio_classifier, data_loader, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    AFM.train()
    audio_gcn_model.train()
    audio_classifier.train()
    criterion = nn.CrossEntropyLoss().cuda()
    correct = 0
    total = 0


    end = time.time()
    for i, data in enumerate(data_loader, 0):
        # measure data loading time
        data_time.update(time.time() - end)

        # clear optimizer
        optimizer.zero_grad()

        # read data
        audio, label = data

        # to cuda
        audio = audio.type(torch.FloatTensor).cuda()
        label = label.cuda()


        # output[0].shape, bs x 2048
        # output[1].shape, bs x args.num_classes
        # output[2].shape, bs x 2048 x 13 x 4 (before fc)
        if (args.fusion == False):

            if(args.mixup == True):

                inputs, targets_a, targets_b, lam = mixup_data(audio, label,
                                                               args.alpha, use_cuda)
                inputs, targets_a, targets_b = map(Variable, (inputs,
                                                              targets_a, targets_b))
                audio_output, _ = AFM(inputs)
                gcn_output = audio_gcn_model(audio_output)
                loss = mixup_criterion(criterion, gcn_output, targets_a, targets_b, lam)

                # loss = criterion(gcn_output, label)
                loss.backward()  # loss propagation
                optimizer.step()  # optimizing

                _, predicted = torch.max(gcn_output.data, 1)
                correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                            + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
                total += label.size(0)

                prec1 = correct / total
                prec5 = correct / total


            else:
                audio_output, _ = AFM(audio)
                gcn_output = audio_gcn_model(audio_output)


                # compute gradient and loss for SGD step
                loss = criterion(gcn_output, label)
                loss.backward()                 # loss propagation
                optimizer.step()                # optimizing

                # measure accuracy and record loss
                prec1, prec5 = accuracy(gcn_output, label, topk=(1, 5))

        # Fusion
        else:
            if (args.mixup == True):
                inputs, targets_a, targets_b, lam = mixup_data(audio, label,
                                                               args.alpha, use_cuda)
                inputs, targets_a, targets_b = map(Variable, (inputs,
                                                              targets_a, targets_b))
                audio_output, resnet_output = AFM(inputs)
                # print('audio_output.shape:', audio_output.shape)
                # print('resnet_output.shape:', resnet_output.shape)

                gcn_output = audio_gcn_model(audio_output)
                # print('gcn_output.shape:', gcn_output.shape)

                fused_output = audio_classifier(gcn_output, resnet_output)
                loss = mixup_criterion(criterion, fused_output, targets_a, targets_b, lam)

                loss.backward()   # loss propagation
                optimizer.step()  # optimizing

                # measure accuracy and record loss
                _, predicted = torch.max(gcn_output.data, 1)
                correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                            + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
                total += label.size(0)

                prec1 = correct / total
                prec5 = correct / total

            else:
                audio_output, resnet_output = AFM(audio)
                # print('audio_output.shape:', audio_output.shape)
                # print('resnet_output.shape:', resnet_output.shape)

                gcn_output = audio_gcn_model(audio_output)
                # print('gcn_output.shape:', gcn_output.shape)

                fused_output = audio_classifier(gcn_output, resnet_output)

                # compute gradient and loss for SGD step
                loss = criterion(fused_output, label)
                loss.backward()  # loss propagation
                optimizer.step()  # optimizing

                # measure accuracy and record loss
                prec1, prec5 = accuracy(fused_output, label, topk=(1, 5))


        losses.update(loss, audio.size(0))
        top1.update(prec1, audio.size(0))
        top5.update(prec5, audio.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  #'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  #'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(data_loader), loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def net_validate(args, AFM, audio_gcn_model, audio_classifier, data_loader, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    AFM.eval()
    audio_gcn_model.eval()
    audio_classifier.eval()

    criterion = nn.CrossEntropyLoss().cuda()

    end = time.time()

    with torch.no_grad():

        for i, data in enumerate(data_loader, 0):
            # measure data loading time
            data_time.update(time.time() - end)

            audio, label = data
            audio = audio.type(torch.FloatTensor).cuda()
            label = label.cuda()

            # output[0].shape, bs x 2048
            # output[1].shape, bs x args.num_classes
            # output[2].shape, bs x 2048 x 13 x 4 (before fc)

            if (args.fusion == False):
                audio_output, _ = AFM(audio)
                # print('audio_output.shape:', audio_output.shape)
                # print('resnet_output.shape:', resnet_output.shape)
                # print('fea3.shape:', fea3.shape)
                # print('fea4.shape:', fea4.shape)
                # atten = AFM(fea3, fea4)
                # print('audio_output.shape:', audio_output.shape)

                # print('Faudio.shape:', Faudio.shape)
                gcn_output = audio_gcn_model(audio_output)
                # print('gcn_output.shape:', gcn_output.shape)

                # fused_output = audio_classifier(audio_output, gcn_output)

                # compute gradient and loss for SGD step
                loss = criterion(gcn_output, label)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(gcn_output, label, topk=(1, 5))

            else:
                # print('audio.shape:', audio.shape)
                audio_output, resnet_output = AFM(audio)
                # print('audio_output.shape:', audio_output.shape)
                # print('resnet_output.shape:', resnet_output.shape)

                gcn_output = audio_gcn_model(audio_output)
                # print('gcn_output.shape:', gcn_output.shape)

                fused_output = audio_classifier(gcn_output, resnet_output)

                # compute gradient and loss for SGD step
                loss = criterion(fused_output, label)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(fused_output, label, topk=(1, 5))


            losses.update(loss, audio.size(0))
            top1.update(prec1, audio.size(0))
            top5.update(prec5, audio.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      #'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      #'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(data_loader), loss=losses, top1=top1, top5=top5))

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

def accuracy(output, target, topk=(1,), mixup=False):
    """Computes the precision@k for the specified values of k"""
    if (mixup == False):
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


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lri))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# main function
if __name__ == '__main__':
    main()



