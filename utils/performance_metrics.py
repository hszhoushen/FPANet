import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.preprocessing import label_binarize

from thop import profile
from thop import clever_format


def thop_params_calculation(network, inputs):
    flops, params = profile(network, inputs=(inputs,))

    return flops, params

def obtain_inference_time_fps(args, total_time, type='train'):
    print('total_time:', total_time)
    if args.dataset_name == 'Places365-7':
        if type == 'train':
            cnt = 35000
        else:
            cnt = 700
    elif args.dataset_name == 'SUN_RGBD':
        if type == 'train':
            cnt = 4845
        else:
            cnt = 4659
    elif args.dataset_name == 'Places365-14':
        if type == 'train':
            cnt = 75000
        else:
            cnt = 1500
    elif args.dataset_name == 'DCASE2021-Visual' or args.dataset_name == 'DCASE2021-Audio-Visual' or args.dataset_name == 'DCASE2021-Audio':
        if type == 'train':
            cnt = 8646
        else:
            cnt = 3645
    elif args.dataset_name == 'ADVANCE' or args.dataset_name == 'ADVANCE-Audio' or args.dataset_name == 'ADVANCE-Visual':
        if type == 'train':
            cnt = 2147
        else:
            cnt = 549

    elif args.dataset_name == 'ESC10':
        if type == 'train':
            cnt = 320
        else:
            cnt = 80

    elif args.dataset_name == 'ESC50':
        if type == 'train':
            cnt = 1600
        else:
            cnt = 400

    inference_time = total_time / cnt
    FPS = 1.0 / inference_time
    print('inference_time:', inference_time)
    print('FPS:', FPS)
    
def performance_matrix(args, true, pred):
    if args.dataset_name in ['ADVANCE', 'ADVANCE-Audio', 'ADVANCE-Visual']:
        precision = metrics.precision_score(true, pred, average='weighted')
        recall = metrics.recall_score(true, pred, average='weighted')
        accuracy = metrics.accuracy_score(true, pred)
        f1_score = metrics.f1_score(true, pred, average='weighted')
        print('Precision: {} Recall: {}, Accuracy: {}: ,f1_score: {}'.format(precision * 100, recall * 100, accuracy * 100,
                                                                            f1_score * 100))
    else:
        precision = metrics.precision_score(true, pred, average='micro')
        recall = metrics.recall_score(true, pred, average='micro')
        accuracy = metrics.accuracy_score(true, pred)
        f1_score = metrics.f1_score(true, pred, average='micro')
        print('Precision: {} Recall: {}, Accuracy: {}: ,f1_score: {}'.format(precision * 100, recall * 100, accuracy * 100,
                                                                            f1_score * 100))

    print('Confusion Matrix:\n', metrics.confusion_matrix(true, pred))


    return f1_score


# precision recall curve
def plot_precision_recall_curve(args, all_scores, all_true, epoch):
    # convert all_scores from cuda to numpy
    scores = all_scores[0]
    for i in range(len(all_scores) - 1):
        scores = torch.cat((scores, all_scores[i + 1]), dim=0)
    all_scores = scores.clone().detach().cpu().data.numpy()

    # filling or replacing those invalid or missing values with 0
    all_scores = np.nan_to_num(all_scores)
    print('all_scores.shape:', all_scores.shape)

    print('start label_binarize')
    all_true = label_binarize(all_true, classes=[*range(args.num_classes)])
    print('finish label_binarize')
    print('all_true.shape:', all_true.shape)

    precision = dict()
    recall = dict()
    for i in range(args.num_classes):
        #print('args.num_classes:', args.num_classes)
        #print('all_true[:, i], all_scores[:, i]:', all_true[:, i], all_scores[:, i])
        precision[i], recall[i], _ = metrics.precision_recall_curve(all_true[:, i], all_scores[:, i])
        # print('precision[i], recall[i]:', precision[i], recall[i])
        plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

    print('finish precision_recall_curve')

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.title("Precision vs. Recall Curve")

    fig_dir = "visualization/pr_curve/{}".format(args.dataset_name)

    print('fig_dir:', fig_dir)
    if not os.path.isdir(fig_dir):
        os.mkdir(fig_dir)
    fig_path = os.path.join(fig_dir, "epoch_{}.jpg".format(epoch))
    print('fig_path:', fig_path)
    plt.savefig(fig_path)
    plt.clf()
    
    
def params_calculation(FPAM_net, gcn_max_med_model):
    fpam_total_params = sum(param.numel() for param in FPAM_net.parameters())
    gcn_max_med_total_params = sum(param.numel() for param in gcn_max_med_model.parameters())
    print('# FPAM_net parameters:', fpam_total_params)
    print('# gcn_max_med_model parameters:', gcn_max_med_total_params)
    total_params = fpam_total_params + gcn_max_med_total_params
    print('total_params:', total_params)

    return total_params, fpam_total_params, gcn_max_med_total_params


def confusion_matrix_cal(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[t, p] += 1
    return conf_matrix

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