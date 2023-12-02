# by Liguang Zhou, 2020.9.30
from model.graph_init import Graph_Init

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import BatchNorm2d
from torch.nn import BatchNorm1d

from resnet_audio.resnet_audio import resnet18, resnet34, resnet50, resnet101,  resnet152


import torch.utils.model_zoo as model_zoo
import torchvision.models as models

# 默认的resnet网络，已预训
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

def audio_forward(K, Faudio):

    N = Faudio.shape[0]
    C = Faudio.shape[1]
    W = Faudio.shape[2]
    H = Faudio.shape[3]

    Fre = torch.empty(N, W, H).cuda()

    # acquire Fre (N,H,W) from Frgb (N,C,H,W)
    for channel in range(C):
        Fre = torch.add(Faudio[:, channel, :, :], Fre)
    # print('Fre:', Fre.shape, Fre[0,:,:])

    # Acquire the top K index of Frgb
    feature = Fre.view(N, H * W)
    # print('feature:', feature.shape)

    sorted, indices = torch.sort(feature, descending=True)
    # print('sorted:', sorted.shape, sorted)
    # print('indices:', indices.shape, indices)

    topK_index = indices[:, 0:K]
    # print('topK_index:', topK_index.shape, topK_index)

    nodes = []
    sound_batch = []
    rows = []
    columns = []
    cnt = 0

    # shape of topK_index is (N, nodes)
    for i in range(N):  # lop from one image to other based on batch size

        for j in range(K):  # loop from one node to other
            # for index in topK_index[0, :]:
            index = topK_index[i, j]
            if index < H:
                row = index // H
            else:
                row = index // H - 1
            rows.append(row)

            if index % H == 0:  # e.g., 36 (8,3)
                columns.append(torch.tensor(H - 1).cuda())
            else:
                columns.append(index % H - 1)


            # switch the feature from 1x1024 to 1024x1
            node = Faudio[i, :, rows[cnt], columns[cnt]].reshape(-1, 1)
            nodes.append(node)
            cnt = cnt + 1

        sound = torch.stack(nodes)
        nodes = []
        sound_batch.append(sound)


    # Adjacency matrix calculation
    sound_graph_construction = Graph_Init(K, N, rows, columns)
    Lnormtop, Lnormmed = sound_graph_construction.Dynamic_Lnorm()

    sound_batch = torch.stack(sound_batch)

    return sound_batch, Lnormtop.cuda(), Lnormmed.cuda()



def audio_med_forward(nodes_num, Faudio):
    N = Faudio.shape[0]  # batch size
    C = Faudio.shape[1]
    W = Faudio.shape[2]
    H = Faudio.shape[3]

    #This line causes nonreproducibility....
    #refer to https://pytorch.org/docs/1.4.0/notes/randomness.html
    # There are some PyTorch functions that use CUDA functions that can be a
    # source of non-determinism. One class of such CUDA functions are atomic
    # operations, in particular atomicAdd, where the order of parallel additions
    # to the same value is undetermined and, for floating-point variables, a source
    #  of variance in the result. PyTorch functions that use atomicAdd in the forward
    #  include torch.Tensor.index_add_(), torch.Tensor.scatter_add_(), torch.bincount().
    #Maybe in torch.add, there is some above-mentioned operations!

    # acquire Fre (N,H,W) from Frgb (N,C,H,W)
    Fre = torch.empty(N, W, H).cuda()
    for channel in range(C):
        Fre = torch.add(Faudio[:, channel, :, :], Fre)
    # Fre = torch.sum(Faudio, dim=1)

    # Acquire the top K index of Frgb
    feature = Fre.view(N, H * W)
    # print('feature:', feature.shape)
    sorted, indices = torch.sort(feature, descending=True)
    # print('sorted:', sorted.shape, sorted)
    # print('indices:', indices.shape, indices)

    # feature = feature.view(x.size(0), -1)
    # print('feature:', feature.shape)

    topK_index = indices[:, 0:nodes_num]
    # print("topK_index",topK_index)
    # print('topK_index:', topK_index.shape, topK_index)
    avg_index = indices[:, (H * W // 2 - nodes_num // 2 - 1):(H * W // 2 - nodes_num // 2 - 1 + nodes_num)]
    # print('avg_index:', avg_index.shape, avg_index)

    nodes = []
    sound_batch = []
    rows = []
    columns = []
    cnt = 0

    # shape of topK_index is (N, nodes)
    for i in range(N):  # lop from one image to other

        # top K
        for j in range(int(nodes_num/2)):  # loop from one node to other

            index = topK_index[i, j]
            # print('top K:', index) #debug
            if index < H:
                row = index // H
            else:
                # row = torch.tensor(index // H - 1).cuda()
                row = torch.tensor((index-1)//H).cuda()
            rows.append(row)

            if index % H == 0:  # e.g., 36 (8,3)
                column = torch.tensor(H-1).cuda()
            else:
                column = index % H - 1
            columns.append(column)

            # switch the feature from 1x1024 to 1024x1
            node = Faudio[i, :, rows[cnt], columns[cnt]].reshape(-1, 1)
            nodes.append(node)
            cnt = cnt + 1

        # median K
        for k in range(int(nodes_num/2)):
            index_avg = avg_index[i, k]
            # print('median K:', index_avg) #debug
            if index_avg < H:
                row = index_avg // H
            else:
                # row = torch.tensor(index_avg // H - 1).cuda()
                row = torch.tensor((index_avg-1)//H).cuda()
            rows.append(row)

            if index_avg % H == 0:
                column = torch.tensor(H-1).cuda()
            else:
                column = index_avg % H - 1
            columns.append(column)

            node = Faudio[i, :, rows[cnt], columns[cnt]].reshape(-1, 1)
            nodes.append(node)
            cnt = cnt + 1

        # top K
        for j in range(int(nodes_num/2), nodes_num):  # loop from one node to other

            index = topK_index[i, j]
            # print('top K:', index) #debug
            if index < H:
                row = index // H
            else:
                # row = torch.tensor(index // H - 1).cuda()
                row = torch.tensor((index-1)//H).cuda()
            rows.append(row)

            if index % H == 0:  # e.g., 36 (8,3)
                column = torch.tensor(H-1).cuda()
            else:
                column = index % H - 1
            columns.append(column)

            # switch the feature from 1x1024 to 1024x1
            node = Faudio[i, :, rows[cnt], columns[cnt]].reshape(-1, 1)
            nodes.append(node)

            cnt = cnt + 1

        # median K
        for k in range(int(nodes_num/2), nodes_num):
            index_avg = avg_index[i, k]
            # print('median K:', index_avg) #debug
            if index_avg < H:
                row = index_avg // H
            else:
                # row = torch.tensor(index_avg // H - 1).cuda()
                row = torch.tensor((index_avg-1)//H).cuda()
            rows.append(row)

            if index_avg % H == 0:
                column = torch.tensor(H - 1).cuda()
            else:
                column = index_avg % H - 1
            columns.append(column)

            node = Faudio[i, :, rows[cnt], columns[cnt]].reshape(-1, 1)
            nodes.append(node)
            cnt = cnt + 1

        sound = torch.stack(nodes)
        nodes = []
        sound_batch.append(sound)

    for row in rows:
        if row < 0:
            print(row.data)
    for column in columns:
        if column < 0:
            print(column.data)

    # testing return data
    # for (row,column) in zip(rows,columns):
    #     print(row.data, column.data)

    print('sound.shape:', sound.shape)
    sound_batch = torch.stack(sound_batch)
    print('sound_batch.shape:', sound_batch.shape)
    print('rows.shape:', rows.shape)
    print('columns.shape:', columns.shape)

    # Adjacency matrix calculation
    sound_graph_construction = Graph_Init(nodes_num, N, rows, columns)
    Lnormtop, Lnormmed = sound_graph_construction.Dynamic_Lnorm()

    return sound_batch, Lnormtop.cuda(), Lnormmed.cuda(), rows, columns




class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        # self.conv = ConvBNReLU(input_dim, 3*embed_dim, ks=1)

        self._reset_parameters()

    def scaled_dot_product(q, k, v, mask=None):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)


    def forward(self, x, mask=None, return_attention=False):
        print('x.size():', x.size())
        batch_size, seq_length, embed_dim, _ = x.size()
        qkv = self.qkv_proj(x)
        print('qkv.shape:', qkv.shape)
        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)
        print('q.shape:', q.shape)
        print('k.shape:', k.shape)
        print('v.shape:', v.shape)

        # Determine value outputs
        # values, attention = self.scaled_dot_product(q, k, v, mask=mask)
        # values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        # values = values.reshape(batch_size, seq_length, embed_dim)
        # o = self.o_proj(values)
        #
        # if return_attention:
        #     return o, attention
        # else:
        #     return o

class AttentionFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionFusionModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, kernel_size=1, stride=1, padding=0)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, feat16, feat32):
        feat32_up = F.interpolate(feat32, feat16.size()[2:], mode='nearest')
        #print('feat32_up:', feat32_up.shape)
        fcat = torch.cat([feat16, feat32_up], dim=1)
        #print('fcat.shape:', fcat.shape)
        feat = self.conv(fcat)
        #print('feat.shape:', feat.shape)

        atten = F.avg_pool2d(feat, feat.size()[2:])
        #print('atten.shape:', atten.shape)
        atten = self.conv_atten(atten)
        #print('atten.shape:', atten.shape)

        atten = self.bn_atten(atten)
        #print('atten.shape:', atten.shape)

        atten = self.sigmoid_atten(atten)
        #print('atten.shape:', atten.shape)

        return atten


class AFM_Net(nn.Module):
    def __init__(self, args, modality_type='audio'):
        super(AFM_Net, self).__init__()
        
        if args.dataset_name == 'ADVANCE' and modality_type == 'visual':
            print('resnet101 loading!')
            backbone = resnet101(num_classes=1000)
            print('Loading the pretrained model from the weights {%s}!' % model_urls['resnet101'])
            backbone.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
            backbone.fc = nn.Linear(2048, args.num_classes)
            # for param in backbone.parameters():
            #     param.requires_grad = True
                
        elif args.arch == 'resnet50':
            if modality_type == 'audio':
                # create the model for classification
                backbone = resnet50(num_classes=527)
                if modality_type == 'audio':
                    backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)

                state = torch.load(args.audio_net_weights)['model']
                backbone.load_state_dict(state)
                for param in backbone.parameters():
                    param.requires_grad = True

            elif modality_type == 'visual':
                if args.pretrain == 'Places365':
                    backbone = resnet50(num_classes=365)

                    print('use the model pretrained on Places365!')
                    weights_path = 'weights/resnet50_places365.pth.tar'
                    #state = torch.load(weights_path)['state_dict']
                    #backbone.load_state_dict(state)

                    import os
                    if os.path.exists(weights_path):
                        prop_saved = torch.load(weights_path)['state_dict']
                        from collections import OrderedDict
                        prop_selected = OrderedDict()
                        for k, v in prop_saved.items():
                            name = k[7:]  # remove `module.`
                            prop_selected[name] = v
                        backbone.load_state_dict(prop_selected)
                        print('Model loaded.')
                    else:
                        print("No model detected.")


                    #backbone.load_state_dict(model_zoo.load_url(model_urls[args.arch]))
                    print('Loading the pretrained model from the weights {%s}!' % weights_path)
                else:
                    print('use the model pretrained on ImageNet!')
                    backbone = resnet50(num_classes=1000)
                    print('Loading the pretrained model from the weights {%s}!' % model_urls[args.arch])
                    backbone.load_state_dict(model_zoo.load_url(model_urls[args.arch]))

                # if modality_type == 'audio':
                #     backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)

                for param in backbone.parameters():
                    param.requires_grad = True

            # elif args.pretrain == 'None':
            #     backbone = resnet50(num_classes=1000)
            #     if modality_type == 'audio':
            #         backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)

            backbone.fc = nn.Linear(2048, args.num_classes)


        elif args.arch == 'resnet101':
            backbone = resnet101(num_classes=1000)
            print('Loading the pretrained model from the weights {%s}!' % model_urls[args.arch])
            backbone.load_state_dict(model_zoo.load_url(model_urls[args.arch]))
            if modality_type == 'audio':
                backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
            backbone.fc = nn.Linear(2048, args.num_classes)

        elif args.arch == 'resnet18':
            backbone = resnet18(num_classes=1000)
            print('Loading the pretrained model from the weights {%s}!' % model_urls[args.arch])
            backbone.load_state_dict(model_zoo.load_url(model_urls[args.arch]))
            if modality_type == 'audio':
                backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
            print('backbone.fc:', backbone.fc)
            backbone.fc = nn.Linear(512, args.num_classes)


        backbone.cuda()
        if args.arch == 'resnet50':
            self.backbone = backbone
            self.afm = AttentionFusionModule(3072, 1024)
            self.conv_head32 = ConvBNReLU(2048, 1024, kernel_size=3, stride=1, padding=1)
            self.conv_head16 = ConvBNReLU(1024, 1024, kernel_size=3, stride=1, padding=1)

        elif args.arch == 'resnet18':
            self.backbone = backbone
            self.afm = AttentionFusionModule(1024, 512)
            self.conv_head32 = ConvBNReLU(512, 512, kernel_size=3, stride=1, padding=1)
            self.conv_head16 = ConvBNReLU(256, 512, kernel_size=3, stride=1, padding=1)

        elif args.arch == 'resnet101':
            self.backbone = backbone
            self.afm = AttentionFusionModule(3072, 1024)
            self.conv_head32 = ConvBNReLU(2048, 1024, kernel_size=3, stride=1, padding=1)
            self.conv_head16 = ConvBNReLU(1024, 1024, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        resnet_output, feat8, feat16, feat32, _ = self.backbone(x)
        # print('feat8.shape:', feat8.shape)
        # print('feat16.shape:', feat16.shape)
        # print('feat32.shape:', feat32.shape)

        h8, w8 = feat8.size()[2:]
        h16, w16 = feat16.size()[2:]

        # Attention Fusion Module
        # print('feat16.shape:', feat16.shape)
        feat16 = self.conv_head16(feat16)       # C: 1024->1024
        # print('feat16.shape:', feat16.shape)
        # print('feat32.shape:', feat32.shape)

        atten = self.afm(feat16, feat32)        # C: 1024, 2048->3072->1024
        # print('atten.shape:', atten.shape)

        feat32 = self.conv_head32(feat32)       # C: 2048->1024
        # print('feat32.shape:', feat32.shape)

        feat32 = torch.mul(feat32, atten)       # C: 1024->1024
        # print('feat32.shape:', feat32.shape)

        feat32_up = F.interpolate(feat32, (h16, w16), mode='nearest')
        # print('feat32_up.shape:', feat32_up.shape)

        feat16 = torch.mul(feat16, (1 - atten))
        # print('feat16.shape:', feat16.shape)

        feat16_sum = feat16 + feat32_up
        # print('feat16_sum.shape:', feat16_sum.shape)

        # feature smoothness
        # feat16_sum = self.conv_head1(feat16_sum)
        # print('feat16_sum.shape:', feat16_sum.shape)
        #
        # # Strip Attention Module
        # feat16_sum = self.sam(feat16_sum)
        # print('feat16_sum.shape:', feat16_sum.shape)
        # feat16_up = F.interpolate(feat16_sum, (h8, w8), mode='nearest')
        # print('feat16_up.shape:', feat16_up.shape)
        # feat16_up = self.conv_head2(feat16_up)
        # print('feat16_up.shape:', feat16_up.shape)
        return feat16_sum, resnet_output





class FeaturePyramidAttentionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeaturePyramidAttentionModule, self).__init__()
        self.conv1 = ConvBNReLU(in_chan, int(out_chan/2), kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBNReLU(in_chan, int(out_chan/2), kernel_size=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        self.conv3 = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1, bias=False)

        self.bn3 = BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, Fm1, Fm2, Fm3):
        # Fm1_down = F.interpolate(Fm1, scale_factor=0.5)
        Fm1_down = F.interpolate(Fm1, Fm2.size()[2:], mode='nearest')

        Fm3_up = F.interpolate(Fm3, Fm2.size()[2:], mode='nearest')

        fcat = torch.cat([Fm2, Fm3_up, Fm1_down], dim=1)
        # dimension reduction
        # print('fcat.shape:', fcat.shape)
        feat_3_3 = self.conv1(fcat)
        feat_1_1 = self.conv2(fcat)
        # print('feat_3_3.shape:', feat_3_3.size(), feat_3_3.size()[2:])

        feat = torch.cat([feat_1_1, feat_3_3], dim=1)
        # print('feat.shape:', feat.shape)

        atten = F.avg_pool2d(feat, feat.size()[2:])
        # print('atten.shape:', atten.shape)

        atten_max = self.maxpool(atten)
        # print('atten_max.shape:', atten.shape)

        atten_avg = self.avgpool(atten)
        # print('atten_avg.shape:', atten.shape)
        atten = torch.add(atten_avg, atten_max)
        # print('atten.shape:', atten.shape)

        atten = self.bn3(atten)
        # print('atten.shape:', atten.shape)

        atten = self.sigmoid_atten(atten)
        # print('atten.shape:', atten.shape)

        return atten

class SpatialAttention(nn.Module):
    def __init__(self, in_fea_dim=1024, in_channels=4, out_channels=1, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(in_fea_dim, 1, 3, padding=1, bias=False)     # 3x3 conv, padding=1
        self.conv2 = nn.Conv2d(in_fea_dim, 1, 1, padding=0, bias=False)     # 1x1 conv, padding=0
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print('x.shape:', x.shape)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # print('avg_out.shape:', avg_out.shape)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # print('max_out.shape:', max_out.shape)
        out_1_1 = self.conv1(x)
        # print('out_1_1.shape:', out_1_1.shape)
        out_3_3 = self.conv2(x)
        # print('out_3_3.shape:', out_3_3.shape)

        x = torch.cat([avg_out, max_out, out_1_1, out_3_3], dim=1)
        # print('x.shape:', x.shape)

        x = self.conv3(x)
        # print('x.shape:', x.shape)

        return self.sigmoid(x)


# for ESC
class Backbone_ESC(nn.Module):
    def __init__(self, args):
        super(Backbone_ESC, self).__init__()
        if args.arch == 'resnet50':
            if args.pretrain == 'AudioSet':
                # create the model for classification
                backbone = resnet50(num_classes=527)
                backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
                state = torch.load(args.audio_net_weights)['model']
                print('Loading the pretrained model from the weights {%s}!' % args.audio_net_weights)
                backbone.load_state_dict(state)
                backbone.fc = nn.Linear(2048, args.num_classes)

            elif args.pretrain == 'ImageNet':
                backbone = resnet50(num_classes=1000)

                print('Loading the pretrained model from the weights {%s}!' % model_urls[args.arch])
                backbone.load_state_dict(model_zoo.load_url(model_urls[args.arch]))
                if args.dataset_name != 'DCASE2021-Visual':
                    backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        
            elif args.pretrain == 'Places365':
                backbone = resnet50(num_classes=365)
                print('use the model pretrained on Places365!')
                weights_path = 'weights/resnet50_places365.pth.tar'
                # state = torch.load(weights_path)['state_dict']
                # backbone.load_state_dict(state)

                import os
                if os.path.exists(weights_path):
                    prop_saved = torch.load(weights_path)['state_dict']
                    from collections import OrderedDict
                    prop_selected = OrderedDict()
                    for k, v in prop_saved.items():
                        name = k[7:]  # remove `module.`
                        prop_selected[name] = v
                    backbone.load_state_dict(prop_selected)
                    
                    if args.dataset_name == 'ADVANCE-Audio' or args.dataset_name =='DCASE2021-Audio':
                        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)

                    print('Model loaded.')
                else:
                    print("No model detected.")

                backbone.fc = nn.Linear(2048, args.num_classes)

                # backbone.load_state_dict(model_zoo.load_url(model_urls[args.arch]))
                print('Loading the pretrained model from the weights {%s}!' % weights_path)

            elif args.pretrain == 'None':
                backbone = resnet50(num_classes=1000)
                backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
                
            backbone.fc = nn.Linear(2048, args.num_classes)

        elif args.arch == 'resnet101':
            backbone = resnet101(num_classes=1000)
            print('Loading the pretrained model from the weights {%s}!' % model_urls[args.arch])
            backbone.load_state_dict(model_zoo.load_url(model_urls[args.arch]))

            backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
            backbone.fc = nn.Linear(2048, args.num_classes)

        # require gradient
        for param in backbone.parameters():
            param.requires_grad = True

        # to cuda
        backbone.cuda()
        self.backbone = backbone

    def forward(self, x):
        _, _, _, _, resnet_output = self.backbone(x)
        return resnet_output
    

class FPA_Network(nn.Module):
    def __init__(self, args):
        super(FPA_Network, self).__init__()
        self.args = args
        if args.arch == 'resnet50':
            glob_fea_dim = 1024
            self.conv_Fm1 = ConvBNReLU(512, glob_fea_dim, ks=3, stride=1, padding=1)
            self.conv_Fm2 = ConvBNReLU(1024, glob_fea_dim, ks=3, stride=1, padding=1)
            self.conv_Fm3 = ConvBNReLU(2048, glob_fea_dim, ks=3, stride=1, padding=1)
            self.fpam = FeaturePyramidAttentionModule(in_chan=3072, out_chan=glob_fea_dim)
            self.bn_fpam = BatchNorm2d(glob_fea_dim)
            self.sa1 = SpatialAttention(in_fea_dim=glob_fea_dim)
            self.sa2 = SpatialAttention(in_fea_dim=glob_fea_dim)
            self.sa3 = SpatialAttention(in_fea_dim=glob_fea_dim)

            self.relu = nn.ReLU()
            self.bn1 = BatchNorm1d(1024)
            self.bn2 = BatchNorm1d(1024)
            self.fc2 = nn.Linear(2048, 1024)

            self.fc = nn.Linear(1024, args.num_classes)

        elif args.arch == 'resnet18':
            glob_fea_dim = 256
            self.conv_Fm1 = ConvBNReLU(128, glob_fea_dim, ks=3, stride=1, padding=1)
            self.conv_Fm2 = ConvBNReLU(256, glob_fea_dim, ks=3, stride=1, padding=1)
            self.conv_Fm3 = ConvBNReLU(512, glob_fea_dim, ks=3, stride=1, padding=1)
            self.fpam = FeaturePyramidAttentionModule(in_chan=768, out_chan=glob_fea_dim)
            self.bn_fpam = BatchNorm2d(glob_fea_dim)
            self.sa1 = SpatialAttention(in_fea_dim=glob_fea_dim)
            self.sa2 = SpatialAttention(in_fea_dim=glob_fea_dim)
            self.sa3 = SpatialAttention(in_fea_dim=glob_fea_dim)
            self.fc = nn.Linear(glob_fea_dim, args.num_classes)


        elif args.arch == 'resnet34':
            glob_fea_dim = 256
            self.conv_Fm1 = ConvBNReLU(128, glob_fea_dim, ks=3, stride=1, padding=1)
            self.conv_Fm2 = ConvBNReLU(256, glob_fea_dim, ks=3, stride=1, padding=1)
            self.conv_Fm3 = ConvBNReLU(512, glob_fea_dim, ks=3, stride=1, padding=1)
            self.fpam = FeaturePyramidAttentionModule(in_chan=glob_fea_dim * 3, out_chan=glob_fea_dim)
            self.bn_fpam = BatchNorm2d(glob_fea_dim)
            self.sa1 = SpatialAttention(in_fea_dim=glob_fea_dim)
            self.sa2 = SpatialAttention(in_fea_dim=glob_fea_dim)
            self.sa3 = SpatialAttention(in_fea_dim=glob_fea_dim)

            self.fc = nn.Linear(glob_fea_dim, args.num_classes)

        elif args.arch == 'resnet101':
            glob_fea_dim = 1024
            self.conv_Fm1 = ConvBNReLU(512, glob_fea_dim, ks=3, stride=1, padding=1)
            self.conv_Fm2 = ConvBNReLU(1024, glob_fea_dim, ks=3, stride=1, padding=1)
            self.conv_Fm3 = ConvBNReLU(2048, glob_fea_dim, ks=3, stride=1, padding=1)
            self.fpam = FeaturePyramidAttentionModule(in_chan=glob_fea_dim * 3, out_chan=glob_fea_dim)
            self.bn_fpam = BatchNorm2d(glob_fea_dim)
            self.sa1 = SpatialAttention(in_fea_dim=glob_fea_dim)
            self.sa2 = SpatialAttention(in_fea_dim=glob_fea_dim)
            self.sa3 = SpatialAttention(in_fea_dim=glob_fea_dim)

            self.fc = nn.Linear(1024, args.num_classes)

    def forward(self, Fm1, Fm2, Fm3, resnet_out):
        hFm2, wFm2 = Fm2.size()[2:]

        Fm1 = self.conv_Fm1(Fm1)
        # print('Fm1.shape:', Fm1.shape)

        Fms1 = self.sa1(Fm1)
        # print('Fms1.shape:', Fms1.shape)

        Fm1 = torch.mul(Fm1, Fms1)
        # print('Fm1.shape:', Fm1.shape)

        Fm2 = self.conv_Fm2(Fm2)  # C: 1024->1024
        Fms2 = self.sa2(Fm2)
        Fm2 = torch.mul(Fm2, Fms2)
        # print('Fm2.shape:', Fm2.shape)

        Fm3 = self.conv_Fm3(Fm3)  # C: 2048->1024
        # print('Fm3.shape:', Fm3.shape)

        Fms3 = self.sa3(Fm3)
        # print('Fms3.shape:', Fms3.shape)

        Fm3 = torch.mul(Fm3, Fms3)
        # print('Fm3.shape:', Fm3.shape)

        atten = self.fpam(Fm1, Fm2, Fm3)  # C: 1024, 1024, 1024 -> 3072 -> 1024
        # print('atten.shape:', atten.shape)

        Fm3_up = F.interpolate(Fm3, (hFm2, wFm2), mode='nearest')
        # print('Fm3_up.shape:', Fm3_up.shape)

        Fm3_up = torch.mul(Fm3_up, atten / 3)
        # print('Fm3_up.shape:', Fm3_up.shape)

        # Fm1_down = F.interpolate(Fm1, scale_factor=0.5)
        Fm1_down = F.interpolate(Fm1, (hFm2, wFm2), mode='nearest')
        # print('Fm1_down.shape:', Fm1_down.shape)

        Fm1_down = torch.mul(Fm1_down, atten / 3)
        # print('Fm1_down.shape:', Fm1_down.shape)

        Fm2 = torch.mul(Fm2, atten / 3)
        # print('Fm2.shape:', Fm2.shape)

        fpam_output = Fm2 + Fm1_down + Fm3_up
        atten_output = fpam_output

        fpam_output = self.bn_fpam(fpam_output)
        # print('fpam_output:', fpam_output.shape, torch.mean(fpam_output), torch.var(fpam_output))
        fpam_output = torch.mean(fpam_output, dim=3)
        # print('fpam_output:', fpam_output.shape, torch.mean(fpam_output), torch.var(fpam_output))
        fpam_output = torch.mean(fpam_output, dim=2)
        # print('fpam_output:', fpam_output.shape, torch.mean(fpam_output), torch.var(fpam_output))


        fpam_output = self.bn1(self.relu(fpam_output))
        # print('fpam_output:', fpam_output.shape, torch.mean(fpam_output), torch.var(fpam_output))

        if self.args.dataset_name == 'DCASE2021-Audio-Visual' or self.args.dataset_name == 'ADVANCE':
            return fpam_output, atten_output

        else:
            resnet_out = self.bn2(self.relu(self.fc2(resnet_out)))
            fpam_output = self.fc(fpam_output+resnet_out)

        return fpam_output, atten_output


# for FPAM
class FPAM_Audio_Visual(nn.Module):
    def __init__(self, args, modality_type='audio'):
        super(FPAM_Audio_Visual, self).__init__()
        # model for the audio net
        if args.dataset_name == 'ADVANCE' and modality_type == 'visual':
            print('resnet101 loading!')
            backbone = resnet101(num_classes=1000)
            backbone.fc = nn.Linear(2048, args.num_classes)

            # print('Loading the pretrained model from the weights {%s}!' % model_urls['resnet101'])
            # backbone.load_state_dict(model_zoo.load_url(model_urls['resnet101']))

            
        elif args.dataset_name == 'ADVANCE-Visual' and modality_type == 'visual':
            print('resnet101 loading!')
            backbone = resnet101(num_classes=1000)
            print('Loading the pretrained model from the weights {%s}!' % model_urls['resnet101'])
            backbone.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
            backbone.fc = nn.Linear(2048, args.num_classes)
            
        elif args.arch == 'resnet50':
            if modality_type == 'audio':
                # create the model for classification
                backbone = resnet50(num_classes=527)
                if modality_type == 'audio':
                    backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
                state = torch.load(args.audio_net_weights)['model']
                print('Loading the pretrained model from the weights {%s}!' % args.audio_net_weights)     # AUDIOSET PRETRAINED MODEL
                backbone.load_state_dict(state)
                backbone.fc = nn.Linear(2048, args.num_classes)

            elif modality_type == 'visual':
                if args.pretrain == 'Places365':
                    backbone = resnet50(num_classes=365)

                    print('use the model pretrained on Places365!')
                    weights_path = 'weights/resnet50_places365.pth.tar'
                    # state = torch.load(weights_path)['state_dict']
                    # backbone.load_state_dict(state)

                    import os
                    if os.path.exists(weights_path):
                        prop_saved = torch.load(weights_path)['state_dict']
                        from collections import OrderedDict
                        prop_selected = OrderedDict()
                        for k, v in prop_saved.items():
                            name = k[7:]  # remove `module.`
                            prop_selected[name] = v
                        backbone.load_state_dict(prop_selected)
                        print('Model loaded.')
                    else:
                        print("No model detected.")

                    backbone.fc = nn.Linear(2048, args.num_classes)

                    # backbone.load_state_dict(model_zoo.load_url(model_urls[args.arch]))
                    print('Loading the pretrained model from the weights {%s}!' % weights_path)

                elif args.pretrain == 'Places365-14':
                    backbone = resnet50(num_classes=14)
                    print('use the model pretrained on Places365-14!')

                    model_file = './weights/resnet50_places365-14.pth.tar'
                    checkpoint = torch.load(model_file)
                    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
                    backbone.load_state_dict(state_dict)
                    backbone.fc = nn.Linear(2048, args.num_classes)

                else:
                    print('use the model pretrained on ImageNet!')
                    backbone = resnet50(num_classes=1000)
                    print('Loading the pretrained model from the weights {%s}!' % model_urls[args.arch])
                    backbone.load_state_dict(model_zoo.load_url(model_urls[args.arch]))

                # backbone = resnet50(num_classes=1000)
                # print('Loading the pretrained model from the weights {%s}!' % model_urls[args.arch])
                # backbone.load_state_dict(model_zoo.load_url(model_urls[args.arch]))
                # if modality_type == 'audio':
                #     backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
                
                backbone.fc = nn.Linear(2048, args.num_classes)

            # elif args.pretrain == 'Places365':
            #     backbone = resnet50(num_classes=365)
            #     model_file = './weights/resnet50_places365.pth.tar'
            #     checkpoint = torch.load(model_file)
            #     state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            #     backbone.load_state_dict(state_dict)
            #
            #     if modality_type == 'audio':
            #         backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
            #     backbone.fc = nn.Linear(2048, args.num_classes)

            elif args.pretrain == 'None':
                backbone = resnet50(num_classes=1000)
                if modality_type == 'audio':
                    backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
                backbone.fc = nn.Linear(2048, args.num_classes)


        elif args.arch == 'resnet18':
            backbone = resnet18(num_classes=1000)
            print('Loading the pretrained model from the weights {%s}!' % model_urls[args.arch])
            backbone.load_state_dict(model_zoo.load_url(model_urls[args.arch]))
            backbone.fc = nn.Linear(512, args.num_classes)
            if modality_type == 'audio':
                backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)


        elif args.arch == 'resnet34':
            backbone = resnet34(num_classes=1000)
            # print('backbone:', backbone)
            print('Loading the pretrained model from the weights {%s}!' % model_urls[args.arch])
            backbone.load_state_dict(model_zoo.load_url(model_urls[args.arch]))
            backbone.fc = nn.Linear(512, args.num_classes)
            if modality_type == 'audio':
                backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)

            # self.maxpool = nn.MaxPool2d(4, stride=2)
            # self.conv_Fm4 = ConvBNReLU(1024, 512, ks=3, stride=2, padding=1)
        elif args.arch == 'resnet101':
            backbone = resnet101(num_classes=1000)
            print('Loading the pretrained model from the weights {%s}!' % model_urls[args.arch])
            backbone.load_state_dict(model_zoo.load_url(model_urls[args.arch]))

            if modality_type == 'audio':
                backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
            backbone.fc = nn.Linear(2048, args.num_classes)


            # self.maxpool = nn.MaxPool2d(4, stride=2)
            # self.conv_Fm4 = ConvBNReLU(1024, 512, ks=3, stride=2, padding=1)
        # require gradient
        for param in backbone.parameters():
            param.requires_grad = True

        # load pretrained model
        if args.dataset_name == 'ADVANCE' and modality_type == 'visual':    
            # print('backbone:', backbone)
            backbone_state_dict = backbone.state_dict()
            
            # for key, value in backbone_state_dict.items():
            #     # print('prev key:', key)

            new_state_dict = {}
            state = torch.load(args.visual_pretrain_weights)['audio_net_state_dict']
            for key, value in state.items():
                # print('key:', key)
                key_names = key.split('.', 1)
                head = key_names[0]
                key_name = key_names[1]

                if key_name in backbone_state_dict and head == 'backbone':
                    #print('key_name:', key_name)
                    new_state_dict.update({key_name: value})
                
            # for key, value in new_state_dict.items():
            #     print('cur key:', key)

            print('Loading the pretrained model from the weights {%s}!' % args.visual_pretrain_weights)  # AUDIOSET PRETRAINED MODEL
            backbone.load_state_dict(new_state_dict)

        if args.dataset_name == 'ADVANCE' and modality_type == 'audio':
            # print('backbone:', backbone)
            backbone_state_dict = backbone.state_dict()

            # for key, value in backbone_state_dict.items():
            #     # print('prev key:', key)

            new_state_dict = {}
            state = torch.load(args.audio_pretrain_weights)['audio_net_state_dict']
            for key, value in state.items():
                # print('key:', key)
                key_names = key.split('.', 1)
                head = key_names[0]
                key_name = key_names[1]

                if key_name in backbone_state_dict and head == 'backbone':
                    # print('key_name:', key_name)
                    new_state_dict.update({key_name: value})

            # for key, value in new_state_dict.items():
            #     print('cur key:', key)

            print(
                'Loading the pretrained model from the weights {%s}!' % args.audio_pretrain_weights)  # AUDIOSET PRETRAINED MODEL
            backbone.load_state_dict(new_state_dict)
            
        # to cuda
        backbone.cuda()
        self.backbone = backbone
        
        if args.dataset_name == 'ADVANCE' and modality_type == 'visual':
            self.fpam = FPA_Network(args)
            # print('self.fpam:', self.fpam)
            fpanet_state_dict = self.fpam.state_dict()
            # for key, value in fpanet_state_dict.items():
            #     print('prev key:', key)

            new_fpanet_state_dict = {}
            state = torch.load(args.visual_pretrain_weights)['audio_net_state_dict']
            for key, value in state.items():
                key_names = key.split('.', 1)
                head = key_names[0]
                key_name = key_names[1]

                if key_name in fpanet_state_dict and head == 'fpam':
                    new_fpanet_state_dict.update({key_name: value})

            # for key, value in new_fpanet_state_dict.items():
            #     print('cur key:', key)

            print(
                'Loading the pretrained model from the weights {%s}!' % args.visual_pretrain_weights)  # AUDIOSET PRETRAINED MODEL
            self.fpam.load_state_dict(new_fpanet_state_dict)
            self.fpam = self.fpam.cuda()

        elif args.dataset_name == 'ADVANCE' and modality_type == 'audio':
            self.fpam = FPA_Network(args)
            # print('self.fpam:', self.fpam)
            fpanet_state_dict = self.fpam.state_dict()
            # for key, value in fpanet_state_dict.items():
            #     print('prev key:', key)

            new_fpanet_state_dict = {}
            state = torch.load(args.audio_pretrain_weights)['audio_net_state_dict']

            for key, value in state.items():
                key_names = key.split('.', 1)
                head = key_names[0]
                key_name = key_names[1]

                if key_name in fpanet_state_dict and head == 'fpam':
                    new_fpanet_state_dict.update({key_name: value})

            # for key, value in new_fpanet_state_dict.items():
            #     print('cur key:', key)

            print('Loading the pretrained model from the weights {%s}!' % args.audio_pretrain_weights)  # AUDIOSET PRETRAINED MODEL
            self.fpam.load_state_dict(new_fpanet_state_dict)
            self.fpam = self.fpam.cuda()
            
        else:
            self.fpam = FPA_Network(args).cuda()

    def forward(self, x):
        # print('x.shape:', x.shape)
        resnet_output, Fm1, Fm2, Fm3, _ = self.backbone(x)
        # print('resnet_output.shape:', resnet_output.shape)

        # fpam_flops, fpam_params = thop_params_calculation(self.fpam, (Fm1, Fm2, Fm3))
        fpam_output, atten_output = self.fpam(Fm1, Fm2, Fm3, resnet_output)
        # print('fpam_output.shape:', fpam_output.shape)
        # print('atten_output.shape:', atten_output.shape)


        # add small value to output to prevent NaN
        fpam_output = torch.where(torch.isnan(fpam_output), torch.zeros_like(fpam_output), fpam_output)
        fpam_output = fpam_output + 1e-7


        return fpam_output, atten_output
    
# for ESC
class FPAM(nn.Module):
    def __init__(self, args):
        super(FPAM, self).__init__()
        if args.arch == 'resnet18':
            if args.pretrain == True:
                backbone = resnet18(num_classes=365)
                model_file = './weights/resnet18_places365.pth.tar'
                print('Loading the pretrained model from the weights {%s}!' % args.arch)
                checkpoint = torch.load(model_file)
                state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
                backbone.load_state_dict(state_dict)
            else:
                backbone = resnet18(num_classes=1000)
                print('Loading the pretrained model from the weights {%s}!' % model_urls[args.arch])
                backbone.load_state_dict(model_zoo.load_url(model_urls[args.arch]))

        elif args.arch == 'resnet50':
            if args.pretrain == True:
                backbone = resnet50(num_classes=14)
                model_file = './weights/resnet50_best_res50.pth.tar'
                print('Loading the pretrained model from the weights {%s}!' % args.arch)
                checkpoint = torch.load(model_file)
                state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
                backbone.load_state_dict(state_dict)
            else:
                backbone = resnet50(num_classes=1000)
                print('Loading the pretrained model from the weights {%s}!' % model_urls[args.arch])
                backbone.load_state_dict(model_zoo.load_url(model_urls[model_arch]))

        # model for the audio net
        else:
            if args.pretrain==True:
                # create the model for classification
                backbone = resnet50(num_classes=527)
                backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
                state = torch.load(args.audio_net_weights)['model']
                print('Loading the pretrained model from the weights {%s}!' % args.audio_net_weights)
                backbone.load_state_dict(state)
                backbone.fc = nn.Linear(2048, args.num_classes)
                
            elif args.pretrain==False:
                backbone = resnet50(num_classes=1000)
                print('Loading the pretrained model from the weights {%s}!' % model_urls[args.arch])
                backbone.load_state_dict(model_zoo.load_url(model_urls[args.arch]))
                backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)

                
        if args.arch == 'resnet50-audio':
            backbone.fc = nn.Linear(2048, args.num_classes)
            glob_fea_dim = 1024

            self.conv_Fm1 = ConvBNReLU(512, glob_fea_dim, ks=3, stride=1, padding=1)
            self.conv_Fm2 = ConvBNReLU(1024, glob_fea_dim, ks=3, stride=1, padding=1)
            self.conv_Fm3 = ConvBNReLU(2048, glob_fea_dim, ks=3, stride=1, padding=1)
            self.fpam = FeaturePyramidAttentionModule(in_chan=3072, out_chan=glob_fea_dim)
            self.bn_fpam = BatchNorm2d(glob_fea_dim)
            self.sa1 = SpatialAttention(in_fea_dim=glob_fea_dim)
            self.sa2 = SpatialAttention(in_fea_dim=glob_fea_dim)
            self.sa3 = SpatialAttention(in_fea_dim=glob_fea_dim)

        elif args.arch == 'resnet18':
            backbone.fc = nn.Linear(512, args.num_classes)

            glob_fea_dim = 256
            self.conv_Fm1 = ConvBNReLU(128, glob_fea_dim, ks=3, stride=1, padding=1)
            self.conv_Fm2 = ConvBNReLU(256, glob_fea_dim, ks=3, stride=1, padding=1)
            self.conv_Fm3 = ConvBNReLU(512, glob_fea_dim, ks=3, stride=1, padding=1)
            self.fpam = FeaturePyramidAttentionModule(in_chan=768, out_chan=glob_fea_dim)
            self.bn_fpam = BatchNorm2d(glob_fea_dim)
            self.sa1 = SpatialAttention(in_fea_dim=glob_fea_dim)
            self.sa2 = SpatialAttention(in_fea_dim=glob_fea_dim)
            self.sa3 = SpatialAttention(in_fea_dim=glob_fea_dim)

        

        # require gradient
        for param in backbone.parameters():
            param.requires_grad = True
        # to cuda
        backbone.cuda()
        self.backbone = backbone


    def forward(self, x):
        # print("input x:", x.shape)
        resnet_output, Fm1, Fm2, Fm3, _ = self.backbone(x)
        # print('Fm1:', Fm1.shape, 'Fm2:', Fm2.shape, 'Fm3:', Fm3.shape, 'resnet_output:', resnet_output.shape)
        hFm2, wFm2 = Fm2.size()[2:]

        Fm1 = self.conv_Fm1(Fm1)
        # print('Fm1.shape:', Fm1.shape)

        Fms1 = self.sa1(Fm1)
        # print('Fms1.shape:', Fms1.shape)

        Fm1 = torch.mul(Fm1, Fms1)
        # print('Fm1.shape:', Fm1.shape)

        Fm2 = self.conv_Fm2(Fm2)       # C: 1024->1024
        Fms2 = self.sa2(Fm2)
        Fm2 = torch.mul(Fm2, Fms2)
        # print('Fm2.shape:', Fm2.shape)

        Fm3 = self.conv_Fm3(Fm3)       # C: 2048->1024
        # print('Fm3.shape:', Fm3.shape)

        Fms3 = self.sa3(Fm3)
        # print('Fms3.shape:', Fms3.shape)

        Fm3 = torch.mul(Fm3, Fms3)
        # print('Fm3.shape:', Fm3.shape)

        atten = self.fpam(Fm1, Fm2, Fm3)        # C: 1024, 1024, 1024 -> 3072 -> 1024
        # print('atten.shape:', atten.shape)

        Fm3_up = F.interpolate(Fm3, (hFm2, wFm2), mode='nearest')
        # print('Fm3_up.shape:', Fm3_up.shape)

        Fm3_up = torch.mul(Fm3_up, atten/3)
        # print('Fm3_up.shape:', Fm3_up.shape)

        # Fm1_down = F.interpolate(Fm1, scale_factor=0.5)
        Fm1_down = F.interpolate(Fm1, (hFm2, wFm2), mode='nearest')
        # print('Fm1_down.shape:', Fm1_down.shape)

        Fm1_down = torch.mul(Fm1_down, atten/3)
        # print('Fm1_down.shape:', Fm1_down.shape)

        Fm2 = torch.mul(Fm2, atten/3)
        # print('Fm2.shape:', Fm2.shape)

        fpam_output = Fm2 + Fm1_down + Fm3_up
        fpam_output = self.bn_fpam(fpam_output)
        # print('fpam_output:', fpam_output.shape, torch.mean(fpam_output), torch.var(fpam_output))

        return fpam_output, resnet_output

class GCN_audio_fea(nn.Module):

    def __init__(self):
        super(GCN_audio_fea, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1, bias=True)
        self.relu = nn.ReLU()
        # self.conv2 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, bias=True)
        self.K = 16

        self.fc1 = nn.Linear(4096, 512)

    def forward(self, sounds):

        sound_graph = audio_forward(self.K, sounds)

        outs = []
        for i in range(sound_graph.shape[0]):
            sound = sound_graph[i]

            # img->16x1024x1 to x->16x1024x1x1
            x = sound.view(sound.shape[0], sound.shape[1], 1, 1)

            # after conv1 and relu, x-> 16x256x1x1
            x = self.relu(self.conv1(x))
            x = x.view(x.shape[0], x.shape[1])
            # print('x:', x.shape)
            # print('self.norm:', self.Lnorm.shape)

            y = torch.mm(self.Lnorm, x)
            # print('y:', y.shape)

            y = y.view(y.shape[0] * y.shape[1])
            # print('y.shape:', y.shape)

            out = self.fc1(y)

            outs.append(out)
        results = torch.stack(outs)
        # print('results:', results.shape)

        return results

class GCN_audio(nn.Module):

    def __init__(self, num_classes=50):
        super(GCN_audio, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        # self.conv2 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, bias=True)
        self.K = 16

        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, self.num_classes)
        self.softmax = nn.Softmax()

    def forward(self, sounds):
        outs = []

        # print('sounds:', len(sounds))
        sound_graph = audio_forward(self.K, sounds)
        # print('sound_graph:', sound_graph.shape)

        for i in range(sound_graph.shape[0]):

            sound = sound_graph[i]

            # img->16x1024x1 to x->16x1024x1x1
            x = sound.view(sound.shape[0], sound.shape[1], 1, 1)
            # print('x:', x.shape)

            # after conv1 and relu, x-> 16x256x1x1
            x = self.relu(self.conv1(x))
            x = x.view(x.shape[0], x.shape[1])
            # print('x:', x.shape)
            # print('self.norm:', self.Lnorm.shape)

            y = torch.mm(self.Lnorm, x)
            # print('y:', y.shape)

            y = y.view(y.shape[0]*y.shape[1])
            # print('y.shape:', y.shape)
            out = self.fc1(y)
            out = self.relu(out)
            out = self.dropout(out)

            out = self.fc2(out)
            outs.append(out)

        results = torch.stack(outs)
        # print('results:', results.shape)

        return results

# audio_gcn_med.py
class GCN_audio_top_med(nn.Module):

    def __init__(self, args):
        super(GCN_audio_top_med, self).__init__()
        self.nodes_num = args.nodes_num
        self.batch_size = args.batch_size
        self.num_classes = args.num_classes

        if(args.dataset_name == 'Places365-7'):
            self.conv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, bias=True)
            self.conv2 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, bias=True)

        elif(args.dataset_name == 'Places365-14'):
            self.conv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, bias=True)
            self.conv2 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, bias=True)

        elif(args.dataset_name == 'MIT67'):
            self.conv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, bias=True)
            self.conv2 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, bias=True)

        # self.conv2 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, bias=True)

        if (self.nodes_num==16):
            self.fc1 = nn.Linear(8192, 1024)
        elif (self.nodes_num==4):
            self.fc1 = nn.Linear(2048, 1024)
        elif (self.nodes_num==8):
            self.fc1 = nn.Linear(4096, 1024)
        elif (self.nodes_num==12):
            self.fc1 = nn.Linear(6144, 1024)
        elif (self.nodes_num==20):
            self.fc1 = nn.Linear(10240, 1024)
        elif (self.nodes_num==24):
            self.fc1 = nn.Linear(12288, 1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, self.num_classes)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(self.num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.softmax = nn.Softmax()

        self.graph_construction = Graph_Init(self.nodes_num, self.batch_size)


    def forward(self, images):
        # images->16,1024,28,28
        sound_graph, Lnormtop, Lnormmed, rows, columns = img_med_forward(self.nodes_num, images, self.graph_construction)
        # sound_graph-> 16,32,1024,1
        # print('sound_graph.shape:', sound_graph.shape)
        # print('Lnormtop.shape:', Lnormtop.shape)
        # print('Lnormmed.shape:', Lnormmed.shape)

        # sound_graph = self.MHA(sound_graph)
        # print('sound_graph.shape after MHA:', sound_graph.shape)


        # for each image, construct a graph
        sound = sound_graph[:,0:self.nodes_num,:,:]
        # print('sound.shape:', sound.shape)
        sound_m = sound_graph[:, self.nodes_num:2*self.nodes_num, :, :]
        # print('sound_m.shape:', sound_m.shape)

        # img->16 x nodes_num x 1024x1 to x-> (16xnodes_num) x 1024 x 1 x 1
        x = sound.reshape(sound.shape[0]*sound.shape[1], sound.shape[2], 1, 1)
        x_m = sound_m.reshape(sound_m.shape[0]*sound_m.shape[1], sound.shape[2], 1, 1)
        # print('x:', x.shape)
        # print('x_m:', x_m.shape)

        # after conv1 and relu, x-> 16x256x1x1
        x = self.relu((self.conv1(x)))
        # print('x:', x.shape)
        # remove 1x1, obtain x-> 16x256
        x = x.view(images.shape[0], int(x.shape[0]/images.shape[0]), x.shape[1])
        # print('x:', x.shape)

        x_m = self.relu((self.conv2(x_m)))
        # print('x_m:', x_m.shape)
        x_m = x_m.view(images.shape[0], int(x_m.shape[0]/images.shape[0]), x_m.shape[1])
        # print('x_m:', x_m.shape)

        graph_fusion_lst = []
        for i in range(sound_graph.shape[0]):
            y = torch.mm(Lnormtop[i], x[i])
            y_m = torch.mm(Lnormmed[i], x_m[i])
            # print('y.shape:', y[i].shape)
            # print('y_m.shape:', y_m[i].shape)
            y = y.view(y.shape[0]*y.shape[1])
            y_m = y_m.view(y_m.shape[0] * y_m.shape[1])
            # print('y.shape:', y[i].shape)
            # print('y_m.shape:', y_m[i].shape)
            graph_fusion = torch.cat((y, y_m), 0)
            # print('graph_fusion.shape:', graph_fusion.shape)
            graph_fusion_lst.append(graph_fusion)

        out = torch.stack(graph_fusion_lst)

        # print('out.shape:', out.shape)
        out = self.dropout(self.relu(self.bn1(self.fc1(out))))
        out = self.dropout(self.relu(self.bn2(self.fc2(out))))

        out = self.fc3(out)
        out = self.bn3(out)
        out = self.softmax(out)
        # print('out.shape:', out.shape)

        return out, rows, columns

class FPAGCN(nn.Module):
    
    def __init__(self, args):
        super(FPAGCN, self).__init__()
        # Attention Model Initialization
        if (args.atten_type == 'afm'):
            FPAM_net = AFM(args)

        elif (args.atten_type == 'fpam'):
            FPAM_net = FPAM(args)

        # single model
        if (args.fusion == False):
            gcn_max_med_model = GCN_audio_top_med(args)
        # fusion model
        else:
            gcn_max_med_model = GCN_max_med_fusion(args)

        for param in FPAM_net.parameters():
            param.requires_grad = True
        for param in gcn_max_med_model.parameters():
            param.requires_grad = True

        if args.status == 'train':
            FPAM_net = FPAM_net.cuda()
            gcn_max_med_model = gcn_max_med_model.cuda()
            
    def forward(self, images):
        fpam_output, resnet_output = FPAM_net(images)
        gcn_output, rows, columns = gcn_max_med_model(fpam_output, resnet_output)

        return gcn_output, rows, columns


class GraphAttentionModule(nn.Module):
    def __init__(self, args):
        super(GraphAttentionModule, self).__init__()

        self.nodes_num = args.nodes_num
        self.num_classes = args.num_classes
        self.graph_type = args.graph_type

        out_dim = 2048
        if self.graph_type == 'cag' or self.graph_type == 'sag':
            expert_num = 2
        else:
            expert_num = 4

        if (self.nodes_num == 4):
            # self.fc1 = nn.Linear(1024, out_dim)
            self.fc1 = nn.ModuleList(
                [nn.Linear(1024, out_dim) for _ in range(expert_num)]
            )

        elif (self.nodes_num == 8):
            self.fc1 = nn.ModuleList(
                [nn.Linear(2048, out_dim) for _ in range(expert_num)]
            )

        elif (self.nodes_num == 12):
            self.fc1 = nn.ModuleList(
                [nn.Linear(3072, out_dim) for _ in range(expert_num)]
            )

        elif (self.nodes_num == 16):
            self.fc1 = nn.ModuleList(
                [nn.Linear(4096, out_dim)  for _ in range(expert_num)]
            )

        elif (self.nodes_num == 20):
            self.fc1 = nn.ModuleList(
                [nn.Linear(5120, out_dim)  for _ in range(expert_num)]
            )

        elif (self.nodes_num == 24):
            self.fc1 = nn.ModuleList(
                [nn.Linear(6144, out_dim) for _ in range(expert_num)]
            )

        self.bn1 = nn.ModuleList(
            [nn.BatchNorm1d(2048) for _ in range(expert_num)]
        )

        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, self.num_classes)
        self.bn3 = nn.BatchNorm1d(self.num_classes)
        self.relu = nn.ReLU()

        self.sigmoid_atten = nn.Sigmoid()


    def forward(self, audio_max_graph, audio_med_graph, audio_resnet_output, visual_max_graph, visual_med_graph, visual_resnet_output):

        if (self.graph_type == 'sag'):
            audio_max_out = self.relu(self.bn1[0](self.fc1[0](audio_max_graph)))
            visual_max_out = self.relu(self.bn1[1](self.fc1[1](visual_max_graph)))
            sag = torch.add(audio_max_out, visual_max_out)
            fused_out = self.relu(self.bn2(self.fc2(sag)))
            fused_out = self.bn3(self.fc3(fused_out))

            return fused_out

        elif (self.graph_type == 'cag'):
            audio_med_out = self.relu(self.bn1[0](self.fc1[0](audio_med_graph)))
            visual_med_out = self.relu(self.bn1[1](self.fc1[1](visual_med_graph)))

            cag = torch.add(audio_med_out, visual_med_out)
            fused_out = self.relu(self.bn2(self.fc2(cag)))
            fused_out = self.bn3(self.fc3(fused_out))

            return fused_out

        elif (self.graph_type == 'fusion'):
            audio_max_out = self.relu(self.bn1[0](self.fc1[0](audio_max_graph)))
            visual_max_out = self.relu(self.bn1[2](self.fc1[2](visual_max_graph)))

            audio_med_out = self.relu(self.bn1[1](self.fc1[1](audio_med_graph)))
            visual_med_out = self.relu(self.bn1[3](self.fc1[3](visual_med_graph)))

            audio_out = torch.add(audio_max_out, audio_med_out)
            visual_out = torch.add(visual_max_out, visual_med_out)
            graph_out = torch.add(audio_out, visual_out)
            
            resent_out = torch.add(audio_resnet_output, visual_resnet_output)
            
            fused_out = torch.add(graph_out, resent_out)
            fused_out = self.sigmoid_atten(fused_out)
            fused_out = self.relu(self.bn2(self.fc2(fused_out)))
            fused_out = self.bn3(self.fc3(fused_out))

            return fused_out












# class AGCN(nn.Module):
#
#     def __init__(self, args):
#         super(AGCN, self).__init__()
#         # Attention Model Initialization
#         if (args.atten_type == 'afm'):
#             FPAM_net = AFM(args)
#
#         elif (args.atten_type == 'fpam'):
#             FPAM_net = FPAM(args)
#
#         # single model
#         if (args.fusion == False):
#             gcn_max_med_model = GCN_audio_top_med(args)
#         # fusion model
#         else:
#             gcn_max_med_model = GCN_max_med_fusion(args)
#
#         for param in FPAM_net.parameters():
#             param.requires_grad = True
#
#         for param in gcn_max_med_model.parameters():
#             param.requires_grad = True
#
#         if args.status == 'train':
#             FPAM_net = FPAM_net.cuda()
#             gcn_max_med_model = gcn_max_med_model.cuda()
#
#     def forward(self, images):
#         fpam_output, resnet_output = FPAM_net(images)
#         gcn_output, rows, columns = gcn_max_med_model(fpam_output, resnet_output)
#
#         return gcn_output, rows, columns


# for ESC
class AGCN_max_med_fusion(nn.Module):

    def __init__(self, args):
        super(AGCN_max_med_fusion, self).__init__()
        self.nodes_num = args.nodes_num
        self.batch_size = args.batch_size
        self.num_classes = args.num_classes
        self.graph_construction = Graph_Init(self.nodes_num, self.batch_size)
        self.graph_type = args.graph_type

        if args.arch == 'resnet50':
            self.convBnRelu_max = ConvBNReLU(1024, 256, kernel_size=1, stride=1, padding=0)
            self.convBnRelu_med = ConvBNReLU(1024, 256, kernel_size=1, stride=1, padding=0)

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.25)
            out_dim = 2048

            if (self.nodes_num == 4):
                self.fc1 = nn.Linear(1024, out_dim)
            elif (self.nodes_num == 8):
                self.fc1 = nn.Linear(2048, out_dim)
            elif (self.nodes_num == 12):
                self.fc1 = nn.Linear(3072, out_dim)
            elif (self.nodes_num == 16):
                self.fc1 = nn.Linear(4096, out_dim)
            elif (self.nodes_num == 20):
                self.fc1 = nn.Linear(5120, out_dim)
            elif (self.nodes_num == 24):
                self.fc1 = nn.Linear(6144, out_dim)

            self.bn1 = nn.BatchNorm1d(2048)

            self.fc2 = nn.Linear(2048, 1024)
            self.bn2 = nn.BatchNorm1d(1024)
            self.fc3 = nn.Linear(1024, self.num_classes)
            self.bn3 = nn.BatchNorm1d(self.num_classes)

        elif args.arch == 'resnet101':
            self.convBnRelu_max = ConvBNReLU(1024, 256, kernel_size=1, stride=1, padding=0)
            self.convBnRelu_med = ConvBNReLU(1024, 256, kernel_size=1, stride=1, padding=0)

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.25)
            out_dim = 2048

            if (self.nodes_num == 4):
                self.fc1 = nn.Linear(1024, out_dim)
            elif (self.nodes_num == 8):
                self.fc1 = nn.Linear(2048, out_dim)
            elif (self.nodes_num == 12):
                self.fc1 = nn.Linear(3072, out_dim)
            elif (self.nodes_num == 16):
                self.fc1 = nn.Linear(4096, out_dim)
            elif (self.nodes_num == 20):
                self.fc1 = nn.Linear(5120, out_dim)
            elif (self.nodes_num == 24):
                self.fc1 = nn.Linear(6144, out_dim)

            self.bn1 = nn.BatchNorm1d(2048)

            self.fc2 = nn.Linear(2048, 1024)
            self.bn2 = nn.BatchNorm1d(1024)
            self.fc3 = nn.Linear(1024, self.num_classes)
            self.bn3 = nn.BatchNorm1d(self.num_classes)


        elif args.arch == 'resnet18':
            glob_fea_dim = 512
            out_dim = 512

            self.convBnRelu_max = ConvBNReLU(glob_fea_dim, 256, kernel_size=1, stride=1, padding=0)
            self.convBnRelu_med = ConvBNReLU(glob_fea_dim, 256, kernel_size=1, stride=1, padding=0)

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.25)

            if (self.nodes_num == 4):
                self.fc1 = nn.Linear(1024, out_dim)
            elif (self.nodes_num == 8):
                self.fc1 = nn.Linear(2048, out_dim)
            elif (self.nodes_num == 12):
                self.fc1 = nn.Linear(3072, out_dim)
            elif (self.nodes_num == 16):
                self.fc1 = nn.Linear(4096, out_dim)
            elif (self.nodes_num == 20):
                self.fc1 = nn.Linear(5120, out_dim)
            elif (self.nodes_num == 24):
                self.fc1 = nn.Linear(6144, out_dim)

            self.bn1 = nn.BatchNorm1d(out_dim)

            self.fc2 = nn.Linear(512, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.fc3 = nn.Linear(256, self.num_classes)
            self.bn3 = nn.BatchNorm1d(self.num_classes)

    def graph_max_med_construct(self, nodes_num, Feature, graph_construction):
        N = Feature.shape[0]  # batch size
        W = Feature.shape[2]
        H = Feature.shape[3]

        # acquire Fre (N,H,W) from Frgb (N,C,H,W)
        Fre = torch.sum(Feature, dim=1)
        # print('Fre.shape:', Fre.shape, Fre)

        # Acquire the top K index of Fre
        feature = Fre.view(N, H * W)
        # print('feature:', feature.shape, feature)

        sorted, indices = torch.sort(feature, descending=True)
        # print('sorted:', sorted.shape, sorted)
        # print('indices:', indices.shape, indices)

        maxK_index = indices[:, 0:nodes_num]
        medK_index = indices[:, (H * W // 2 - nodes_num // 2 - 1):(H * W // 2 - nodes_num // 2 - 1 + nodes_num)]
        # print('topK_index:', topK_index.shape)
        # print('avg_index:', avgK_index.shape)

        scene_index = torch.cat([maxK_index, medK_index], dim=1)
        # print('scene_index:', scene_index.shape, scene_index)

        rows = scene_index // H
        columns = scene_index % H

        nodes = []
        graph_batch = []

        # shape of topK_index is (N, nodes_num*2)
        for i in range(N):  # lop from one image to other
            for j in range(int(rows.shape[1])):  # loop from one node to other
                # switch the feature from 1x1024 to 1024x1
                node = Feature[i, :, rows[i, j], columns[i, j]].reshape(-1, 1)
                nodes.append(node)

            # convert the list to torch tensor
            graph = torch.stack(nodes)
            # clear nodes
            nodes = []
            # save the torch tensor to list
            graph_batch.append(graph)

        graph_batch = torch.stack(graph_batch)
        # graph_batch, shape of [bs, nodes_num*2, 1024, 1]
        # print('graph_batch:', graph_batch.shape, graph_batch)
        # Adjacency matrix calculation
        Lnormtop, Lnormmed = graph_construction.Dynamic_Lnorm(rows, columns)
        # print('Lnormtop:', Lnormtop.shape, Lnormtop, 'Lnormmed:', Lnormmed.shape, Lnormmed)

        return graph_batch, Lnormtop.cuda(), Lnormmed.cuda(), rows, columns

    def forward(self, fpam_output, resnet_output):
        scene_graph, Lnormtop, Lnormmed, rows, columns = self.graph_max_med_construct(self.nodes_num, fpam_output,
                                                                                      self.graph_construction)
        # print('scene_graph.shape:', scene_graph.shape)

        scene_max_graph = scene_graph[:, 0:self.nodes_num]
        # print('scene_max_graph.shape:', scene_max_graph.shape)

        scene_med_graph = scene_graph[:, self.nodes_num:2 * self.nodes_num]
        # print('scene_med_graph.shape:', scene_med_graph.shape)

        # img->16x1024x1 to x->16x1024x1x1
        x_max = scene_max_graph.reshape(scene_max_graph.shape[0] * scene_max_graph.shape[1], scene_max_graph.shape[2],
                                        1, 1)
        x_med = scene_med_graph.reshape(scene_med_graph.shape[0] * scene_med_graph.shape[1], scene_med_graph.shape[2],
                                        1, 1)
        # print('x_max:', x_max.shape)
        # print('x_med:', x_med.shape)

        # after conv1 and relu, x-> 16x256x1x1
        x_max = self.convBnRelu_max(x_max)
        x_max = x_max.view(fpam_output.shape[0], int(x_max.shape[0] / fpam_output.shape[0]), x_max.shape[1])
        # print('x_max:', x_max.shape)

        x_med = self.convBnRelu_med(x_med)
        x_med = x_med.view(fpam_output.shape[0], int(x_med.shape[0] / fpam_output.shape[0]), x_med.shape[1])
        # print('x_med:', x_med.shape)

        # graph_fusion_lst = []
        max_graph_lst = []
        med_graph_lst = []
        for i in range(scene_graph.shape[0]):
            y_max = torch.mm(Lnormtop[i], x_max[i])
            y_med = torch.mm(Lnormmed[i], x_med[i])
            # print('y_max.shape:', y_max.shape)
            # print('y_med.shape:', y_med.shape)

            y_max = y_max.view(y_max.shape[0] * y_max.shape[1])
            y_med = y_med.view(y_med.shape[0] * y_med.shape[1])
            # print('reshape y_max.shape:', y_max.shape)
            # print('reshape y_med.shape:', y_med.shape)

            max_graph_lst.append(y_max)
            med_graph_lst.append(y_med)

            # if (self.graph_type == 'cag'):
            #     graph_fusion = y_med
            #
            # elif (self.graph_type == 'sag'):
            #     graph_fusion = y_max
            #
            # elif (self.graph_type == 'fusion'):
            #     graph_fusion = (y_max + y_med) / 2

            # graph_fusion_lst.append(graph_fusion)

        max_graph = torch.stack(max_graph_lst)
        med_graph = torch.stack(med_graph_lst)

        if self.graph_type == 'sag':
            return max_graph, None, rows, columns
        elif self.graph_type == 'cag':
            return None, med_graph, rows, columns
        elif self.graph_type == 'fusion':
            return max_graph, med_graph, rows, columns

        # out = self.relu(self.bn1(self.fc1(max_graph)))

        # fused_out = (out + resnet_output) / 2
        # print('fused_out:', fused_out.shape, fused_out)

        # fused_out = self.dropout(self.relu(self.bn2(self.fc2(fused_out))))
        # print('fused_out:', fused_out.shape, fused_out)

        #fused_out = self.fc3(fused_out)
        # print('fused_out:', fused_out.shape, fused_out)

        #fused_out = self.bn3(fused_out)
        # print('fused_out:', fused_out.shape, fused_out)

        # return max_graph, med_graph, rows, columns

#
class FPAM_Fusion_Net(nn.Module):
    def __init__(self, args):
        super(FPAM_Fusion_Net, self).__init__()

        self.num_classes = args.num_classes
        self.fc1 = nn.Linear(2048, 1024)
        self.bn = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(1024, self.num_classes)


    def forward(self, audio_fpam_output, visual_fpam_output):
        # print('audio_fpam_output:', audio_fpam_output.shape)
        # print('visual_fpam_output:', visual_fpam_output.shape)
        concat_rep = torch.cat((audio_fpam_output, visual_fpam_output), dim = 1)
        # print('concat_rep:', concat_rep.shape)
        out = self.relu(self.bn(self.fc1(concat_rep)))
        # print('out:', out.shape)

        out = self.fc2(out)
        # print('out:', out.shape)

        return out



# for ESC
class GCN_max_med_fusion(nn.Module):

    def __init__(self, args):
        super(GCN_max_med_fusion, self).__init__()
        self.nodes_num = args.nodes_num
        self.batch_size = args.batch_size
        self.num_classes = args.num_classes
        self.graph_construction = Graph_Init(self.nodes_num, self.batch_size)
        self.graph_type = args.graph_type
        
        if args.arch == 'resnet50':
            self.convBnRelu_max = ConvBNReLU(1024, 256, kernel_size=1, stride=1, padding=0)
            self.convBnRelu_med = ConvBNReLU(1024, 256, kernel_size=1, stride=1, padding=0)

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.25)
            out_dim = 2048

            if (self.nodes_num == 4):
                self.fc1 = nn.Linear(1024, out_dim)
            elif (self.nodes_num == 8):
                self.fc1 = nn.Linear(2048, out_dim)
            elif (self.nodes_num == 12):
                self.fc1 = nn.Linear(3072, out_dim)
            elif (self.nodes_num == 16):
                self.fc1 = nn.Linear(4096, out_dim)
            elif (self.nodes_num == 20):
                self.fc1 = nn.Linear(5120, out_dim)
            elif (self.nodes_num == 24):
                self.fc1 = nn.Linear(6144, out_dim)

            self.bn1 = nn.BatchNorm1d(2048)

            self.fc2 = nn.Linear(2048, 1024)
            self.bn2 = nn.BatchNorm1d(1024)
            self.fc3 = nn.Linear(1024, self.num_classes)
            self.bn3 = nn.BatchNorm1d(self.num_classes)

        elif args.arch == 'resnet101':
            self.convBnRelu_max = ConvBNReLU(1024, 256, kernel_size=1, stride=1, padding=0)
            self.convBnRelu_med = ConvBNReLU(1024, 256, kernel_size=1, stride=1, padding=0)

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.25)
            out_dim = 2048

            if (self.nodes_num == 4):
                self.fc1 = nn.Linear(1024, out_dim)
            elif (self.nodes_num == 8):
                self.fc1 = nn.Linear(2048, out_dim)
            elif (self.nodes_num == 12):
                self.fc1 = nn.Linear(3072, out_dim)
            elif (self.nodes_num == 16):
                self.fc1 = nn.Linear(4096, out_dim)
            elif (self.nodes_num == 20):
                self.fc1 = nn.Linear(5120, out_dim)
            elif (self.nodes_num == 24):
                self.fc1 = nn.Linear(6144, out_dim)

            self.bn1 = nn.BatchNorm1d(2048)

            self.fc2 = nn.Linear(2048, 1024)
            self.bn2 = nn.BatchNorm1d(1024)
            self.fc3 = nn.Linear(1024, self.num_classes)
            self.bn3 = nn.BatchNorm1d(self.num_classes)


        elif args.arch == 'resnet18':
            glob_fea_dim = 512
            out_dim = 512

            self.convBnRelu_max = ConvBNReLU(glob_fea_dim, 256, kernel_size=1, stride=1, padding=0)
            self.convBnRelu_med = ConvBNReLU(glob_fea_dim, 256, kernel_size=1, stride=1, padding=0)

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.25)

            if (self.nodes_num == 4):
                self.fc1 = nn.Linear(1024, out_dim)
            elif (self.nodes_num == 8):
                self.fc1 = nn.Linear(2048, out_dim)
            elif (self.nodes_num == 12):
                self.fc1 = nn.Linear(3072, out_dim)
            elif (self.nodes_num == 16):
                self.fc1 = nn.Linear(4096, out_dim)
            elif (self.nodes_num == 20):
                self.fc1 = nn.Linear(5120, out_dim)
            elif (self.nodes_num == 24):
                self.fc1 = nn.Linear(6144, out_dim)

            self.bn1 = nn.BatchNorm1d(out_dim)

            self.fc2 = nn.Linear(512, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.fc3 = nn.Linear(256, self.num_classes)
            self.bn3 = nn.BatchNorm1d(self.num_classes)

    def graph_max_med_construct(self, nodes_num, Feature, graph_construction):
        N = Feature.shape[0]  # batch size
        W = Feature.shape[2]
        H = Feature.shape[3]

        # acquire Fre (N,H,W) from Frgb (N,C,H,W)
        Fre = torch.sum(Feature, dim=1)
        # print('Fre.shape:', Fre.shape, Fre)

        # Acquire the top K index of Fre
        feature = Fre.view(N, H * W)
        # print('feature:', feature.shape, feature)

        sorted, indices = torch.sort(feature, descending=True)
        # print('sorted:', sorted.shape, sorted)
        # print('indices:', indices.shape, indices)

        maxK_index = indices[:, 0:nodes_num]
        medK_index = indices[:, (H * W // 2 - nodes_num // 2 - 1):(H * W // 2 - nodes_num // 2 - 1 + nodes_num)]
        # print('topK_index:', topK_index.shape)
        # print('avg_index:', avgK_index.shape)

        scene_index = torch.cat([maxK_index, medK_index], dim=1)
        # print('scene_index:', scene_index.shape, scene_index)

        rows = scene_index // H
        columns = scene_index % H

        nodes = []
        graph_batch = []

        # shape of topK_index is (N, nodes_num*2)
        for i in range(N):  # lop from one image to other
            for j in range(int(rows.shape[1])):  # loop from one node to other
                # switch the feature from 1x1024 to 1024x1
                node = Feature[i, :, rows[i, j], columns[i, j]].reshape(-1, 1)
                nodes.append(node)

            # convert the list to torch tensor
            graph = torch.stack(nodes)
            # clear nodes
            nodes = []
            # save the torch tensor to list
            graph_batch.append(graph)

        graph_batch = torch.stack(graph_batch)
        # graph_batch, shape of [bs, nodes_num*2, 1024, 1]
        # print('graph_batch:', graph_batch.shape, graph_batch)
        # Adjacency matrix calculation
        Lnormtop, Lnormmed = graph_construction.Dynamic_Lnorm(rows, columns)
        # print('Lnormtop:', Lnormtop.shape, Lnormtop, 'Lnormmed:', Lnormmed.shape, Lnormmed)

        return graph_batch, Lnormtop.cuda(), Lnormmed.cuda(), rows, columns

    def forward(self, fpam_output, resnet_output):
        scene_graph, Lnormtop, Lnormmed, rows, columns = self.graph_max_med_construct(self.nodes_num, fpam_output, self.graph_construction)
        # print('scene_graph.shape:', scene_graph.shape)

        scene_max_graph = scene_graph[:, 0:self.nodes_num]
        # print('scene_max_graph.shape:', scene_max_graph.shape)

        scene_med_graph = scene_graph[:, self.nodes_num:2 * self.nodes_num]
        # print('scene_med_graph.shape:', scene_med_graph.shape)

        # img->16x1024x1 to x->16x1024x1x1
        x_max = scene_max_graph.reshape(scene_max_graph.shape[0]*scene_max_graph.shape[1], scene_max_graph.shape[2], 1, 1)
        x_med = scene_med_graph.reshape(scene_med_graph.shape[0]*scene_med_graph.shape[1], scene_med_graph.shape[2], 1, 1)
        # print('x_max:', x_max.shape)
        # print('x_med:', x_med.shape)

        # after conv1 and relu, x-> 16x256x1x1
        x_max = self.convBnRelu_max(x_max)
        x_max = x_max.view(fpam_output.shape[0], int(x_max.shape[0]/fpam_output.shape[0]), x_max.shape[1])
        # print('x_max:', x_max.shape)

        x_med = self.convBnRelu_med(x_med)
        x_med = x_med.view(fpam_output.shape[0], int(x_med.shape[0]/fpam_output.shape[0]), x_med.shape[1])
        # print('x_med:', x_med.shape)

        graph_fusion_lst = []
        for i in range(scene_graph.shape[0]):
            y_max = torch.mm(Lnormtop[i], x_max[i])
            y_med = torch.mm(Lnormmed[i], x_med[i])
            # print('y_max.shape:', y_max.shape)
            # print('y_med.shape:', y_med.shape)

            y_max = y_max.view(y_max.shape[0] * y_max.shape[1])
            y_med = y_med.view(y_med.shape[0] * y_med.shape[1])
            # print('reshape y_max.shape:', y_max.shape)
            # print('reshape y_med.shape:', y_med.shape)

            if (self.graph_type == 'cag'):
                graph_fusion = y_med

            elif (self.graph_type == 'sag'):
                graph_fusion = y_max

            elif (self.graph_type == 'fusion'):
                graph_fusion = (y_max + y_med) / 2

            graph_fusion_lst.append(graph_fusion)

        # print("graph_fusion_lst[0].shape",graph_fusion_lst[0].shape)
        out = torch.stack(graph_fusion_lst)
        # print("out.shape", out.shape)

        out = self.relu(self.bn1(self.fc1(out)))
        # print("out.shape", out.shape)
        # print("resnet_output.shape", resnet_output.shape)



        fused_out = (out + resnet_output) / 2
        # print('fused_out:', fused_out.shape, fused_out)

        fused_out = self.dropout(self.relu(self.bn2(self.fc2(fused_out))))
        # print('fused_out:', fused_out.shape, fused_out)

        fused_out = self.fc3(fused_out)
        # print('fused_out:', fused_out.shape, fused_out)

        fused_out = self.bn3(fused_out)
        # print('fused_out:', fused_out.shape, fused_out)

        return fused_out, rows, columns

# audio_fusion_med.py
class GCN_audio_top_med_fea(nn.Module):

    def __init__(self, args):
        super(GCN_audio_top_med_fea, self).__init__()
        self.nodes_num = args.nodes_num
        self.batch_size = args.batch_size
        self.num_classes = args.num_classes

        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, bias=True)
        self.convbn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.K = args.nodes_num

        # if (self.K==16):
        #     self.fc1 = nn.Linear(8192, 1024)
        # elif (self.K==8):
        #     self.fc1 = nn.Linear(4096, 1024)
        # elif (self.K==12):
        #     self.fc1 = nn.Linear(6144, 1024)
        # elif (self.K==4):
        #     self.fc1 = nn.Linear(2048, 1024)
        # elif (self.K==20):
        #     self.fc1 = nn.Linear(10240, 1024)
        # elif (self.K==24):
        #     self.fc1 = nn.Linear(12288, 1024)

        if self.K == 4:
            self.fc1 = nn.Linear(1024, 1024)
        elif (self.K == 8):
            self.fc1 = nn.Linear(2048, 1024)
        elif (self.K == 12):
            self.fc1 = nn.Linear(3072, 1024)
        elif (self.K == 16):
            self.fc1 = nn.Linear(4096, 1024)
        elif (self.K == 20):
            self.fc1 = nn.Linear(5120, 1024)
        elif (self.K == 24):
            self.fc1 = nn.Linear(6144, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.num_classes)
        self.softmax = nn.Softmax()
        self.graph_construction = Graph_Init(self.nodes_num, self.batch_size)


    def forward(self, sounds):
        # print('imgs:', len(imgs)/self.K)
        sound_graph, Lnormtop, Lnormmed, rows, columns = img_med_forward(self.K, sounds, self.graph_construction)
        # print('sound_graph.shape:', sound_graph.shape)

        # for i in range(self .K):
        sound = sound_graph[:, 0:self.K]
        # print('sound.shape:', sound.shape)
        sound_m = sound_graph[:, self.K:2 * self.K]
        # print('sound_m.shape:', sound_m.shape)

        # img->16x1024x1 to x->16x1024x1x1
        x = sound.reshape(sound.shape[0]*sound.shape[1], sound.shape[2], 1, 1)
        x_m = sound_m.reshape(sound_m.shape[0]*sound_m.shape[1], sound_m.shape[2], 1, 1)
        # print('x:', x.shape)
        # print('x_m:', x_m.shape)

        # after conv1 and relu, x-> 16x256x1x1
        x = self.relu(self.convbn1(self.conv1(x)))
        x = x.view(sounds.shape[0], int(x.shape[0]/sounds.shape[0]), x.shape[1])
        # print('x:', x.shape)

        x_m = self.relu(self.convbn1(self.conv1(x_m)))
        x_m = x_m.view(sounds.shape[0], int(x_m.shape[0]/sounds.shape[0]), x_m.shape[1])
        # print('x_m:', x_m.shape)


        graph_fusion_lst = []
        for i in range(sound_graph.shape[0]):
            y = torch.mm(Lnormtop[i], x[i])
            y_m = torch.mm(Lnormmed[i], x_m[i])
            # print('y.shape:', y.shape)
            # print('y_m.shape:', y_m.shape)
            y = y.view(y.shape[0]*y.shape[1])
            y_m = y_m.view(y_m.shape[0] * y_m.shape[1])
            # print('y.shape:', y.shape)
            # print('y_m.shape:', y_m.shape)
            # graph_fusion = torch.add(y, y_m)
            graph_fusion = (y + y_m) / 2
            # print('graph_sum.shape:', graph_fusion.shape)
            graph_fusion_lst.append(graph_fusion)


        out = torch.stack(graph_fusion_lst)
        # print('out.shape:', out.shape, out)
        out = self.dropout(self.relu(self.bn1(self.fc1(out))))
        # print('out.shape:', out.shape, out)
        out = self.dropout(self.relu(self.bn2(self.fc2(out))))
        # print('out.shape:', out.shape, out)

        return out, rows, columns

        # # for i in range(self .K):
        # for i in range(sound_graph.shape[0]):
        #     sound = sound_graph[i][0:self.K]
        #     # print('sound.shape:', sound.shape)
        #     sound_m = sound_graph[i][self.K:2*self.K]
        #
        #     # img->16x1024x1 to x->16x1024x1x1
        #     x = sound.view(sound.shape[0], sound.shape[1], 1, 1)
        #     x_m = sound_m.view(sound_m.shape[0], sound_m.shape[1], 1, 1)
        #     print('x:', x.shape)
        #     print('x_m:', x_m.shape)
        #
        #     # after conv1 and relu, x-> 16x256x1x1
        #     x = self.relu(self.bn1(self.conv1(x)))
        #     x = x.view(x.shape[0], x.shape[1])
        #     print('x:', x.shape)
        #
        #     x_m = self.relu(self.bn1(self.conv1(x_m)))
        #     x_m = x_m.view(x_m.shape[0], x_m.shape[1])
        #     print('x_m:', x_m.shape)
        #
        #
        #     y = torch.mm(Lnormtop[i], x)
        #     y = y.view(y.shape[0]*y.shape[1])
        #     # print('y:', y.shape)
        #     # print('y.shape:', y.shape)
        #
        #     y_m = torch.mm(Lnormmed[i], x_m)
        #     y_m = y_m.view(y_m.shape[0]*y_m.shape[1])
        #
        #     y = torch.cat((y,y_m),0)
        #     # print('y.con shape:', y.shape)
        #     out = self.fc1(y)
        #     out = self.relu(out)
        #     out = self.dropout(out)
        #     out = self.fc2(out)
        #     out = self.relu(out)
        #     outs.append(out)
        #
        # results = torch.stack(outs)
        #
        # return results, rows, columns




