#!/usr/bin/env bash
source activate llod


BATCH_SIZE=16

MODEL_TYPE='audio_gcn_max_med'
ATTEN_TYPE='fpam_esc'
ARCH='resnet50-audio'


# 95.0, 'audio_gcn_max_med_fpam_7f_20_epoch_44_20.pth.tar'
# 93.75, 'audio_gcn_max_med_fpam_7f_20_epoch_22_20.pth.tar'
# 83.75, 'audio_gcn_max_med_fpam_7f_20_epoch_6_20.pth.tar'
# 75.0, 'audio_gcn_max_med_fpam_7f_20_epoch_4_20.pth.tar'
# 51.25, 'audio_gcn_max_med_fpam_7f_20_epoch_2_20.pth.tar'

MODEL_NAME='audio_gcn_max_med_fpam_esc_7f_24_epoch_20_20.pth.tar'

NODES_NUM=20
DATASET_NAME='ESC10'   # ESC10, ESC50, US8K
NUM_CLASSES=10

CUDA_VISIBLE_DEVICES=0 python train_fpagcn_esc.py --model_type $MODEL_TYPE --experiment_id 2 --nodes_num $NODES_NUM --model_name $MODEL_NAME --arch $ARCH --bs $BATCH_SIZE --dataset_name $DATASET_NAME --num_classes $NUM_CLASSES --atten_type $ATTEN_TYPE --fusion True --status test --pretrain
