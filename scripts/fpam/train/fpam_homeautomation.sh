#!/usr/bin/env bash
source activate llod
#source activate swin
LR=0.01
LRI=20
EPOCHS=60
NODES_NUM=20
BATCH_SIZE=32

MODEL_TYPE='FPAM_resnet50_nomixup'
Pretrain='AudioSet'   # ImageNet, AudioSet, None
ARCH='resnet50'
atten_type='fpam_esc' #  ['fpam', 'baseline, fpam_esc']

DATASET_NAME='DCASE2019'   # ESC10, ESC50, DCASE2019, US8K

LOG_DIR="./logs/""$DATASET_NAME""/"
current_time=$(date  "+%Y%m%d-%H%M%S-")


LOG_FILE=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$Pretrain""_""$DATASET_NAME""_""$BATCH_SIZE"".txt""


echo $LOG_FILE


CUDA_VISIBLE_DEVICES=1 python train_fpam_esc.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 2 --atten_type $atten_type --nodes_num $NODES_NUM --arch $ARCH --fusion True --num_classes 10 --batch_size $BATCH_SIZE  --dataset_name $DATASET_NAME --pretrain $Pretrain >> $LOG_FILE &

