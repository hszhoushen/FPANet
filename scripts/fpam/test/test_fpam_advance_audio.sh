#!/usr/bin/env bash
#source activate llod

LR=0.01
LRI=20
EPOCHS=60
NODES_NUM=20
BATCH_SIZE=16

MODEL_TYPE='FPAM_resnet50_nomixup_AudioSet'
Pretrain='AudioSet'   # ImageNet, AudioSet, Places365
ARCH='resnet50'
atten_type='fpam' #  ['fpam', 'baseline, fpam_esc']
DATASET_NAME='ADVANCE-Audio'   # ADVANCE-Visual
LOG_DIR="./logs/""$DATASET_NAME""/"
current_time=$(date  "+%Y%m%d-%H%M%S-")

LOG_FILE=""$LOG_DIR""test_""$current_time""$MODEL_TYPE""_""$Pretrain""_""$DATASET_NAME""_""$BATCH_SIZE"".txt""

echo $LOG_FILE

export CUDA_VISIBLE_DEVICES="3"

audio_pretrain_weights='./weights/fpam_advance_audio_pretrain.pth.tar'
visual_pretrain_weights='./weights/fpam_advance_visual_pretrain.pth.tar'


python -m torch.distributed.launch --master_port 40010 --nproc_per_node=1 train_fpam_esc.py \
       --model_type $MODEL_TYPE \
       --epochs $EPOCHS \
       --lr $LR --lri $LRI \
       --test_set_id 2 \
       --pretrain $Pretrain \
       --atten_type $atten_type \
       --nodes_num $NODES_NUM \
       --audio_pretrain_weights $audio_pretrain_weights \
       --visual_pretrain_weights $visual_pretrain_weights\
       --arch $ARCH \
       --fusion True \
       --status test \
       --model_name ADVANCE-Audio/FPAM_resnet50_nomixup_AudioSet_fpam_7f_20_epoch_60_20.pth.tar \
       --num_classes 13 \
       --batch_size $BATCH_SIZE \
       --dataset_name $DATASET_NAME >> $LOG_FILE 

