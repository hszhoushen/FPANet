#!/usr/bin/env bash
#source activate llod

LR=0.00001
LRI=20
EPOCHS=100
NODES_NUM=20
BATCH_SIZE=8

MODEL_TYPE='baseline_resnet50_nomixup_AudioSet'
Pretrain='Places365'   # ImageNet, AudioSet, Places365
ARCH='resnet50'
atten_type='baseline' #  ['fpam', 'baseline, fpam_esc']
DATASET_NAME='DCASE2021-Audio' 
LOG_DIR="./logs/""$DATASET_NAME""/"
current_time=$(date  "+%Y%m%d-%H%M%S-")

LOG_FILE=""$LOG_DIR""test_""$current_time""$MODEL_TYPE""_""$Pretrain""_""$DATASET_NAME""_""$BATCH_SIZE"".txt""

echo $LOG_FILE

export CUDA_VISIBLE_DEVICES="0"

audio_pretrain_weights='./weights/fpam_advance_audio_pretrain.pth.tar'
visual_pretrain_weights='./weights/fpam_advance_visual_pretrain.pth.tar'

python -m torch.distributed.launch --master_port 60010 --nproc_per_node=1 train_fpam_esc.py \
       --model_type $MODEL_TYPE \
       --epochs $EPOCHS \
       --lr $LR --lri $LRI \
       --test_set_id 2 \
       --pretrain $Pretrain \
       --atten_type $atten_type \
       --nodes_num $NODES_NUM \
       --arch $ARCH \
       --status test \
       --model_name DCASE2021-Audio/baseline_resnet50_nomixup_AudioSet_baseline_7f_20_epoch_60_20.pth.tar \
       --fusion True \
       --num_classes 10 \
       --batch_size $BATCH_SIZE \
       --dataset_name $DATASET_NAME >> $LOG_FILE 

       #--audio_pretrain_weights $audio_pretrain_weights \
       #--visual_pretrain_weights $visual_pretrain_weights\

