#!/usr/bin/env bash
source activate llod

LR=0.01
LRI=20
EPOCHS=60
BATCH_SIZE=16
NUM_CLASSES=50

MODEL_TYPE='FPAM_resnet50'
Pretrain='AudioSet'   # ImageNet, AudioSet, None
ARCH='resnet50'
ATTEN_TYPE='fpam'   # 'fpam', 'baseline', fpam_esc'

DATASET_NAME='ESC50'   # ESC10, ESC50, US8K

MODEL_NAME='FPAM_resnet50_nomixup_fpam_esc_7f_20_epoch_60_20.pth.tar'

CUDA_VISIBLE_DEVICES=1

python -m torch.distributed.launch --master_port 31010 --nproc_per_node=1 train_fpam_esc.py \
        --model_type $MODEL_TYPE \
        --pretrain $Pretrain \
        --epochs $EPOCHS \
        --lr $LR \
        --lri $LRI \
        --test_set_id 1 \
        --arch $ARCH \
        --num_classes $NUM_CLASSES \
        --atten_type $ATTEN_TYPE \
        --fusion True \
        --batch_size $BATCH_SIZE \
        --dataset_name $DATASET_NAME \
        --status draw_atten \
        --model_name $MODEL_NAME