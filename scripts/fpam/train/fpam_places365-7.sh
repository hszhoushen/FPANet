#!/usr/bin/env bash
# source activate llod
# source activate swin
LR=0.01
LRI=20
EPOCHS=2
NODES_NUM=20
BATCH_SIZE=32

MODEL_TYPE='FPAM_resnet50_nomixup_Places365'
Pretrain='Places365'   # ImageNet, AudioSet, None
ARCH='resnet50'
atten_type='fpam' #  ['fpam', 'baseline']

DATASET_NAME='Places365-7'   # ESC10, ESC50, US8K, SUN_RGBD, NYUdata, Places365-7
LOG_DIR="./logs/""$DATASET_NAME""/"
current_time=$(date  "+%Y%m%d-%H%M%S-")

LOG_FILE1=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$Pretrain""_""$DATASET_NAME""_""$BATCH_SIZE""_1.txt""

echo $LOG_FILE1
export CUDA_VISIBLE_DEVICES=1

python -m torch.distributed.launch --master_port 31238 --nproc_per_node=1 train_fpam_esc.py \
       --model_type $MODEL_TYPE \
       --epochs $EPOCHS \
       --lr $LR \
       --lri $LRI \
       --test_set_id 5 \
       --atten_type $atten_type \
       --nodes_num $NODES_NUM \
       --arch $ARCH \
       --fusion True \
       --num_classes 7 \
       --batch_size $BATCH_SIZE \
       --dataset_name $DATASET_NAME \
       --pretrain $Pretrain  >> $LOG_FILE1 &

