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
atten_type='fpam_esc' #  ['fpam', 'baseline']

DATASET_NAME='ESC50'   # ESC10, ESC50, US8K
LOG_DIR="./logs/""$DATASET_NAME""/"
current_time=$(date  "+%Y%m%d-%H%M%S-")


LOG_FILE1=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$Pretrain""_""$DATASET_NAME""_""$BATCH_SIZE""_1.txt""
LOG_FILE2=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$Pretrain""_""$DATASET_NAME""_""$BATCH_SIZE""_2.txt""
LOG_FILE3=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$Pretrain""_""$DATASET_NAME""_""$BATCH_SIZE""_3.txt""
LOG_FILE4=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$Pretrain""_""$DATASET_NAME""_""$BATCH_SIZE""_4.txt""
LOG_FILE5=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$Pretrain""_""$DATASET_NAME""_""$BATCH_SIZE""_5.txt""

echo $LOG_FILE1, $LOG_FILE2, $LOG_FILE3, $LOG_FILE4, $LOG_FILE5


#CUDA_VISIBLE_DEVICES=0 python train_fpam_esc.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 1 --atten_type $atten_type --nodes_num $NODES_NUM --arch $ARCH --fusion True --num_classes 50 --batch_size $BATCH_SIZE  --dataset_name $DATASET_NAME --pretrain $Pretrain >> $LOG_FILE1 &
#CUDA_VISIBLE_DEVICES=1 python train_fpam_esc.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 2 --atten_type $atten_type --nodes_num $NODES_NUM --arch $ARCH --fusion True --num_classes 50 --batch_size $BATCH_SIZE  --dataset_name $DATASET_NAME --pretrain $Pretrain >> $LOG_FILE2 &
#CUDA_VISIBLE_DEVICES=2 python train_fpam_esc.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 3 --atten_type $atten_type --nodes_num $NODES_NUM --arch $ARCH --fusion True --num_classes 50 --batch_size $BATCH_SIZE  --dataset_name $DATASET_NAME --pretrain $Pretrain >> $LOG_FILE3 &
#CUDA_VISIBLE_DEVICES=3 python train_fpam_esc.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 4 --atten_type $atten_type --nodes_num $NODES_NUM --arch $ARCH --fusion True --num_classes 50 --batch_size $BATCH_SIZE  --dataset_name $DATASET_NAME --pretrain $Pretrain >> $LOG_FILE4 &
CUDA_VISIBLE_DEVICES=3 python train_fpam_esc.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 5 --atten_type $atten_type --nodes_num $NODES_NUM --arch $ARCH --fusion True --num_classes 50 --batch_size $BATCH_SIZE  --dataset_name $DATASET_NAME --pretrain $Pretrain >> $LOG_FILE5 &

