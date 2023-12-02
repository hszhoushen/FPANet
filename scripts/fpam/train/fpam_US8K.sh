#!/usr/bin/env bash
source activate llod

LR=0.01
LRI=20
EPOCHS=60
NODES_NUM=20
BATCH_SIZE=32
MODEL_TYPE='FPAM_resnet50_nomixup'
Pretrain='AudioSet'   # ImageNet, AudioSet, None
ARCH='resnet50'
atten_type='fpam_esc'

DATASET_NAME='US8K'   # ESC10, ESC50, US8K
LOG_DIR="./logs/""$DATASET_NAME""/"
current_time=$(date  "+%Y%m%d-%H%M%S-")


LOG_FILE1=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$DATASET_NAME""_""$BATCH_SIZE""_""$NODES_NUM""_1.txt""
LOG_FILE2=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$DATASET_NAME""_""$BATCH_SIZE""_""$NODES_NUM""_2.txt""
LOG_FILE3=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$DATASET_NAME""_""$BATCH_SIZE""_""$NODES_NUM""_3.txt""
LOG_FILE4=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$DATASET_NAME""_""$BATCH_SIZE""_""$NODES_NUM""_4.txt""
LOG_FILE5=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$DATASET_NAME""_""$BATCH_SIZE""_""$NODES_NUM""_5.txt""

echo $LOG_FILE1,$LOG_FILE2,$LOG_FILE3,$LOG_FILE4,$LOG_FILE5


CUDA_VISIBLE_DEVICES=0 python train_fpam_esc.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 1 --atten_type $atten_type --nodes_num $NODES_NUM --arch $ARCH --fusion True --num_classes 10 --bs $BATCH_SIZE  --dataset_name $DATASET_NAME --pretrain $Pretrain >> $LOG_FILE1 &
CUDA_VISIBLE_DEVICES=1 python train_fpam_esc.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 2 --atten_type $atten_type --nodes_num $NODES_NUM --arch $ARCH --fusion True --num_classes 10 --bs $BATCH_SIZE  --dataset_name $DATASET_NAME --pretrain $Pretrain >> $LOG_FILE2 &
CUDA_VISIBLE_DEVICES=2 python train_fpam_esc.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 3 --atten_type $atten_type --nodes_num $NODES_NUM --arch $ARCH --fusion True --num_classes 10 --bs $BATCH_SIZE  --dataset_name $DATASET_NAME --pretrain $Pretrain >> $LOG_FILE3 &
CUDA_VISIBLE_DEVICES=3 python train_fpam_esc.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 4 --atten_type $atten_type --nodes_num $NODES_NUM --arch $ARCH --fusion True --num_classes 10 --bs $BATCH_SIZE  --dataset_name $DATASET_NAME --pretrain $Pretrain >> $LOG_FILE4 &
#CUDA_VISIBLE_DEVICES=0 python train_fpam_esc.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 5 --atten_type $atten_type --nodes_num $NODES_NUM --arch $ARCH --fusion True --num_classes 10 --bs $BATCH_SIZE  --dataset_name $DATASET_NAME --pretrain >> $LOG_FILE5 &

if [[ $DATASET_NAME = 'US8K' ]];
then

LOG_FILE6=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$DATASET_NAME""_""$BATCH_SIZE""_""$NODES_NUM""_6.txt""
LOG_FILE7=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$DATASET_NAME""_""$BATCH_SIZE""_""$NODES_NUM""_7.txt""
LOG_FILE8=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$DATASET_NAME""_""$BATCH_SIZE""_""$NODES_NUM""_8.txt""
LOG_FILE9=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$DATASET_NAME""_""$BATCH_SIZE""_""$NODES_NUM""_9.txt""
LOG_FILE10=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$DATASET_NAME""_""$BATCH_SIZE""_""$NODES_NUM""_10.txt""

echo $LOG_FILE6,$LOG_FILE7,$LOG_FILE8,$LOG_FILE9,$LOG_FILE10

#CUDA_VISIBLE_DEVICES=1 python train_fpam_esc.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 6 --atten_type $atten_type --nodes_num $NODES_NUM --arch $ARCH --fusion True --num_classes 10 --bs $BATCH_SIZE  --dataset_name $DATASET_NAME --pretrain >> $LOG_FILE6 &
#CUDA_VISIBLE_DEVICES=2 python train_fpam_esc.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 7 --atten_type $atten_type --nodes_num $NODES_NUM --arch $ARCH --fusion True --num_classes 10 --bs $BATCH_SIZE  --dataset_name $DATASET_NAME --pretrain >> $LOG_FILE7 &
#CUDA_VISIBLE_DEVICES=3 python train_fpam_esc.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 8 --atten_type $atten_type --nodes_num $NODES_NUM --arch $ARCH --fusion True --num_classes 10 --bs $BATCH_SIZE  --dataset_name $DATASET_NAME --pretrain >> $LOG_FILE8 &
#CUDA_VISIBLE_DEVICES=0 python train_fpam_esc.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 9 --atten_type $atten_type --nodes_num $NODES_NUM --arch $ARCH --fusion True --num_classes 10 --bs $BATCH_SIZE  --dataset_name $DATASET_NAME --pretrain >> $LOG_FILE9 &
#CUDA_VISIBLE_DEVICES=1 python train_fpam_esc.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 10 --atten_type $atten_type --nodes_num $NODES_NUM --arch $ARCH --fusion True --num_classes 10 --bs $BATCH_SIZE  --dataset_name $DATASET_NAME --pretrain >> $LOG_FILE10 &
fi
