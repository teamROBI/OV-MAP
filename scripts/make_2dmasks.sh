#!/bin/bash

MASK_MODEL=$1 #sam, oneformer, cropformer
GPU_NUM=$2 #(0,1) (2,3)
PART=$3 #012 345, 0123 4567
SUBPROCESS_NUM=4 #4

# Change dataset and parameter
DATASET_TYPE="scannetv2" # scannetv2, replica, real_world

# Input data
# Scannetv2
RGB_PATH="data/${DATASET_TYPE}/input/scannetv2_images/val"
IMG_SIZE="480,640"

# Replica
# RGB_PATH="data/${DATASET_TYPE}/input/replica_processed/replica_2d"
# IMG_SIZE="360,640"

# Real-world
# RGB_PATH="data/${DATASET_TYPE}/input/real_world/real_world_images"
# IMG_SIZE="720,1280"

# Process save checkpoints
SAVE_2DMASK_PATH="data/${DATASET_TYPE}/process_saved/2d_seg/$1"

# bash build_ovmap.sh cropformer 0 0
CUDA_VISIBLE_DEVICES=$GPU_NUM python make_2dmasks.py --dataset_type $DATASET_TYPE --mask_model $MASK_MODEL --part $PART --subprocess_num $SUBPROCESS_NUM \
 --rgb_path $RGB_PATH --save_2dmask_path $SAVE_2DMASK_PATH --img_size $IMG_SIZE