#!/bin/bash

# Function to log and exit on errors
function error_exit {
  echo "[ERROR] $1"
  exit 1
}

# Default values for arguments
DEFAULT_MASK_MODEL="cropformer"
DEFAULT_DEPTH_TYPE="sup"

# Check if the required number of arguments are provided
if [ "$#" -lt 0 ]; then
  error_exit "Usage: $0 [MASK_MODEL] [DEPTH_TYPE] [GPU_NUM] [PART]"
fi

# Arguments with defaults
MASK_MODEL=${1:-$DEFAULT_MASK_MODEL}  # Default to "cropformer" if not provided
DEPTH_TYPE="${2:-$DEFAULT_DEPTH_TYPE}_depth"  # Default to "sup_depth" if not provided

# Change dataset and parameter
DATASET_TYPE="scannetv2" # scannetv2, replica, real_world

OVERLAP_CRITERION="iou3" #large_overlap(lo3), small overlap(so5)
VOXEL_SIZE=0.03
IMAGE_ITER=5
HYPER_PARAMETERS="OVM_OC${OVERLAP_CRITERION}_VX${VOXEL_SIZE}_IT${IMAGE_ITER}"  #ov_large_overlap_0.3, large_overlap_0.3, small_overlap_0.5, save_bivi, save

# MERGING_CRITERION="l" #heavy(h), light(l)
# HYPER_PARAMETERS="MC${MERGING_CRITERION}_OC${OVERLAP_CRITERION}_VX${VOXEL_SIZE}_IT${IMAGE_ITER}"  #ov_large_over    lap_0.3, large_overlap_0.3, small_overlap_0.5, save_bivi, save

# Input data
# ScannetV2
PCD_PATH="data/${DATASET_TYPE}/input/pointcept_process/val"
META_PATH="scannet-preprocess/meta_data/scannetv2_val.txt"

# Replica
# PCD_PATH="data/${DATASET_TYPE}/input/replica_processed/replica_3d"
# META_PATH="data/${DATASET_TYPE}/input/replica_processed/replica.txt"

# Result saved output directory
SAVE_PATH="output/${DATASET_TYPE}/${HYPER_PARAMETERS}"

# bash scripts/prepare_evaluation.sh
python prepare_evaluation.py --dataset_type $DATASET_TYPE --scannetv2_val_path $META_PATH --scannetv2_pcd_path $PCD_PATH --seg_save_path $SAVE_PATH --model ${MASK_MODEL}_${DEPTH_TYPE}

echo "[INFO] Proceed with the feature extraction from the 'openmask3d/run_openmask3d_scannet200_eval.sh'. Current dataset is ${DATASET_TYPE}"