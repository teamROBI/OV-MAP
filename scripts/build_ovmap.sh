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
GPU_NUM=${3:-0}       # GPU index (e.g., 0, 1), default to 0 if not provided
PART=${4:-0}          # Process part (e.g., 012, 345), default to 0 if not provided
SUBPROCESS_NUM=1      # Default value set to 4 (can adjust if necessary)

# Dataset and parameters
DATASET_TYPE="scannetv2"   # Possible values: scannetv2, replica, real_world
OVERLAP_CRITERION="lo2"     # large_overlap (lo3) or small_overlap (so5), intersection of union (iou3)
VOXEL_SIZE=0.03             # Voxel size for processing
IMAGE_ITER=10               # Number of image iterations

# Function to set paths and image size based on dataset
function set_dataset_paths {
  case $DATASET_TYPE in
    "scannetv2")
      RGB_PATH="data/${DATASET_TYPE}/input/scannetv2_images/val"
      PCD_PATH="data/${DATASET_TYPE}/input/pointcept_process"
      IMG_SIZE="480,640"
      ;;
    "replica")
      RGB_PATH="data/${DATASET_TYPE}/input/replica_processed/replica_2d"
      PCD_PATH="data/${DATASET_TYPE}/input/replica_processed/replica_3d"
      IMG_SIZE="360,640"
      ;;
    "real_world")
      RGB_PATH="data/${DATASET_TYPE}/input/real_world_images"
      PCD_PATH="data/${DATASET_TYPE}/input/real_world_ply"
      IMG_SIZE="720,1280"
      ;;
    *)
      error_exit "Unsupported DATASET_TYPE: $DATASET_TYPE"
      ;;
  esac
}

# Set the dataset paths based on DATASET_TYPE
set_dataset_paths

# Derived variables
HYPER_PARAMETERS="OVM_OC${OVERLAP_CRITERION}_VX${VOXEL_SIZE}_IT${IMAGE_ITER}"

# Process save checkpoints
SAVE_2DMASK_PATH="data/${DATASET_TYPE}/process_saved/2d_seg/${MASK_MODEL}"

# Result saved output directory
SAVE_PATH="output/${DATASET_TYPE}/${HYPER_PARAMETERS}/save_3d_mask/${MASK_MODEL}_${DEPTH_TYPE}"

# Function to log info messages
function log_info {
  echo "[INFO] $1"
}

# Logging the configuration
log_info "Running with the following configuration:"
log_info "MASK_MODEL: $MASK_MODEL"
log_info "DEPTH_TYPE: $DEPTH_TYPE"
log_info "GPU_NUM: $GPU_NUM"
log_info "PART: $PART"
log_info "DATASET_TYPE: $DATASET_TYPE"
log_info "Overlap Criterion: $OVERLAP_CRITERION"
log_info "Voxel Size: $VOXEL_SIZE"
log_info "Image Size: $IMG_SIZE"
log_info "Save Path: $SAVE_PATH"

# Main command to run the ovmap script
log_info "Starting ovmap processing..."
CUDA_VISIBLE_DEVICES=$GPU_NUM python ovmap.py --dataset_type $DATASET_TYPE --mask_model $MASK_MODEL --part $PART --subprocess_num $SUBPROCESS_NUM \
  --rgb_path $RGB_PATH --pcd_path $PCD_PATH --save_path $SAVE_PATH --save_2dmask_path $SAVE_2DMASK_PATH --img_size $IMG_SIZE \
  --overlap_criterion $OVERLAP_CRITERION --voxel_size $VOXEL_SIZE --image_iter $IMAGE_ITER \
  --depth_type $DEPTH_TYPE \
  
  
# Uncomment the following options for development or visualization
#  --developing \
#  --visualize_segments \
#  --visualize_result

# Completion message
log_info "ovmap processing completed for dataset type: ${DATASET_TYPE}"
log_info "Check output at: ${SAVE_PATH}"

# Evaluation reminder for scannetv2 dataset (if applicable)
if [ "$DATASET_TYPE" == "scannetv2" ]; then
  log_info "Run 'scripts/prepare_evaluation.sh' if all 312 masks are made for scannetv2 validation scenes."
fi