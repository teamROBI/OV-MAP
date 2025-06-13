#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
set -e

WS_PATH="$(pwd)"

# OPENMASK3D SCANNET200 EVALUATION SCRIPT
# This script performs the following in order to evaluate OpenMask3D predictions on the ScanNet200 validation set
# 1. Compute class agnostic masks and save them
# 2. Compute mask features for each mask and save them
# 3. Evaluate for closed-set 3D semantic instance segmentation

# --------
# NOTE: SET THESE PARAMETERS!

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
MASK_MODEL=${1:-$DEFAULT_MASK_MODEL}  # Default to "cropformer" if not provided (e.g., cropformer, sam, oneformer)
DEPTH_TYPE="${2:-$DEFAULT_DEPTH_TYPE}_depth"  # Default to "sup_depth" if not provided (e.g., sup, pc, raw)
GPU_NUM=${3:-0}       # GPU index (e.g., 0, 1), default to 0 if not provided
PART=${4:-0}          # Process part (e.g., 012, 345), default to 0 if not provided
SUBPROCESS_NUM=1      # Default value set to 4 (can adjust if necessary)

# Change dataset and parameter
DATASET_TYPE="scannetv2" # scannetv2, replica, real_world
SAM_CKPT_PATH="${WS_PATH}/weights/sam_vit_h_4b8939.pth"

# input directories of scannetv2 data
SCANS_PATH="${WS_PATH}/data/scannetv2/input/pointcept_process/val" # 3D data
SCANNET_PROCESSED_DIR="${WS_PATH}/data/scannetv2/processed_data/scannetv2_200_openmask3d" # 3D data with labeling
SCANS_2D_PATH="${WS_PATH}/data/scannetv2/input/scannetv2_images/val" # 2D data

# 3D mask path
OVERLAP_CRITERION="lo3" #large_overlap(lo3), small overlap(so5), intersection over union(iou3)
VOXEL_SIZE=0.03
IMAGE_ITER=10
HYPER_PARAMETERS="OVM_OC${OVERLAP_CRITERION}_VX${VOXEL_SIZE}_IT${IMAGE_ITER}"  #ov_large_overlap_0.3, large_overlap_0.3, small_overlap_0.5, save_bivi, save

# For openmask3d. Set frequency to see how much image you want. Set intrinsic parameter in your favor.
FREQUENCY=10
INTRINSIC="C" # DEPTH(D) or COLOR(C)
VIS_THRESHOLD=0.25
DEPTH_SUPPLEMENTED=false

DEBUG_VIS_MODE=0
SAVE_SUBS=false

# output directories to save masks and mask features
EXPERIMENT_NAME="scannetv2_200"
OUTPUT_MASK_DIRECTORY="$(pwd)/open_query/clip_output/${EXPERIMENT_NAME}/${MASK_MODEL}_${DEPTH_TYPE}/${HYPER_PARAMETERS}"

if [ "$DEPTH_SUPPLEMENTED" = true ]; then
  OUTPUT_DIRECTORY="${OUTPUT_MASK_DIRECTORY}/F${FREQUENCY}_I${INTRINSIC}_VT${VIS_THRESHOLD}_DS"
else
  OUTPUT_DIRECTORY="${OUTPUT_MASK_DIRECTORY}/F${FREQUENCY}_I${INTRINSIC}_VT${VIS_THRESHOLD}"
fi

# Prepend DEBUG_ if DEBUG_VIS_MODE is not 0 or SAVE_SUBS is true
if [ "$DEBUG_VIS_MODE" -ne 0 ] || [ "$SAVE_SUBS" = true ]; then
  OUTPUT_DIRECTORY="${OUTPUT_MASK_DIRECTORY}/DEBUG_${OUTPUT_DIRECTORY#${OUTPUT_MASK_DIRECTORY}/}"
fi

OUTPUT_FOLDER_DIRECTORY="${OUTPUT_DIRECTORY}/validation"
MASK_FEATURE_SAVE_DIR="${OUTPUT_FOLDER_DIRECTORY}/mask_features"
EVALUATION_OUTPUT_DIR="${OUTPUT_FOLDER_DIRECTORY}/evaluation_result.txt"

# input directory where class-agnositc masks are saved
MASK_SAVE_DIR="${WS_PATH}/output/${DATASET_TYPE}/${HYPER_PARAMETERS}/mask_array/${MASK_MODEL}_${DEPTH_TYPE}"

# Paremeters below are AUTOMATICALLY set based on the parameters above:
SCANNET_LABEL_DB_PATH="${SCANNET_PROCESSED_DIR%/}/label_database.yaml"
SCANNET_INSTANCE_GT_DIR="${SCANNET_PROCESSED_DIR%/}/instance_gt/validation"
if [ "$INTRINSIC" == "C" ]; then
  INTRINSIC_PATH='intrinsics/intrinsic_color.txt'
  INTRINSIC_RESOLUTION="[968,1296]"
elif [ "$INTRINSIC" == "D" ]; then
  INTRINSIC_PATH='intrinsics/intrinsic_depth.txt'
  INTRINSIC_RESOLUTION="[480,640]"
fi

# Set to true if you wish to save the 2D crops of the masks from which the CLIP features are extracted. It can be helpful for debugging and for a qualitative evaluation of the quality of the masks.
SAVE_CROPS=false

# gpu optimization
OPTIMIZE_GPU_USAGE=false

cd open_query/clip_embedding

# 2. Compute mask features
echo "[INFO] Computing mask features..."
python compute_features_scannet200.py \
data.scans_path=${SCANS_PATH} \
data.masks.masks_path=${MASK_SAVE_DIR} \
data.scans_2d_path=${SCANS_2D_PATH}  \
data.camera.intrinsic_path=${INTRINSIC_PATH} \
data.camera.intrinsic_resolution=${INTRINSIC_RESOLUTION} \
data.depths.depth_supplemented=${DEPTH_SUPPLEMENTED} \
openmask3d.vis_threshold=${VIS_THRESHOLD} \
output.output_directory=${MASK_FEATURE_SAVE_DIR} \
output.experiment_name=${EXPERIMENT_NAME} \
output.save_crops=${SAVE_CROPS} \
external.sam_checkpoint=${SAM_CKPT_PATH} \
openmask3d.frequency=${FREQUENCY} \
gpu.optimize_gpu_usage=${OPTIMIZE_GPU_USAGE} \
gpu.device_num=${GPU_NUM} \
multiprocess.part=${PART} \
multiprocess.subprocess_num=${SUBPROCESS_NUM} \
debug.vis_mode=${DEBUG_VIS_MODE} \
debug.save_subs=${SAVE_SUBS} \
hydra.run.dir="${OUTPUT_FOLDER_DIRECTORY}/hydra_outputs/mask_features_computation"
echo "[INFO] Feature computation done!"

# 3. Evaluate for closed-set 3D semantic instance segmentation
echo "[INFO] Evaluating..."
python evaluation_scannet/run_eval_close_vocab_inst_seg.py \
--gt_dir=${SCANNET_INSTANCE_GT_DIR} \
--mask_pred_dir=${MASK_SAVE_DIR} \
--mask_features_dir=${MASK_FEATURE_SAVE_DIR} \
--evaluation_output_dir=${EVALUATION_OUTPUT_DIR}
