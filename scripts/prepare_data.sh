# RAW_SCANNET_DIR: the directory of downloaded ScanNet v2 raw dataset.
# PROCESSED_SCANNET_DIR: the directory of processed ScanNet dataset (output dir).

SCANNET_GIT_DIR="/data/jokim/ScanNet/"
RAW_SCANNET_DIR="/data/jokim/ScanNet/dataset/"
PROCESSED_SCANNET_POINTCLOUD_DIR="/data/jokim/OVMap/scannetv2_pointclouds/"

RAW_SCANNET_SCANS_DIR="/data/jokim/ScanNet/dataset/scans"
PROCESSED_SCANNET_IMAGE_DIR="/data/jokim/OVMap/scannetv2_images/val/"
LABEL_MAP_FILE_DIR='/data/jokim/ScanNet/dataset/scannetv2-labels.combined.tsv'

PROCESSED_SCANNET_OPENMASK3D_DIR="/data/jokim/OVMap/scannetv2_openmask3d/"
META_DATA="scannet-preprocess/meta_data/"

MODES="validation" # "train,validation,test"
SELECT_SCENES="scene0660_00" # "scene0011_00,scene0660_00"

# preprocess point cloud data
# python scannet-preprocess/preprocess_scannet.py --dataset_root ${RAW_SCANNET_DIR} --output_root ${PROCESSED_SCANNET_POINTCLOUD_DIR}

# preprocess RGB-D data
python scannet-preprocess/prepare_2d_data/prepare_2d_data.py --scannet_path ${RAW_SCANNET_SCANS_DIR} --output_path ${PROCESSED_SCANNET_IMAGE_DIR} \
--modes "validation" --meta_data ${META_DATA} --export_label_images --label_map_file ${LABEL_MAP_FILE_DIR} \
# --select_scenes ${SELECT_SCENES}

# preprocess data for openmask3d evaluation
# cd openmask3d/openmask3d/class_agnostic_mask_computation
# python -m datasets.preprocessing.scannet_preprocessing preprocess \
# --data_dir=${RAW_SCANNET_DIR} \
# --save_dir=${PROCESSED_SCANNET_OPENMASK3D_DIR} \
# --label_dir=${LABEL_MAP_FILE_DIR} \
# --git_repo=${SCANNET_GIT_DIR} \
# --scannet200=true

# --modes=${MODES} \