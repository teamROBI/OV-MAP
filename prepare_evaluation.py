import os
from glob import glob
from os.path import join
import torch
import numpy as np
import argparse

import utils.make_mask as make_mask
from tqdm import tqdm

def mkdir(path):
    # Check if the directory already exists
    if not os.path.exists(path):
        # Create the directory
        os.makedirs(path)
    else:
        print(f"[INFO] Directory '{path}' already exists.")
        return 'pass'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', default='scannetv2')
    parser.add_argument('--scannetv2_val_path', type=str, default='scannet-preprocess/meta_data/scannetv2_val.txt', help='path to scannetv2 validation meta data')
    parser.add_argument('--scannetv2_pcd_path', type=str, default='data/scannetv2/input/pointcept_process/val', help='path to scannetv2 point clouds')
    parser.add_argument('--seg_save_path', type=str, required=True, help='path to the saved segmentation masks')
    parser.add_argument('--model', type=str, default="cropformer_sup_depth", help='2D segmentation model used')

    args = parser.parse_args()    
    # 경로 지정
    # scannetv2_val_path = 'scannet-preprocess/meta_data/scannetv2_train.txt'
    # scannetv2_dir_path = 'data/scannetv2/input/pointcept_process/train'
    
    param1 = "0.05"
    param2 = "20"
    mesh_seg_path = 'data/scannetv2/input/org_process/mesh_segmentation/'+f'{param1}_{param2}'

    mask_label_dir_path = join(args.seg_save_path, 'save_3d_mask', args.model)
    mkdir(mask_label_dir_path)
    mask_array_dir_path = join(args.seg_save_path, 'mask_array', args.model)
    mkdir(mask_array_dir_path)
    # mask_xyz_dir_path = join(args.seg_save_path, 'mask_xyz', args.model)
    # mkdir(mask_xyz_dir_path)
    # label_dict_dir_path = join(args.seg_save_path, 'label_dict', args.model)
    # mkdir(label_dict_dir_path)

    org_suffix = f'_vh_clean_2.{param1}0000.segs.json'

    print('======== MAKE SCANNET EVAL ========')
    print('[INFO] Prepare OpenMask3D Mask Input\n')

    # 생성된 SAM3D Output(Voxel voting O)으로 OpenMask3D Input 만들기
    with open(args.scannetv2_val_path) as val_file:
        val_scenes = val_file.read().splitlines()

    for scene in tqdm(sorted(val_scenes)):
        mask_label_path = join(mask_label_dir_path, scene+'.pth') # (N,P) N:mask num, P:point num
        # Check Mask Label Exist
        if os.path.exists(mask_label_path):
            if os.path.exists(join(mask_array_dir_path, scene+'.pt')):
                print(f"[INFO] {join(mask_array_dir_path, scene+'.pt')} Already exists.")
                continue
                
            scene_pth_path = join(args.scannetv2_pcd_path, scene+'.pth') #(P,3) -> [x,y,z]
            scene_pth = torch.load(scene_pth_path)
            mask_label = torch.load(mask_label_path)
            unique_labels = np.unique(mask_label)

            label_dict = make_mask.save_label_dict(unique_labels)#, label_dict_dir_path, scene+'.npy')
            # make_mask.save_mask_xyz(args, scene_pth, mask_label, mask_xyz_dir_path, scene+'.npy')
            make_mask.make_mask_array(args, scene_pth, label_dict, mask_label, mask_array_dir_path, scene+'.pt')
        else:
            print(f'[ERROR] Point cloud for {scene} is not made.')
            continue
            
    print('======== Finish Save Mask Input ========')


