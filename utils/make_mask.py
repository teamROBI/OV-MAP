import numpy as np
import open3d as o3d
import torch
import random
import matplotlib.pyplot as plt
from os.path import join

def save_label_dict(unique_labels, save_pth=None, name='label_dict.npy'):
    label_dict = {}
    for i, label in enumerate(unique_labels):
        label_dict[label] = i
    # np.save(join(save_pth,name), label_dict) # {0: 12, 1: 14 ..}
    return label_dict

def save_mask_xyz(args, scene, labels, save_pth, name='mask_xyz.npy'):
    mask_dict = {}
    
    if args.dataset_type == 'scannetv2':
        index = 'coord'
    elif args.dataset_type == 'replica':
        index = 0
    
    # Iterate through labels and points, append arrays to corresponding label key
    for label, point in zip(labels, scene[index]):
        if label not in mask_dict:
            mask_dict[label] = []
        mask_dict[label].append(point)
        avg_list = []
    for mask in mask_dict:
        mask_arr = np.array(mask_dict[mask])
        average_mask = np.mean(mask_arr, axis=0)
        avg_list.append(average_mask)
    avg_list = np.array(avg_list)
    np.save(join(save_pth,name), avg_list)

def make_mask_array(args, scene, label_dict, scene_labels, save_pth, name='mask_array.pt'):
    if args.dataset_type == 'scannetv2':
        index = 'coord'
    elif args.dataset_type == 'replica':
        index = 0
        
    # Define the shape of the array
    unique_label = np.unique(scene_labels)
    array_shape = (scene[index].shape[0], len(unique_label)) # num of points, length of labels
    mask_array = np.zeros(array_shape, dtype=float)
    for i, label in enumerate(scene_labels):
        mask_array[i, label_dict[label]] = 1.0
    torch.save(mask_array, join(save_pth,name))

def main(scene_pcd_path, sam_3d_pcd_path, save_pth):
    scene = torch.load(scene_pcd_path)
    labels = torch.load(sam_3d_pcd_path)
    unique_labels = np.unique(labels)

    label_dict = save_label_dict(unique_labels, save_pth, 'no_ensemble_label_dict.npy')
    save_mask_xyz(scene, labels, save_pth, 'no_ensemble_mask_xyz.npy')
    make_mask_array(scene, label_dict, labels, save_pth, 'no_ensemble_mask_array.pt')

if __name__ == "__main__":
    scene_pcd_path = 'pointcept_process/train/scene0000_00.pth' #'save/save/scene0000_00_SAM1.pth'
    sam_3d_pcd_path = 'output/scannetv2/save5/label.pth'#label.pth
    save_pth = 'output/scannetv2/save5'

    # main(scene_pcd_path, sam_3d_pcd_path, save_pth)
