import os
import cv2
import numpy as np
import open3d as o3d
import torch
import copy
from collections import defaultdict, Counter

from PIL import Image
from os.path import join
from utils.util import *
from utils.depth_pc_util import *
import imageio


# 2D segmentation utils
def get_sam(image, mask_generator):
    masks = mask_generator.generate(image)
    
    group_ids = np.full((image.shape[0], image.shape[1]), -1, dtype=int)
    num_masks = len(masks)
    group_counter = 0
    for i in reversed(range(num_masks)):
        # print(masks[i]["predicted_iou"])
        group_ids[masks[i]["segmentation"]] = group_counter
        group_counter += 1

    return group_ids, masks

def get_oneformer(image, mask_generator):
    inputs = mask_generator[0](image, ["panoptic"], return_tensors="pt")
    with torch.no_grad():
        outputs = mask_generator[1](**inputs)

    predicted_panoptic_map = mask_generator[0].post_process_panoptic_segmentation(outputs, target_sizes=[(image.shape[0], image.shape[1])])[0]
    masks, masks_info = predicted_panoptic_map["segmentation"], predicted_panoptic_map["segments_info"]
    print(masks.shape)
    assert False, "DEBUG"

    group_ids = np.full((image.shape[0], image.shape[1]), -1, dtype=int)
    group_counter = 0
    for info in masks_info:
        if info['id'] == 0:
            continue
        mask = masks == info['id']
        group_ids[mask] = group_counter
        group_counter += 1

    return group_ids, masks, masks_info

def get_cropformer(image, mask_generator):
    predictions = mask_generator.run_on_image(image)
    
    pred_masks = predictions["instances"].pred_masks
    pred_scores = predictions["instances"].scores
    
    # select by confidence threshold
    selected_indexes = (pred_scores >= 0.7) #0.6 for large_overlap_0.3 and small_overlap_0.5
    selected_scores = pred_scores[selected_indexes].cpu().numpy()
    selected_masks  = pred_masks[selected_indexes].cpu().numpy()
    
    sorted_indexes = np.argsort(selected_scores)
    sorted_masks = selected_masks[sorted_indexes]

    group_ids = np.full((image.shape[0], image.shape[1]), -1, dtype=int)
    group_counter = 0
    for mask in sorted_masks:
        group_ids[mask.astype(bool)] = group_counter
        group_counter += 1

    return group_ids, sorted_masks


def get_pcd(args, scene_name, color_name, mask_generator, dense_scene_pcd):
    intrinsic_path = join(args.rgb_path, scene_name, 'intrinsics', 'intrinsic_depth.txt')
    depth_intrinsic = np.loadtxt(intrinsic_path)

    pose = join(args.rgb_path, scene_name, 'pose', color_name[0:-4] + '.txt')
    depth = join(args.rgb_path, scene_name, 'depth', color_name[0:-4] + '.png')
    color = join(args.rgb_path, scene_name, 'color', color_name)

    # Read data
    pose = np.loadtxt(pose)
    # pc_dict = torch.load(pc_path)
    # Choose which depth image to use. Camera raw depth, PCD rendered depth, Supplemented depth
    if args.depth_type == "raw_depth":
        if args.dataset_type == "scannetv2" or args.dataset_type == "real_world":
            depth_img = cv2.imread(depth, -1) # read 16bit grayscale image, scale=1000, uint16
        elif args.dataset_type == "replica":
            depth_img = imageio.v2.imread(depth) / 6533.5
    else:
        if args.depth_type == "sup_depth":
            raw_depth_img = cv2.imread(depth, -1) / 1000.0  # read 16bit grayscale image, scale=1000, uint16
            if args.print_info: print("[INFO] Using supplemented depth image.")
        elif args.depth_type == "pc_depth":            
            if args.print_info: print("[INFO] Using point cloud depth.")

        scene_depth_dir = f'{args.working_dir}/data/{args.dataset_type}/process_saved/depth_from_pc/{scene_name}'
        scene_vis_dir = f'{args.working_dir}/data/{args.dataset_type}/process_saved/depth_from_pc_vis/{scene_name}'
        os.makedirs(scene_depth_dir, exist_ok=True)
        os.makedirs(scene_vis_dir, exist_ok=True)
        
        if not os.path.exists(f'{scene_depth_dir}/{color_name[0:-4]}.png'):
            # Get depth information from point cloud, scale=1.0, np.float32
            depth_img = render_point_cloud_to_depth_image(np.asarray(dense_scene_pcd.points), pose, depth_intrinsic, image_shape=args.img_size) #(480, 640)

            # Save synthetic depth image
            cv2.imwrite(f'{scene_depth_dir}/{color_name[0:-4]}.png', (depth_img * 1000).astype(np.uint16))
            # np.savez_compressed(f'{scene_depth_dir}/{color_name[0:-4]}.npz', synthetic_depth=depth_img.astype(np.float32))
            
            depth_image_visual = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            cv2.imwrite(f'{scene_vis_dir}/{color_name[0:-4]}.jpg', depth_image_visual)
        else:
            synthetic_depth = join(f'{scene_depth_dir}/{color_name[0:-4]}.png')
            depth_img = cv2.imread(synthetic_depth, -1) / 1000.0 # read 16bit grayscale image, scale=1000, uint16
            # synthetic_depth = join(f'{scene_depth_dir}/{color_name[0:-4]}.npz')
            # depth_img = np.load(synthetic_depth)["synthetic_depth"].astype(np.float64)
            if args.print_info: print("[INFO] Use saved depth image, extracted from dense point cloud")

    if args.depth_type == "sup_depth": # Update depth image
        depth_img = supplement_depth(raw_depth_img, depth_img)   

    mask = (depth_img != 0)
    
    # Get 2D RGB segmented result
    color_image = cv2.imread(color)
    color_image = cv2.resize(color_image, (args.img_size[1], args.img_size[0])) #(640, 480)

    save_2dmask_path = join(args.save_2dmask_path, scene_name)
    if mask_generator is not None:
        if not os.path.exists(save_2dmask_path):
            os.makedirs(save_2dmask_path)

        group_path = join(save_2dmask_path, color_name[0:-4] + '.png')
        if not os.path.exists(group_path):
            if args.mask_model == "oneformer":
                group_ids, masks, masks_info = get_oneformer(color_image, mask_generator)
                # save_oneformer_result(color_image, masks, masks_info, join(save_2dmask_path, color_name[0:-4] + '.jpg'))
            elif args.mask_model == "cropformer":
                group_ids, masks = get_cropformer(color_image, mask_generator)
                debug_img = save_cropformer_result(color_image, masks, join(save_2dmask_path, color_name[0:-4] + '.jpg'))

                # debug_img = (debug_img[:, :, :3] * 255).astype(np.uint8)
                # debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
                # if int(color_name[0:-4]) % 1 == 0:
                #     backproject_rgbd_to_pointcloud(debug_img, copy.deepcopy(depth_img), depth_intrinsic)
            elif args.mask_model == "sam":
                group_ids, masks = get_sam(color_image, mask_generator)
                save_sam_result(color_image, masks, join(save_2dmask_path, color_name[0:-4] + '.jpg'))  # Full mask annotation

            # Save segmentation result for future convineience
            img = Image.fromarray(num_to_natural(group_ids).astype(np.int16), mode='I;16')
            img.save(join(save_2dmask_path, color_name[0:-4] + '.png'))

        else:
            if args.print_info: print("[INFO] Use saved masks")
            img = Image.open(group_path)
            group_ids = np.array(img, dtype=np.int16)
    
    # object_ids, group_ids = clipcapture(color_image, group_ids)
    
    group_ids = remove_group_ids_near_edge(group_ids)
    masked_group_ids = group_ids[mask]
    points_world = camera2world(depth_img, depth_intrinsic, pose)
    save_dict = dict(coord=points_world[:, :3], group=masked_group_ids) #, object=object_ids)
    
    return save_dict



# 3D merging utils
def cal_2_scenes(args, pcd_list, index, voxelize):
    if len(index) == 1:
        return(pcd_list[index[0]])
    # print(index, flush=True)
    input_dict_0 = pcd_list[index[0]]
    input_dict_1 = pcd_list[index[1]]

    try:
        pcd0 = o3d.geometry.PointCloud()
        pcd0.points = o3d.utility.Vector3dVector(input_dict_0["coord"])
    except:
        pcd0 = None
    try:
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(input_dict_1["coord"])
    except:
        pcd1 = None
        
    if pcd0 == None:
        if pcd1 == None:
            return None
        else:
            return input_dict_1
    elif pcd1 == None:
        return input_dict_0

    # Cal Dul-overlap
    # print("coord len:", len(input_dict_0["coord"]), len(input_dict_1["coord"]))
    # print("group before:", len(input_dict_0["group"]), np.unique(input_dict_0["group"], return_counts=True), len(input_dict_1["group"]), np.unique(input_dict_1["group"], return_counts=True))
    pcd0_tree = o3d.geometry.KDTreeFlann(copy.deepcopy(pcd0))
    match_inds = get_matching_indices(pcd1, pcd0_tree, 2 * args.voxel_size, 1)
    _, pcd1_new_group = cal_group(args, input_dict_0, input_dict_1, match_inds)
    # print("group after:", len(pcd0_new_group), np.unique(pcd0_new_group, return_counts=True), len(pcd1_new_group), np.unique(pcd1_new_group, return_counts=True))

    # Remove unassigned -1 points
    # pcd0_remain_indices = np.where(pcd0_new_group != -1)[0]
    # input_dict_0["coord"] = input_dict_0["coord"][pcd0_remain_indices]
    # pcd0_new_group = pcd0_new_group[pcd0_remain_indices]

    # pcd1_remain_indices = np.where(pcd1_new_group != -1)[0]
    # input_dict_1["coord"] = input_dict_1["coord"][pcd1_remain_indices]
    # pcd1_new_group = pcd1_new_group[pcd1_remain_indices]
    # print("group after:", len(pcd0_new_group), np.unique(pcd0_new_group, return_counts=True), len(pcd1_new_group), np.unique(pcd1_new_group, return_counts=True))

    pcd1_tree = o3d.geometry.KDTreeFlann(copy.deepcopy(pcd1))
    match_inds = get_matching_indices(pcd0, pcd1_tree, 2 * args.voxel_size, 1)
    input_dict_1["group"] = pcd1_new_group
    _, pcd0_new_group = cal_group(args, input_dict_1, input_dict_0, match_inds)
    # print(pcd0_new_group)

    pcd_new_coord = np.concatenate((input_dict_0["coord"], input_dict_1["coord"]), axis=0)
    pcd_new_group = np.concatenate((pcd0_new_group, pcd1_new_group), axis=0)
    pcd_new_group = nearest_neighbor_smoothing(pcd_new_coord, pcd_new_group, k=50)
    pcd_new_group = num_to_natural(pcd_new_group)
    pcd_dict = dict(coord=pcd_new_coord, group=pcd_new_group)

    assert len(pcd_new_group) == len(pcd_new_coord), "Coord and group has to have same length"
    # find_duplicate_coordinates(pcd_new_coord)

    pcd_dict = voxelize(pcd_dict)#, custom_voxel_size=0.01)
    return pcd_dict


def cal_2_scenes_ovmap(args, pcd_list, index, voxelize):
    if len(index) == 1:
        return(pcd_list[index[0]])
    # print(index, flush=True)
    input_dict_0 = pcd_list[index[0]]
    input_dict_1 = pcd_list[index[1]]

    try:
        pcd0 = o3d.geometry.PointCloud()
        pcd0.points = o3d.utility.Vector3dVector(input_dict_0["coord"])
    except:
        pcd0 = None
    try:
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(input_dict_1["coord"])
    except:
        pcd1 = None
        
    if pcd0 == None:
        if pcd1 == None:
            return None
        else:
            return input_dict_1
    elif pcd1 == None:
        return input_dict_0

    # Cal Dul-overlap
    # print("coord len:", len(input_dict_0["coord"]), len(input_dict_1["coord"]))
    # print("group before:", len(input_dict_0["group"]), np.unique(input_dict_0["group"], return_counts=True), len(input_dict_1["group"]), np.unique(input_dict_1["group"], return_counts=True))
    pcd0_tree = o3d.geometry.KDTreeFlann(copy.deepcopy(pcd0))
    match_inds = get_matching_indices(pcd1, pcd0_tree, 2 * args.voxel_size, 1)
    pcd0_new_group, pcd1_new_group = cal_group(args, input_dict_0, input_dict_1, match_inds)
    # print("group after:", len(pcd0_new_group), np.unique(pcd0_new_group, return_counts=True), len(pcd1_new_group), np.unique(pcd1_new_group, return_counts=True))

    # Remove unassigned -1 points
    # pcd0_remain_indices = np.where(pcd0_new_group != -1)[0]
    # input_dict_0["coord"] = input_dict_0["coord"][pcd0_remain_indices]
    # pcd0_new_group = pcd0_new_group[pcd0_remain_indices]

    # pcd1_remain_indices = np.where(pcd1_new_group != -1)[0]
    # input_dict_1["coord"] = input_dict_1["coord"][pcd1_remain_indices]
    # pcd1_new_group = pcd1_new_group[pcd1_remain_indices]
    # print("group after:", len(pcd0_new_group), np.unique(pcd0_new_group, return_counts=True), len(pcd1_new_group), np.unique(pcd1_new_group, return_counts=True))

    # pcd1_tree = o3d.geometry.KDTreeFlann(copy.deepcopy(pcd1))
    # match_inds = get_matching_indices(pcd0, pcd1_tree, 2 * args.voxel_size, 1)
    # input_dict_1["group"] = pcd1_new_group
    # pcd0_new_group = cal_group(args, input_dict_1, input_dict_0, match_inds)
    # print(pcd0_new_group)

    pcd_new_coord = np.concatenate((input_dict_0["coord"], input_dict_1["coord"]), axis=0)
    pcd_new_group = np.concatenate((pcd0_new_group, pcd1_new_group), axis=0)
    pcd_new_group = nearest_neighbor_smoothing(pcd_new_coord, pcd_new_group, k=50)
    pcd_new_group = num_to_natural(pcd_new_group)
    pcd_dict = dict(coord=pcd_new_coord, group=pcd_new_group)

    assert len(pcd_new_group) == len(pcd_new_coord), "Coord and group has to have same length"
    # find_duplicate_coordinates(pcd_new_coord)

    pcd_dict = voxelize(pcd_dict)#, custom_voxel_size=0.01)
    return pcd_dict

def merge_2_frames_FGP(frames):
    frame1, frame2 = frames
    if frame2 is None:
        return frame1  # If no second frame, return the first frame unchanged

    # Extract unique group IDs
    unique_groups_frame1 = np.unique(frame1['group'])
    unique_groups_frame2 = np.unique(frame2['group'])

    # Dictionary to store the highest IoU for each group in frame1
    highest_iou_scores = {}

    for group_id1 in unique_groups_frame1:
        # Get indices of the points in frame1 for this group
        frame1_coords = frame1['coord'][frame1['group'] == group_id1]
        
        best_iou = 0
        best_group_id2 = None
        
        for group_id2 in unique_groups_frame2:
            # Get indices of the points in frame2 for this group
            frame2_coords = frame2['coord'][frame2['group'] == group_id2]
            
            # Calculate IoU score between these two groups
            iou = iou_score(frame1_coords, frame2_coords)
            
            # Track the highest IoU for this group_id1
            if iou > best_iou:
                best_iou = iou
                best_group_id2 = group_id2

        # Store the highest IoU for the current group_id1
        highest_iou_scores[group_id1] = (best_group_id2, best_iou)
        
    print("highest iou:", highest_iou_scores)

    return highest_iou_scores ##################### scene pcd 위에 group id 올리는 걸로 다시 해야 할듯, frame pcd coord로 하려고 하니까 겹치는 걸 잡는 게 쉽지 않네. 가까운 거 하나만 할당을 하는 방식으로 되서


def merge_2_frames_FGP(frames):
    frame1, frame2, args = frames
    if frame2 is None:
        return frame1

    merged_frame = defaultdict(list)
    new_group_id = 0  # Start assigning group IDs from zero

    # Step 1: Calculate IoU for all pairs and store them in a list
    overlap_pairs = []
    for group_id1, indices1 in frame1.items():
        for group_id2, indices2 in frame2.items():
            set1, set2 = set(indices1), set(indices2)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            if args.overlap_criterion[:-1] == 'lo':
                overlap_ratio = intersection / max(len(set1), len(set2))
            elif args.overlap_criterion[:-1] == 'so':
                overlap_ratio = intersection / min(len(set1), len(set2))
            elif args.overlap_criterion[:-1] == 'iou':
                overlap_ratio = intersection / union if union > 0 else 0
            else:
                raise ValueError("[ERROR] Overlap criterion must be large overlap(lo) or small overlap(so) or iou")
            
            overlap_th = int(args.overlap_criterion[-1]) * 0.1
            if overlap_ratio >= overlap_th:
                overlap_pairs.append((overlap_ratio, group_id1, group_id2, indices1, indices2))

    # Step 2: Sort pairs by IoU in descending order
    overlap_pairs.sort(reverse=True, key=lambda x: x[0])
    
    # Track merged group IDs
    merged_groups_frame1 = set()
    merged_groups_frame2 = set()

    # Step 3: Merge groups based on highest IoU first
    for iou, group_id1, group_id2, indices1, indices2 in overlap_pairs:
        if group_id1 not in merged_groups_frame1 and group_id2 not in merged_groups_frame2:
            # Merge these groups and assign a new ID
            merged_indices = indices1 + indices2  # Concatenate to keep duplicates
            merged_frame[new_group_id].extend(merged_indices)
            new_group_id += 1  # Increment for the next ID

            # Mark these groups as merged
            merged_groups_frame1.add(group_id1)
            merged_groups_frame2.add(group_id2)

    # Step 4: Add remaining unmerged groups from frame1
    for group_id1, indices1 in frame1.items():
        if group_id1 not in merged_groups_frame1:
            merged_frame[new_group_id].extend(indices1)
            new_group_id += 1

    # Step 5: Add remaining unmerged groups from frame2
    for group_id2, indices2 in frame2.items():
        if group_id2 not in merged_groups_frame2:
            merged_frame[new_group_id].extend(indices2)
            new_group_id += 1

    return merged_frame

def iou_score(group1_coords, group2_coords):
    """Calculate the IoU score between two sets of coordinates."""
    intersection = len(np.intersect1d(group1_coords, group2_coords))
    union = len(np.union1d(group1_coords, group2_coords))
    return intersection / union if union > 0 else 0


def cal_group(args, input_dict, new_input_dict, match_inds, merge_ratio=0.3): #merge=S0.5, L0.3
    group_0 = input_dict["group"]
    group_1 = new_input_dict["group"]
    group_1[group_1 != -1] += group_0.max() + 1
    
    unique_groups, group_0_counts = np.unique(group_0, return_counts=True)
    group_0_counts = dict(zip(unique_groups, group_0_counts))
    unique_groups, group_1_counts = np.unique(group_1, return_counts=True)
    group_1_counts = dict(zip(unique_groups, group_1_counts))

    # Calculate the group number correspondence of overlapping points
    group_overlap_count = {}
    group_overlap_indices = {}
    for i, j in match_inds:
        group_i = group_1[i]
        group_j = group_0[j]
        if group_i == -1:
            group_1[i] = group_0[j]
            continue
        if group_j == -1:
            continue
        if group_i not in group_overlap_count:
            group_overlap_count[group_i] = {}
            group_overlap_indices[group_i] = {}
        if group_j not in group_overlap_count[group_i]:
            group_overlap_count[group_i][group_j] = 0
            group_overlap_indices[group_i][group_j] = []
        group_overlap_count[group_i][group_j] += 1
        group_overlap_indices[group_i][group_j].append((i, j))

    # Update group information for point cloud 1
    for group_i, overlap_count in group_overlap_count.items():
        # for group_j, count in overlap_count.items():
        max_index = np.argmax(np.array(list(overlap_count.values())))
        group_j = list(overlap_count.keys())[max_index]
        count = list(overlap_count.values())[max_index]
        large_total_count = max(group_0_counts[group_j], group_1_counts[group_i]).astype(np.float32)
        small_total_count = min(group_0_counts[group_j], group_1_counts[group_i]).astype(np.float32)
        
        if args.overlap_criterion[:-1] == 'lo':
            merge_ratio = int(args.overlap_criterion[-1]) * 0.1
            overlap_ratio = count / large_total_count
        elif args.overlap_criterion[:-1] == 'so':
            merge_ratio = int(args.overlap_criterion[-1]) * 0.1
            overlap_ratio = count / small_total_count
        elif args.overlap_criterion[:-1] == 'iou':
            merge_ratio = int(args.overlap_criterion[-1]) * 0.1
            overlap_ratio = count / (small_total_count + large_total_count - count)
            
        else:
            raise ValueError("[ERROR] Overlap criterion must be large overlap(lo) or small overlap(so)")
        
        if overlap_ratio >= merge_ratio: #small_total_count, large_total_count
            group_1[group_1 == group_i] = group_j
        # else:
        #     if group_0_counts[group_j] < group_1_counts[group_i]:
        #         for i, j in group_overlap_indices[group_i][group_j]:
        #             group_1[i] = -1
        #     else:
        #         for i, j in group_overlap_indices[group_i][group_j]:
        #             group_0[j] = -1

    return group_0, group_1



def dominant_voting(merged_g2s, mesh_seg_path):
    # Load the mesh segmentation JSON file
    with open(mesh_seg_path, 'r') as f:
        mesh_segmentation = json.load(f)

    # Extract segmentation indices
    mesh_seg_indices = mesh_segmentation['segIndices']  # This should be a list of shape (M,)

    # Step 1: Build reverse mappings for fast lookup
    # Map each scene index to its mesh segmentation ID
    scene_to_mesh_seg = {i: mesh_seg_id for i, mesh_seg_id in enumerate(mesh_seg_indices)}

    # Map each scene index to its group ID based on g2s_list
    scene_to_group = {}
    for group_id, indices in merged_g2s.items():
        for idx in indices:
            if idx in scene_to_group:
                scene_to_group[idx].append(group_id)
            else:
                scene_to_group[idx] = [group_id]  # Retain duplicates

    # Step 2: Count the most dominant group ID for each mesh segmentation ID
    dominant_groups = {}

    # Initialize counters for each mesh segmentation ID
    mesh_seg_to_group_counts = defaultdict(Counter)

    # For each scene index, update the count of each group ID for the corresponding mesh segmentation ID
    for idx, group_ids in tqdm(scene_to_group.items()):
        mesh_seg_id = scene_to_mesh_seg.get(idx)
        if mesh_seg_id is not None:
            mesh_seg_to_group_counts[mesh_seg_id].update(group_ids)

    # Step 3: Find the most common group ID for each mesh segmentation ID
    for mesh_seg_id, group_count in mesh_seg_to_group_counts.items():
        dominant_group_id, _ = group_count.most_common(1)[0]  # Get the group ID with the highest count
        dominant_groups[mesh_seg_id] = dominant_group_id
        
    # Step 4: Assign the dominant group IDs to mesh_seg_indices
    dominant_group_indices = [dominant_groups.get(mesh_seg_id, -1) for mesh_seg_id in mesh_seg_indices]

    return np.array(dominant_group_indices)














###################
def make_open3d_point_cloud(input_dict, voxelize, th):
    input_dict["group"] = remove_small_group(input_dict["group"], th)
    # input_dict = voxelize(input_dict)

    xyz = input_dict["coord"]
    if np.isnan(xyz).any():
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def compare_2_scene_clusters(pcd_list, index, voxel_size, voxelize):
    if len(index) == 1:
        return(pcd_list[index[0]])
    # print(index, flush=True)
    input_dict_0 = pcd_list[index[0]]
    input_dict_1 = pcd_list[index[1]]
    coord_0, group_0 = input_dict_0["coord"], input_dict_0["group"]
    coord_1, group_1 = input_dict_1["coord"], input_dict_1["group"]
    
    merged_coord, merged_group = [], []

    unique_groups_0 = np.unique(group_0)
    unique_groups_1 = np.unique(group_1)

    for i, cluster_0 in enumerate(unique_groups_0):
        coord_cluster_0 = coord_0[np.where(group_0 == cluster_0)[0]]
        for j, cluster_1 in enumerate(unique_groups_1):
            coord_cluster_1 = coord_1[np.where(group_1 == cluster_1)[0]]
            large_overlapping, overlapping_indices = calculate_cluster_overlap(coord_cluster_0, coord_cluster_1, voxel_size)
            # print(large_overlapping, len(overlapping_indices))

            if large_overlapping > 0.4:
                merged_coord.extend(coord_cluster_0)
                merged_coord.extend(coord_cluster_1)
                group_len = len(coord_cluster_0) + len(coord_cluster_1)
                merged_group.extend(np.full(group_len, cluster_0))
            elif len(overlapping_indices) > 0:
                if len(coord_cluster_0) < len(coord_cluster_1):
                    coord_cluster_0 = np.delete(coord_cluster_0, overlapping_indices, axis=0)
                    merged_coord.extend(coord_cluster_1)
                    merged_group.extend(np.full(len(coord_cluster_1), cluster_1))
                else:
                    coord_cluster_1 = np.delete(coord_cluster_1, overlapping_indices, axis=0)
                    merged_coord.extend(coord_cluster_1)
                    merged_group.extend(np.full(len(coord_cluster_1), cluster_1))
            else:
                merged_coord.extend(coord_cluster_1)
                merged_group.extend(np.full(len(coord_cluster_1), cluster_1))

        merged_coord.extend(coord_cluster_0)
        merged_group.extend(np.full(len(coord_cluster_0), cluster_0))
    
    pcd_dict = dict(coord=np.array(merged_coord), group=np.array(merged_group))
    pcd_dict = voxelize(pcd_dict)
    # dup = find_duplicate_coordinates(pcd_dict["coord"])
    # print(len(dup))

    return pcd_dict