import os
import numpy as np
import open3d as o3d
import torch
import multiprocessing as mp
import pointops
import argparse
from scipy.spatial import cKDTree
from collections import defaultdict
import math

# from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
# import open_clip
from segment_anything import build_sam, SamAutomaticMaskGenerator
from os.path import join
import sys

from utils.util import *
from utils.depth_pc_util import *
from utils.ovmap_util import *

# def pcd_ensemble(org_path, new_path, pcd_path, vis_path):
#     new_pcd = torch.load(new_path)
#     new_pcd = num_to_natural(remove_small_group(new_pcd, 20))
#     with open(org_path) as f:
#         segments = json.load(f)
#         org_pcd = np.array(segments['segIndices'])
#     match_inds = [(i, i) for i in range(len(new_pcd))]
#     print(len(match_inds))
#     new_group = cal_group(dict(group=new_pcd), dict(group=org_pcd), match_inds)
#     print(new_group.shape)
#     data = torch.load(pcd_path)

#     basename = os.path.splitext(os.path.basename(pcd_path))[0]
#     visualize_partition(data["coord"], new_group, vis_path, basename)



def seg_pcd(args, scene_name, mask_generator, voxelize):
    print(scene_name, flush=True)
    if os.path.exists(join(args.save_path, scene_name + ".pth")):
        return
    
    ### Data preparation part
    # Select images used for the segmentation process
    color_names = sorted(os.listdir(join(args.rgb_path, scene_name, 'color')), key=lambda a: int(os.path.basename(a).split('.')[0]))
    color_names = color_names[::args.image_iter] #Iterate every 10 images
    print(f"[INFO] Iterate image in every {args.image_iter} iter")

    # Build dense pcd from RGB-D data
    if args.depth_type == "pc_depth" or args.depth_type == "sup_depth":
        print(f"[INFO] Using {args.depth_type} image.")
        count_existing_depth_pc = count_files(f'data/{args.dataset_type}/process_saved/depth_from_pc/{scene_name}')
        if count_existing_depth_pc >= len(color_names):
            print("[INFO] Use saved pc depth", count_existing_depth_pc, ">=", len(color_names))
            dense_scene_pcd = None
        else:
            print("[INFO] Build new 3d scene for pc depth", count_existing_depth_pc, "<", len(color_names))
            dense_scene_pcd = build_scene_point_cloud(args, color_names, scene_name, args.visualize_dense_pcd)
    else:
        print("[INFO] Using raw depth image.")
        dense_scene_pcd = None
    
    ### Get 3D segmented point cloud from 2D segemented image
    print("[PROCESSING] Getting 3D segmented pointclouds from 2D segemented images...")  
    pcd_list = []
    for color_name in tqdm(color_names):
        if args.print_info: print(color_name, flush=True)
        
        pcd_dict = get_pcd(args, scene_name, color_name, mask_generator, dense_scene_pcd)
        if len(pcd_dict["coord"]) == 0:
            continue

        # pl = len(pcd_dict["coord"])
        pcd_dict = voxelize(pcd_dict)
        # if args.print_info: print("[INFO] Voxelized Pecentage:", len(pcd_dict["coord"]) / pl * 100)

        # pcd_dict = filter_pcd(args, pcd_dict, color_name, args.visualize_segments)
        pcd_list.append(pcd_dict)
    
    
    ### Using multiprocessing to parallelize filter point clouds
    print("[PROCESSING] Filtering point clouds in multiprocess process...")
    pcd_list = filter_pcd_mp(args, pcd_list)
    
    ### Plotting to scene pcd
    # Assign predicted instances to the voxels of the scene point cloud.
    # Load scene data if applicable
    # Determine scene path based on dataset type and scene
    scene_path = None
    if args.dataset_type == "scannetv2":
        if scene_name in args.train_scenes:
            # scene_path = join(args.pcd_path, "train", scene_name + ".pth")
            scene_path = join(args.pcd_path, "train", scene_name + ".ply")
        elif scene_name in args.val_scenes:
            # scene_path = join(args.pcd_path, "val", scene_name + ".pth")
            scene_path = join(args.pcd_path, "val", scene_name + ".ply")
    elif args.dataset_type == "replica" and scene_name in args.replica_scenes:
        scene_path = join(args.pcd_path, scene_name + "_mesh.ply")
    elif args.dataset_type == "real_world" and scene_name in args.real_world_scenes:
        scene_path = join(args.pcd_path, scene_name + ".ply")
     
    if scene_path and args.dataset_type in ["scannetv2", "replica", "real_world"]:
        scene_pcd = o3d.io.read_point_cloud(scene_path)
        scene_coord = np.copy(np.asarray(scene_pcd.points))
        scene_colors = np.copy(np.asarray(scene_pcd.colors))

    # Build a KD-Tree for efficient nearest neighbor search
    kdtree = cKDTree(scene_coord)
    # Query the nearest neighbor for each point (k=2 returns the point itself and its nearest neighbor)
    distances, _ = kdtree.query(scene_coord, k=2)
    # The first column of `distances` is the distance to itself (0), so we take the second column
    nearest_neighbor_distances = distances[:, 1]
    # The smallest non-zero distance should give you the voxel size
    scene_voxel_size = np.min(nearest_neighbor_distances)
    print(f"[INFO] Estimated voxel size: {scene_voxel_size}")
    
    scene_coord = torch.tensor(scene_coord).cuda().contiguous().float()
    scene_offset = torch.tensor(scene_coord.shape[0]).cuda()
    g2s_list = []

    print("[PROCESSING] Plotting coord world to scene voxels...")        
    for pd in tqdm(pcd_list):
        # Plot the instance coord to the scene point cloud.
        inst_coord = torch.tensor(pd["coord"]).cuda().contiguous().float()
        inst_offset = torch.tensor(inst_coord.shape[0]).cuda()
        inst_group = pd["group"]
        
        # For the scene point cloud, get the closest points of the instance point cloud
        indices, dis = pointops.knn_query(1, inst_coord, inst_offset, scene_coord, scene_offset)        
        indices = indices.cpu().numpy().flatten()
        
        # Remove -1 values from indices
        indices = indices[indices != -1]
        # Check if indices is empty after filtering out -1 values
        if len(indices) == 0:
            # Skip this iteration as there are no valid indices
            continue

        # Now this frame has a scene shape array that is represented with each instances' group id in the frame
        scene_group = inst_group[indices].astype(np.int16)
        mask_dis = dis.reshape(-1).cpu().numpy() > 0.03
        scene_group[mask_dis] = -1
        
        # Initialize the dictionary to hold the group_id -> scene_coord indices mapping
        group_to_scene_indices = defaultdict(list)

        for group_id in np.unique(scene_group):
            if group_id == -1:
                continue

            # Get the indices in scene_coord for this group
            mapped_indices = np.where(scene_group == group_id)[0]

            # Store these indices in the dictionary
            group_to_scene_indices[group_id].extend(mapped_indices)

        # Now, group_to_scene_indices holds each group_id with the corresponding scene_coord indices, excluding -1
        g2s_list.append(group_to_scene_indices)
    
    scene_coord = scene_coord.cpu().numpy()    
    
    
    ### Merge 3D segmented point clouds        
    print("[PROCESSING] Merging 3D segmented point clouds...")
    while len(g2s_list) != 1:
        print(f">>> Remaining frames: {len(g2s_list)}", flush=True)
        
        pool_size = min(math.ceil(len(g2s_list) / 2), mp.cpu_count())  # Assume pairwise needs half of g2s_list

        with mp.Pool(processes=pool_size) as pool:
            # Prepare tasks as pairs of frames
            tasks = [(frame1, frame2, args) for frame1, frame2 in pairwise_frames(g2s_list)]

            # Use imap with a progress bar to process each task in parallel and retain order
            merged_g2s_list = list(tqdm(pool.imap(merge_2_frames_FGP, tasks), total=len(tasks)))

        g2s_list = merged_g2s_list
        

    ### Dominant Voting
    param1 = "0.05"
    param2 = "20"
    mesh_seg_dir = 'data/scannetv2/input/org_process/mesh_segmentation/'+f'{param1}_{param2}'
    suffix = f'_vh_clean_2.{param1}0000.segs.json'
    mesh_seg_path = join(mesh_seg_dir, scene_name + suffix)
    
    print("[PROCESSING] Voting the most dominant group to the mesh segmentation...")
    dominant_group_indices = dominant_voting(g2s_list[0], mesh_seg_path)
    
    
    ### Saving the result
    seg_dict = dict(coord=scene_coord, group=dominant_group_indices) # final 3D segmentation result
    print(seg_dict["coord"].shape, seg_dict["group"].shape)
        
    if args.visualize_result: # visualize the raw merged result
        visualize_pcd_with_random_colors(seg_dict["coord"], seg_dict["group"])
    # np.save(f"visualization/{len(pcd_list)}.npy", np.array(merged_pcd_list, dtype=object))

    seg_dict["group"] = nearest_neighbor_smoothing(seg_dict["coord"], seg_dict["group"], k=80)
    seg_dict["group"] = num_to_natural(remove_small_group(seg_dict["group"], args.th))

    scene_coord = torch.tensor(scene_coord).cuda().contiguous()
    
    new_offset = torch.tensor(scene_coord.shape[0]).cuda()
    gen_coord = torch.tensor(seg_dict["coord"]).cuda().contiguous().float()
    offset = torch.tensor(gen_coord.shape[0]).cuda()
    gen_group = seg_dict["group"]
    indices, dis = pointops.knn_query(1, gen_coord, offset, scene_coord, new_offset)
    indices = indices.cpu().numpy()
    group = gen_group[indices.reshape(-1)].astype(np.int16)
    mask_dis = dis.reshape(-1).cpu().numpy() > 0.6
    group[mask_dis] = -1
    group = group.astype(np.int16)
    
    scene_coord = scene_coord.cpu().numpy()

    if args.visualize_result: # visualize the result fitted to the original scan data
        if args.dataset_type == "scannetv2":
            visualize_pcd_with_random_colors(scene_coord, group)
        elif args.dataset_type == "replica":
            visualize_pcd_with_random_colors(scene_coord, group)
        
    torch.save(num_to_natural(group), join(args.save_path, scene_name + ".pth"))


def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='Segment Anything on ScanNet.')
    parser.add_argument('--working_dir', type=str, default=os.getcwd(), help='root of the working directory')
    parser.add_argument('--dataset_type', type=str, required=True, help='Dataset to be processed.')
    parser.add_argument('--mask_model', type=str, help='mask generator model')
    parser.add_argument('--part', type=int, help='subprocess of each scenes')
    parser.add_argument('--subprocess_num', type=int, help='Total subprocess')
    parser.add_argument('--rgb_path', type=str, help='the path of color data')
    parser.add_argument('--pcd_path', type=str, default='', help='the path of pointcload data')
    parser.add_argument('--save_path', type=str, help='Where to save the pcd results')
    parser.add_argument('--save_2dmask_path', type=str, default='', help='Where to save 2D segmentation result from SAM')
    parser.add_argument('--sam_checkpoint_path', type=str, default="weights/sam_vit_h_4b8939.pth", help='the path of checkpoint for SAM')
    parser.add_argument('--th', type=int, default=50, help='threshold of ignoring small groups to avoid noise pixel')

    parser.add_argument('--scannetv2_train_path', type=str, default='scannet-preprocess/meta_data/scannetv2_train.txt', help='the path of scannetv2_train.txt')
    parser.add_argument('--scannetv2_val_path', type=str, default='scannet-preprocess/meta_data/scannetv2_val.txt', help='the path of scannetv2_val.txt')
    parser.add_argument('--replica_path', type=str, default='data/replica/input/replica_processed/replica.txt', help='the path of replica.txt')
    parser.add_argument('--real_world_path', type=str, default='data/real_world/input/real_world.txt', help='the path of real_world.txt')
    
    parser.add_argument('--train_scenes', default="")
    parser.add_argument('--val_scenes', default="")
    parser.add_argument('--replica_scenes', default="")
    parser.add_argument('--real_world_scenes', default="")

    parser.add_argument('--img_size', type=tuple_type, default=(480, 640)) #480,640 720,1280
    parser.add_argument('--voxel_size', type=float, default=0.03) #0.03
    parser.add_argument('--image_iter', type=int, default=10) #10
    
    parser.add_argument('--overlap_criterion', default="lo", help='Overlap criterion of 3D segments. Large overlpa or small overlap.') #10

    # parser.add_argument('--use_only_raw_depth', action='store_true', help='Use only raw camera depth image.')
    # parser.add_argument('--build_dense_pc', action='store_true', help='Build raw point cloud to render depth image from dense point cloud.')
    # parser.add_argument('--supplement_depth', action='store_true', help='Supplement lacking raw camera depth image with rendered depth')
    parser.add_argument('--depth_type', type=str, default='sup_depth', help="Choose depth type among supplemented(sup), point cloud(pc), raw(raw) depths.")

    parser.add_argument('--visualize_dense_pcd', action='store_true', help='Visualize the result of built dense point cloud.')
    parser.add_argument('--visualize_segments', action='store_true', help='Visualize 3D segments for debugging.')
    parser.add_argument('--visualize_result', action='store_true', help='Visualize the final results for debugging.')
    
    parser.add_argument('--print_info', type=bool, default=False)
    parser.add_argument('--developing', action='store_true')

    args = parser.parse_args()
    return args

def get_cf_setting():
    setting = {
    "config-file": "./libs/detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/cropformer_hornet_3x.yaml",
    "confidence-threshold": 0.5,
    "model-weights": "./weights/CropFormer_hornet_3x_03823a.pth"
    }
    
    return setting

def setup_cfg(setting):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(setting["config-file"])
    cfg.merge_from_list(["MODEL.WEIGHTS", setting["model-weights"]])
    cfg.freeze()
    return cfg

if __name__ == '__main__':    
    module_directory = './libs/detectron2/projects/CropFormer'
    # Add the directory to sys.path
    if module_directory not in sys.path:
        sys.path.append(module_directory)
        
    from detectron2.config import get_cfg
    from detectron2.data.detection_utils import read_image
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.utils.logger import setup_logger
    from mask2former import add_maskformer2_config
    from demo_cropformer.predictor import VisualizationDemo
    
    args = get_args()
    print("Arguments:")
    print(args)

    # clipcapture = ClipCapture(model_name='ViT-bigG-14', pretrained='laion2b_s39b_b160k')    
    
    if args.dataset_type == "scannetv2":
        with open(args.scannetv2_train_path) as train_file:
            args.train_scenes = train_file.read().splitlines()
        with open(args.scannetv2_val_path) as val_file:
            args.val_scenes = val_file.read().splitlines()
    elif args.dataset_type == "replica":
        with open(args.replica_path) as replica_file:
            args.replica_scenes = replica_file.read().splitlines()
    elif args.dataset_type == "real_world":            
        with open(args.real_world_path) as real_world_file:
            args.real_world_scenes = real_world_file.read().splitlines()
    else:
        assert False, "Data must be scannetv2, replica, real_world"

    if args.mask_model == "sam":
        print(">>> Setting SAM Model")
        # mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint=args.sam_checkpoint_path).to(device="cuda"))
        mask_generator = "debug"
    elif args.mask_model == "oneformer":
        print(">>> Setting oneformer Model")
        processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_dinat_large")
        model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_dinat_large")
        mask_generator = [processor, model]
    elif args.mask_model == "cropformer":
        print(">>> Setting cropformer Model")
        if not args.developing:
            mp.set_start_method("spawn", force=True)
            cf_setting = get_cf_setting()
            cfg = setup_cfg(cf_setting)
            mask_generator = VisualizationDemo(cfg)
        else:
            mask_generator = "debug"
    else:
        assert False, "Choose mask model between sam, oneformer and cropformer!"

    voxelize = Voxelize(voxel_size=args.voxel_size, mode="train", keys=("coord", "group"))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    scene_names = sorted(os.listdir(args.rgb_path))

    print('Processing OVMap...')
    print("Total scenes:", len(scene_names))
    print("Scene names:", scene_names)

    # Distributing subprocesses
    # scene_names[2:30], scene_names[30:60], scene_names[60:90], scene_names[90:120], scene_names[120:150], scene_names[150:180] 
    # scene_names[180:210], scene_names[210:240], scene_names[240:270], scene_names[270:300]
    full_process = True
    part_length = len(scene_names) // args.subprocess_num
    start = args.part * part_length
    if args.part == args.subprocess_num - 1:
        end = len(scene_names)
    else:
        end = start + part_length
    print(f"Subprocess {start} : {end}")

    # Scenes with good output
    # 'scene0011_01', 'scene0084_02', 'scene0549_00', 'scene0552_00', 'scene0552_01', 'scene0616_00', 'scene0655_00', 'scene0655_01', 'scene0655_02', 'scene0686_02'
    choosen_scene = ['scene0549_00']

    for scene_name in scene_names[start:end]: #[::-1]: #[:1]:
        if "scene_115" in scene_name:
            continue
        
        if full_process:
            scene_pth_path = join(args.save_path, scene_name+'.pth')
            if not os.path.exists(scene_pth_path):
                seg_pcd(args, scene_name, mask_generator, voxelize)
                manage_temp_directory("data/temp_dense_pcd")
            else:
                print(f"File '{scene_pth_path}' already exists.")
        else:
            if scene_name in choosen_scene:
                scene_pth_path = join(args.save_path, scene_name+'.pth')
                if not os.path.exists(scene_pth_path):
                    seg_pcd(args, scene_name, mask_generator, voxelize)
                else:
                    print(f"File '{scene_pth_path}' already exists.")

    # scene_name = "/home/robi/PycharmProjects/SegmentAnything3D/scene_414_full"
    # seg_pcd(scene_name, args.rgb_path, args.pcd_path, args.save_path, mask_generator, args.voxel_size, voxelize, args.th, train_scenes, val_scenes, args.save_2dmask_path)
