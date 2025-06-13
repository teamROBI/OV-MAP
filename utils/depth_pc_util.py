import numpy as np
import torch
import open3d as o3d
from scipy.spatial import KDTree
import cv2
from os.path import join
import os, json
from tqdm import tqdm
from numba import jit
from sklearn.cluster import DBSCAN
# from utils.util import *
from utils.util import num_to_natural, remove_small_group, visualize_partition
import multiprocessing as mp
import math


def build_scene_point_cloud(args, file_names, scene_name, vis_final=False):
    dense_scene_pcd_dir = "data/temp_dense_pcd"
    if not os.path.exists(dense_scene_pcd_dir):
        os.makedirs(dense_scene_pcd_dir)
    
    if os.path.exists(join(dense_scene_pcd_dir, f"{scene_name}.pcd")):
        dense_scene_pcd = o3d.io.read_point_cloud(join(dense_scene_pcd_dir, f"{scene_name}.pcd"))
        print('[INFO] Using temporary saved dense point cloud!')
        return dense_scene_pcd
    
    intrinsic_path = join(args.rgb_path, scene_name, 'intrinsics', 'intrinsic_depth.txt')
    depth_intrinsic = np.loadtxt(intrinsic_path)

    if scene_name in args.train_scenes:
        pc_path = join(args.pcd_path, "train", scene_name + ".pth")
        voxelized_pcd = o3d.geometry.PointCloud()
        voxelized_pcd.points = o3d.utility.Vector3dVector(torch.load(pc_path)["coord"])
    elif scene_name in args.val_scenes:
        pc_path = join(args.pcd_path, "val", scene_name + ".pth")
        voxelized_pcd = o3d.geometry.PointCloud()
        voxelized_pcd.points = o3d.utility.Vector3dVector(torch.load(pc_path)["coord"])
    elif scene_name in args.replica_scenes:
        pc_path = join(args.pcd_path, scene_name + ".pth")
        voxelized_pcd = o3d.geometry.PointCloud()
        voxelized_pcd.points = o3d.utility.Vector3dVector(torch.load(pc_path)[0])
    elif scene_name in args.real_world_scenes:
        if os.path.exists(join(args.pcd_path, scene_name + ".ply")):
            voxelized_pcd = o3d.io.read_point_cloud(join(args.pcd_path, scene_name + ".ply"))
        else:
            voxelized_pcd = o3d.io.read_point_cloud(join(args.pcd_path, scene_name + ".pcd"))
        # o3d.visualization.draw_geometries([voxelized_pcd], width=1080, height=720)
    
    dense_scene_pcd = o3d.geometry.PointCloud()
    # dense_scene_pcd = build_scene_point_cloud_parallel(file_names, scene_name, rgb_path, depth_intrinsic)

    print("[INFO] Start building raw scene point cloud")
    if args.image_iter == 1:
        image_iter = 2 #10
    else:
        image_iter = args.image_iter
        
    for file_name in tqdm(file_names[::image_iter]): #2:30
        pose = join(args.rgb_path, scene_name, 'pose', file_name[0:-4] + '.txt')
        depth = join(args.rgb_path, scene_name, 'depth', file_name[0:-4] + '.png')
        # rgb = join(args.rgb_path, scene_name, 'color', file_name[0:-4] + '.jpg')

        pose = np.loadtxt(pose)
        depth_img = cv2.imread(depth, -1)

        points_world = camera2world(depth_img, depth_intrinsic, pose)

        # Filtering
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(points_world[:, :3])
        temp_pcd = temp_pcd.select_by_index(np.where(points_world[:, 2] <= 3)[0])
        temp_pcd, ind = temp_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        temp_pcd, ind = temp_pcd.remove_radius_outlier(nb_points=20, radius=0.05)

        # Initialize points at first view
        if int(file_name[0:-4]) == 0:
            dense_scene_pcd.points = temp_pcd.points
        else:
            dense_scene_pcd.points.extend(temp_pcd.points)

    # Final Cluster filtering. Spatial filter raw with voxelized pcd
    if vis_final: # Scene
        print(">>> Before filtered raw scene point cloud total points:", len(dense_scene_pcd.points))
        o3d.visualization.draw_geometries([dense_scene_pcd])

    # # Iterate through larger points 7 minute
    # distance_threshold = 0.08
    # kdtree_voxelized = o3d.geometry.KDTreeFlann(voxelized_pcd)
    # keep_indices = []

    # for i, point in enumerate(tqdm(dense_scene_pcd.points)):
    #     [k, idx, _] = kdtree_voxelized.search_knn_vector_3d(point, 1)
    #     nearest_point = np.asarray(voxelized_pcd.points)[idx[0], :]
    #     distance = np.linalg.norm(np.asarray(point) - nearest_point)
    #     if distance <= distance_threshold:
    #         keep_indices.append(i)
            
    # Iterate through larger points 7 minute
    distance_threshold = 0.08
    kdtree_voxelized = o3d.geometry.KDTreeFlann(voxelized_pcd)
    dense_points = np.asarray(dense_scene_pcd.points)

    # Use numpy for faster distance calculations
    voxelized_points = np.asarray(voxelized_pcd.points)
    keep_indices = []

    for i, point in enumerate(tqdm(dense_points)):
        [k, idx, _] = kdtree_voxelized.search_knn_vector_3d(point, 1)
        nearest_point = voxelized_points[idx[0]]
        distance = np.linalg.norm(point - nearest_point)
        if distance <= distance_threshold:
            keep_indices.append(i)

    dense_scene_pcd = dense_scene_pcd.select_by_index(keep_indices)

    print("[INFO] Building raw scene point cloud done! Total points:", len(dense_scene_pcd.points))
    if vis_final:
        o3d.visualization.draw_geometries([dense_scene_pcd])

    # Save the combined point cloud
    o3d.io.write_point_cloud(f"{dense_scene_pcd_dir}/{scene_name}.pcd", dense_scene_pcd)
    
    return dense_scene_pcd


def camera2world(depth_img, depth_intrinsic, pose, depth_shift=1000.0):
    if depth_img.dtype in [np.float64, np.float32] and np.max(depth_img) < 100:
        depth_shift = 1

    x, y = np.meshgrid(np.linspace(0, depth_img.shape[1] - 1, depth_img.shape[1]),
                       np.linspace(0, depth_img.shape[0] - 1, depth_img.shape[0]))
    uv_depth = np.zeros((depth_img.shape[0], depth_img.shape[1], 3))
    uv_depth[:, :, 0] = x
    uv_depth[:, :, 1] = y
    uv_depth[:, :, 2] = depth_img / depth_shift
    uv_depth = np.reshape(uv_depth, [-1, 3])
    uv_depth = uv_depth[np.where(uv_depth[:, 2] != 0), :].squeeze()

    fx, fy, cx, cy = depth_intrinsic[0, 0], depth_intrinsic[1, 1], depth_intrinsic[0, 2], depth_intrinsic[1, 2]
    bx, by = depth_intrinsic[0, 3], depth_intrinsic[1, 3]

    points = np.ones((uv_depth.shape[0], 4))
    X = (uv_depth[:, 0] - cx) * uv_depth[:, 2] / fx + bx
    Y = (uv_depth[:, 1] - cy) * uv_depth[:, 2] / fy + by
    points[:, 0], points[:, 1], points[:, 2] = X, Y, uv_depth[:, 2]
    points_world = np.dot(points, np.transpose(pose))

    return points_world


def render_point_cloud_to_depth_image(point_cloud, pose, intrinsic, image_shape, use_radius_z_buffer=False):
    # Invert the pose to transform points from world to camera coordinates
    pose_inv = np.linalg.inv(pose)

    # Extract the coordinates from the point cloud
    coords_world = point_cloud.copy()  # Shape: (N, 3)

    # Add a column of ones to the coordinates for matrix multiplication
    coords_world_homogeneous = np.hstack([coords_world, np.ones((coords_world.shape[0], 1))])

    # Transform coordinates to camera frame
    coords_camera_homogeneous = np.dot(coords_world_homogeneous, pose_inv.T)

    # Divide by the homogeneous coordinate to get Cartesian coordinates
    coords_camera = coords_camera_homogeneous[:, :3] / coords_camera_homogeneous[:, [3]]

    # Project points onto the image plane using the intrinsic matrix
    coords_image_homogeneous = np.dot(coords_camera, intrinsic[:3, :3].T)

    # Convert homogeneous image coordinates to Cartesian coordinates
    u = coords_image_homogeneous[:, 0] / coords_image_homogeneous[:, 2]
    v = coords_image_homogeneous[:, 1] / coords_image_homogeneous[:, 2]
    depth_values = coords_image_homogeneous[:, 2]
    depth_idx = np.where(depth_values > 0)[0]

    # Draw depth with non-occluded objects
    depth_image = np.full(image_shape, np.inf, dtype=np.float64)
    depth_image = update_depth_image_jit(depth_idx, u, v, depth_values, image_shape, depth_image)
    depth_image[depth_image == np.inf] = 0

    # backproject_rgbd_to_pointcloud(rgb_image, depth_image, intrinsic) # Check if the 3d crop is correct
    # print(">>> Depth cropped:", len(u), np.where(depth_image != 0)[0].shape)

    return depth_image


def render_3d_seg_to_2d_seg(pcd_queue, pose, intrinsic, image_shape, per_instance=False):    
    if type(pcd_queue) == dict:
        # Extract the coordinates from the point cloud
        coords_world = pcd_queue["coord"] # coord: (N, 3)
        groups_world = pcd_queue["group"] # group: (N, 1)
    else:
        coords_world, groups_world = [], []
        for pcd_dict in list(pcd_queue)[:-1]:
            coords_world.extend(pcd_dict["coord"])
            groups_world.extend(pcd_dict["group"])
        coords_world, groups_world = np.array(coords_world), np.array(groups_world)
        
    if not per_instance:
        seg_image = tf_3d_to_2d(coords_world, groups_world, pose, intrinsic, image_shape, render_type="segmentation")
        return seg_image
    else:
        seg_image_dict = {}
        for id in np.unique(groups_world[groups_world != -1]):
            inst_coords_world = coords_world[groups_world == id]
            inst_groups_world = groups_world[groups_world == id]
            seg_image = tf_3d_to_2d(inst_coords_world, inst_groups_world, pose, intrinsic, image_shape, render_type="segmentation")            
            seg_image_dict[id] = (seg_image == id)
            
        return seg_image_dict
        

def tf_3d_to_2d(coords_world, groups_world, pose, intrinsic, image_shape, render_type):
    # Invert the pose to transform points from world to camera coordinates
    pose_inv = np.linalg.inv(pose)
    
    # Add a column of ones to the coordinates for matrix multiplication
    coords_world_homogeneous = np.hstack([coords_world, np.ones((coords_world.shape[0], 1))])

    # Transform coordinates to camera frame
    coords_camera_homogeneous = np.dot(coords_world_homogeneous, pose_inv.T)

    # Divide by the homogeneous coordinate to get Cartesian coordinates
    coords_camera = coords_camera_homogeneous[:, :3] / coords_camera_homogeneous[:, [3]]

    # Project points onto the image plane using the intrinsic matrix
    coords_image_homogeneous = np.dot(coords_camera, intrinsic[:3, :3].T)

    # Convert homogeneous image coordinates to Cartesian coordinates
    u = coords_image_homogeneous[:, 0] / coords_image_homogeneous[:, 2]
    v = coords_image_homogeneous[:, 1] / coords_image_homogeneous[:, 2]
    depth_values = coords_image_homogeneous[:, 2]
    depth_idx = np.where(depth_values > 0)[0]
    
    if render_type == "segmentation":
        # Initialize the 2D instance segmentation image and depth buffer
        rendered_image = np.full(image_shape, -1, dtype=np.int32)  # Instance IDs, initialized to -1
        depth_buffer = np.full(image_shape, np.inf, dtype=np.float64)  # Depth buffer
        rendered_image = update_segmentation_image_jit(depth_idx, u, v, depth_values, groups_world, image_shape, rendered_image, depth_buffer)
        
    elif render_type == "depth":
        # Draw depth with non-occluded objects
        rendered_image = np.full(image_shape, np.inf, dtype=np.float64)
        rendered_image = update_depth_image_jit(depth_idx, u, v, depth_values, image_shape, rendered_image)
        rendered_image[rendered_image == np.inf] = 0

    return rendered_image  


@jit(nopython=True)  # Decorator to compile this function with Numba
def update_segmentation_image_jit(depth_idx, u, v, depth_values, groups_world, image_shape, seg_image, depth_buffer):
    for idx in depth_idx:
        x, y = int(round(u[idx])), int(round(v[idx]))
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            # Update only if the current point is closer than the existing one
            if depth_values[idx] < depth_buffer[y, x]:
                depth_buffer[y, x] = depth_values[idx]
                seg_image[y, x] = groups_world[idx]

    return seg_image


@jit(nopython=True)  # Decorator to compile this function with Numba
def update_depth_image_jit(depth_idx, u, v, depth_values, image_shape, depth_image):
    for idx in depth_idx:
        x, y = int(round(u[idx])), int(round(v[idx]))
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            depth_image[y, x] = min(depth_image[y, x], depth_values[idx])

    return depth_image


def update_depth_image(depth_idx, u, v, depth_values, image_shape, depth_image):
    for idx in tqdm(depth_idx):
        x, y = int(round(u[idx])), int(round(v[idx]))
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            depth_image[y, x] = min(depth_image[y, x], depth_values[idx])

    return depth_image


def backproject_rgbd_to_pointcloud(rgb_image, depth_image, intrinsic):
    print(">>> Depth type and max value:", depth_image.dtype, np.max(depth_image))
    if depth_image.dtype != np.uint16:
        depth_image = (depth_image * 1000).astype(np.uint16)
        print(">>> Edited Depth type and max value:", depth_image.dtype, np.max(depth_image))

    # Create Open3D RGBDImage from the RGB and depth images
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb_image),
        o3d.geometry.Image(depth_image),
        depth_scale=1000.0,  # adjust this scale according to your depth image format
        # depth_trunc=0.0003,  # adjust truncation value to remove distant points
        convert_rgb_to_intensity=False
    )

    # Create Open3D PinholeCameraIntrinsic object from the intrinsic parameters
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=rgb_image.shape[1],
        height=rgb_image.shape[0],
        fx=intrinsic[0, 0],
        fy=intrinsic[1, 1],
        cx=intrinsic[0, 2],
        cy=intrinsic[1, 2]
    )

    # Create point cloud from the RGBD image and the camera intrinsics
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        camera_intrinsic
    )
    o3d.visualization.draw_geometries([point_cloud], width=1080, height=720)

    coord = np.asarray(point_cloud.points)
    group_ids = (np.average(np.asarray(point_cloud.colors), axis=1) * 255).astype("int")

    # Visualize the point cloud
    coord, group_ids = cluster_filtering(coord, group_ids)
    debug_pcd = o3d.geometry.PointCloud()
    debug_pcd.points = o3d.utility.Vector3dVector(coord[:, :3])
    unique_labels = np.unique(group_ids)
    colors = {label: np.random.rand(3) for label in unique_labels}
    debug_pcd.colors = o3d.utility.Vector3dVector([colors[label] for label in group_ids])
    o3d.visualization.draw_geometries([debug_pcd], width=1080, height=720)
    

def visualize_clusters(seg_points, indices, noise_removed=False):
    # Visualization of clusters (debug mode)
    if noise_removed:
        removed_indices = np.setdiff1d(np.arange(len(seg_points)), indices)
        noise_removed_indices = indices
    else:
        removed_indices = indices
    
    debug_pcd = o3d.geometry.PointCloud()
    debug_pcd.points = o3d.utility.Vector3dVector(seg_points)

    # Color points: green for noise_removed_indices, red for removed_indices
    colors = np.zeros_like(seg_points)
    
    if noise_removed:
        colors[noise_removed_indices, :] = [0, 1, 0]  # Green for retained clusters
    colors[removed_indices, :] = [1, 0, 0]  # Red for removed points

    debug_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([debug_pcd])
    

def cluster_filtering(points, group_ids, eps=0.1, min_points=5, min_whole_cluster=50, min_part_cluster=None, depth_th=2.5, debug=False):
    filtered_points = np.empty((0, points.shape[1]))  # Empty array to accumulate points
    filtered_group_ids = np.empty(0, dtype=group_ids.dtype)  # Empty array for group IDs

    # Get unique group IDs (excluding any potential noise labels like -1)
    unique_group_ids = np.unique(group_ids[group_ids > -1])

    for group_id in unique_group_ids:
        # Get the points and group IDs for the current group
        indices = np.where(group_ids == group_id)[0]
        seg_points = points[indices]
        seg_group_ids = group_ids[indices]

        # Check the distance in the Z-dimension
        seg_dist = np.median(seg_points[:, 2])

        if len(seg_points) > min_whole_cluster and seg_dist < depth_th:
            # DBSCAN clustering
            dbscan_eps = eps if len(seg_points) > 30 else 0.5
            db = DBSCAN(eps=dbscan_eps, min_samples=min_points).fit(seg_points)
            labels = db.labels_

            unique_labels, counts = np.unique(labels, return_counts=True)

            if min_part_cluster is None:
                min_part_cluster = np.max(counts) * 0.6

            # Remove noise points and select valid clusters
            noise_removed_labels = unique_labels[(counts >= min_part_cluster) & (unique_labels != -1)]
            noise_removed_indices = np.where(np.isin(labels, noise_removed_labels))[0]

            # Append filtered points
            filtered_points = np.concatenate([filtered_points, seg_points[noise_removed_indices]], axis=0)
            filtered_group_ids = np.concatenate([filtered_group_ids, seg_group_ids[noise_removed_indices]], axis=0)

            if debug:
                print(f"[DEBUG] len(seg_points): {len(seg_points)}, noise_removed_indices: {len(noise_removed_indices)}")
                visualize_clusters(seg_points, noise_removed_indices, noise_removed=True)

            # Only add points from max_cluster_indices if they are not already in noise_removed_indices
            max_cluster_group_id = unique_labels[np.argmax(counts)]
            max_cluster_indices = np.where(labels == max_cluster_group_id)[0]
            unique_max_cluster_indices = np.setdiff1d(max_cluster_indices, noise_removed_indices)

            if len(unique_max_cluster_indices) > 0:
                filtered_points = np.concatenate([filtered_points, seg_points[unique_max_cluster_indices]], axis=0)
                filtered_group_ids = np.concatenate([filtered_group_ids, seg_group_ids[unique_max_cluster_indices]], axis=0)

        elif debug:
            print(f"[DEBUG] len(seg_points):{len(seg_points)} > min_whole_cluster:{min_whole_cluster} and seg_dist:{seg_dist} < depth_th:{depth_th}")
            visualize_clusters(points[:, :3], indices)

    return filtered_points, filtered_group_ids


def nearest_neighbor_smoothing(xyz, labels, k=20): #, object_ids
    if len(labels) <= 1:
        print("[ERROR] Only one group id exists. Returning the original label.")
        return labels
    labels = labels.astype(np.int64)
    # Build a KDTree for efficient nearest neighbor searches
    tree = KDTree(xyz)

    # For each point, find the k nearest neighbors (excluding the point itself)
    _, indices = tree.query(xyz, k=min(k + 1, len(labels)))  # Ensure the query doesn't request more neighbors than available

    # Adjust labels based on majority voting of neighbors
    new_labels = labels.copy()
    for idx, neighbors in enumerate(indices):
        if len(neighbors) <= 1:
            continue  # If there are no neighbors, skip
        
        neighbor_labels = labels[neighbors[1:]]  # Exclude the point itself
        valid_labels = neighbor_labels[neighbor_labels >= 0]

        if len(valid_labels) > 0:
            most_common_label = np.bincount(valid_labels).argmax()
            new_labels[idx] = most_common_label
        else:
            # If all neighbors have -1 label, keep the original label
            # print(">>> All neighbors are noise!!!")
            new_labels[idx] = labels[idx]

    # # Find the unique labels and sort them
    # unique_labels = np.unique(new_labels)
    # unique_labels.sort()

    # # Create a mapping from original labels to new labels (0, 1, 2, ...)
    # label_mapping = {original_label: new_label - 1 for new_label, original_label in enumerate(unique_labels)}

    # # Apply the mapping to get contiguous labels
    # contiguous_labels = np.vectorize(label_mapping.get)(new_labels)
    # # remapped_object_ids = {new_label: object_ids[original_label] for original_label, new_label in label_mapping.items() if original_label in object_ids}

    return new_labels#, remapped_object_ids




# Global variable placeholder for args-> filter pcd mp
args_global = None
    
def init_worker(init_args):
    global args_global
    args_global = init_args  # Initialize the global args in each worker

def filter_pcd_worker(pcd_dict, debug=False):
    points_world = pcd_dict["coord"]
    group_ids = pcd_dict["group"]

    # Use `args` as needed in the function, no need to pass it explicitly
    if args_global.dataset_type == "scannetv2" or args_global.dataset_type == "real_world":
        group_ids = nearest_neighbor_smoothing(points_world, group_ids)
    
    if args_global.dataset_type == "replica":
        filterd_points_world, filtered_group_ids = cluster_filtering(points_world, group_ids, debug=debug, min_whole_cluster=8)
    else:
        filterd_points_world, filtered_group_ids = cluster_filtering(points_world, group_ids, debug=debug)

    # Continue with the rest of the code...

    pcd_dict["coord"] = filterd_points_world
    pcd_dict["group"] = filtered_group_ids

    return pcd_dict

def filter_pcd_mp(args, pcd_list):
    pool_size = min(math.ceil(len(pcd_list) / 2), mp.cpu_count())
    
    with mp.Pool(pool_size, initializer=init_worker, initargs=(args,)) as pool:
        # Pass only `pcd_dict` to `filter_pcd`, since `args` is now global within each worker
        pcd_list = list(tqdm(pool.imap(filter_pcd_worker, pcd_list), total=len(pcd_list)))
        
    return pcd_list


def filter_pcd(args, pcd_dict, color_name=0, debug=False):
    points_world = pcd_dict["coord"]
    group_ids = pcd_dict["group"]
    
    if int(color_name[0:-4]) % 1 == 0 and debug:
        debug_pcd = o3d.geometry.PointCloud()

        # Visualize non-filtered
        debug_pcd.points = o3d.utility.Vector3dVector(points_world[:, :3])
        unique_labels = np.unique(group_ids)
        colors = {label: np.random.rand(3) for label in unique_labels}
        debug_pcd.colors = o3d.utility.Vector3dVector([colors[label] for label in group_ids])
        o3d.visualization.draw_geometries([debug_pcd])

    if args.dataset_type == "scannetv2" or args.dataset_type == "real_world":
        group_ids = nearest_neighbor_smoothing(points_world, group_ids) #, filtered_object_ids , object_ids
        
    if args.dataset_type == "replica":
        filterd_points_world, filtered_group_ids = cluster_filtering(points_world, group_ids, debug=debug, min_whole_cluster=8) #, filtered_object_ids
    else:
        filterd_points_world, filtered_group_ids = cluster_filtering(points_world, group_ids, debug=debug) #, filtered_object_ids
    # filtered_group_ids = num_to_natural(filtered_group_ids)

    if int(color_name[0:-4]) % 1 == 0 and debug:
        # Visualize filtered
        debug_pcd.points = o3d.utility.Vector3dVector(filterd_points_world[:,:3])
        unique_labels = np.unique(filtered_group_ids)
        colors = {label: np.random.rand(3) for label in unique_labels}
        debug_pcd.colors = o3d.utility.Vector3dVector([colors[label] for label in filtered_group_ids])
        o3d.visualization.draw_geometries([debug_pcd])
    
    pcd_dict["coord"] = filterd_points_world
    pcd_dict["group"] = filtered_group_ids

    return pcd_dict


def filter_pcd_test(pcd_dict, debug=False):
    points_world = pcd_dict["coord"]
    group_ids = pcd_dict["group"]
    
    if debug:
        debug_pcd = o3d.geometry.PointCloud()

        # Visualize non-filtered
        debug_pcd.points = o3d.utility.Vector3dVector(points_world[:, :3])
        unique_labels = np.unique(group_ids)
        colors = {label: np.random.rand(3) for label in unique_labels}
        debug_pcd.colors = o3d.utility.Vector3dVector([colors[label] for label in group_ids])
        o3d.visualization.draw_geometries([debug_pcd])

    group_ids = nearest_neighbor_smoothing(points_world, group_ids) #, filtered_object_ids , object_ids
    filterd_points_world, filtered_group_ids = cluster_filtering(points_world, group_ids, debug=debug) #, filtered_object_ids

    if debug:
        # Visualize filtered
        debug_pcd.points = o3d.utility.Vector3dVector(filterd_points_world[:,:3])
        unique_labels = np.unique(filtered_group_ids)
        colors = {label: np.random.rand(3) for label in unique_labels}
        debug_pcd.colors = o3d.utility.Vector3dVector([colors[label] for label in filtered_group_ids])
        o3d.visualization.draw_geometries([debug_pcd])
    
    pcd_dict["coord"] = filterd_points_world
    pcd_dict["group"] = filtered_group_ids

    return pcd_dict


def voxel_voting(save_mesh_seg, save_pcd_path, pth_dir_path, vis_path=None, debug=False):
    # ply = o3d.io.read_point_cloud(ply_dir_path)
    # coord = np.asarray(ply.points)
    data = torch.load(pth_dir_path)
    coord = data["coord"]

    sam3d_group_ids = torch.load(save_pcd_path)
    sam3d_group_ids = num_to_natural(remove_small_group(sam3d_group_ids, 20))
    with open(save_mesh_seg) as f:
        segments = json.load(f)
        mesh_group_ids = np.array(segments['segIndices'])
    print(coord.shape, sam3d_group_ids.shape, mesh_group_ids.shape)
    # print(np.unique(sam3d_group_ids, return_counts=True))
    # print(np.unique(mesh_group_ids, return_counts=True))

    # valid_indices = np.where(sam3d_group_ids != -1)[0]
    # coord = coord[valid_indices]
    # sam3d_group_ids = sam3d_group_ids[valid_indices]
    # mesh_group_ids = mesh_group_ids[valid_indices]
    # print(coord.shape, sam3d_group_ids.shape, mesh_group_ids.shape)
    # print(np.unique(sam3d_group_ids, return_counts=True))
    # print(np.unique(mesh_group_ids, return_counts=True))

    sam3d_unique_group_ids = np.unique(sam3d_group_ids)
    mesh_unique_group_ids = np.unique(mesh_group_ids)

    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(coord)
    kdtree_scene = o3d.geometry.KDTreeFlann(scene_pcd)
    print(len(scene_pcd.points))
    
    print(">>> Voting....", os.path.splitext(os.path.basename(pth_dir_path))[0])
    for m_group_id in tqdm(mesh_unique_group_ids):
        scene_indices = np.where(mesh_group_ids == m_group_id)[0]
        group_vote = np.zeros(len(sam3d_unique_group_ids))
        for indice in scene_indices:
            [k, idx, _] = kdtree_scene.search_radius_vector_3d(coord[indice], 0.08)
            for i in idx:
                if sam3d_group_ids[i] != -1:
                    group_vote[sam3d_group_ids[i]] += 1

        # print(f"MG={m_group_id} voted to SG={np.argmax(group_vote)}")
        mesh_group_ids[mesh_group_ids == m_group_id] = np.argmax(group_vote) if np.max(group_vote) != 0 else -1

    if debug:
        unique_segs = np.unique(mesh_group_ids)
        seg_colors = {seg: np.random.uniform(0, 1, 3) for seg in unique_segs}  # Generate a random color for each segment
        colors = np.array([seg_colors[seg] for seg in mesh_group_ids])
        # Assign colors to ply
        scene_pcd.colors = o3d.utility.Vector3dVector(colors)
        # Visualize
        o3d.visualization.draw_geometries([scene_pcd])

        mesh_group_ids = nearest_neighbor_smoothing(coord, mesh_group_ids, k=50)
        unique_segs = np.unique(mesh_group_ids)
        seg_colors = {seg: np.random.uniform(0, 1, 3) for seg in unique_segs}  # Generate a random color for each segment
        colors = np.array([seg_colors[seg] for seg in mesh_group_ids])
        # Assign colors to ply
        scene_pcd.colors = o3d.utility.Vector3dVector(colors)
        # Visualize
        o3d.visualization.draw_geometries([scene_pcd])

    if vis_path != None:
        basename = os.path.splitext(os.path.basename(pth_dir_path))[0]
        visualize_partition(coord, mesh_group_ids, vis_path, basename)


def visualize_pcd_with_random_colors(points, labels):
    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    unique_labels = np.unique(labels)
    
    # Assign random colors to labels, but set -1 to [0.9, 0.9, 0.9]
    colors = {label: np.random.rand(3) if label != -1 else [0.9, 0.9, 0.9] for label in unique_labels}
    
    # Assign colors to the point cloud
    vis_pcd.colors = o3d.utility.Vector3dVector([colors[label] for label in labels])
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries([vis_pcd])
    
    
    
    
########## CDIS

def load_intrinsics(intrinsics_file):
    # Load the intrinsic camera parameters from file (assuming 3x3 matrix, last row is redundant)
    intrinsics = np.loadtxt(intrinsics_file)
    return intrinsics[:3, :3]  # Return 3x3 intrinsics (ignore the last row)

def supplement_depth(depth, syn_depth):
    # Replace zero values in the original depth image with values from syn_depth
    supplemented_depth = np.where(depth == 0, syn_depth, depth)
    return supplemented_depth

def backproject_to_3d(mask, depth, intrinsics, depth_shift=1000.0):
    if depth.dtype in [np.float64, np.float32] and np.max(depth) < 100:
        depth_shift = 1.0
        
    h, w = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]    
    # bx, by = intrinsics[0, 3], intrinsics[1, 3]
    
    # Meshgrid to get pixel coordinates
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # Only consider pixels that belong to the mask and have valid (non-zero) depth values
    valid_mask = (mask >= 0) & (depth > 0)
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    z = depth[valid_mask] / depth_shift

    # Back-project pixel coordinates to 3D space
    x = (x_coords - cx) * z / fx # + bx
    y = (y_coords - cy) * z / fy # + by
    points_3d = np.stack((x, y, z), axis=-1)  # Shape: (N, 3)
    
    return points_3d, mask[valid_mask], valid_mask

def transform_points(points_3d, pose):
    # Add homogeneous coordinate (1s) to the points
    N = points_3d.shape[0]
    points_3d_homogeneous = np.hstack((points_3d, np.ones((N, 1))))  # Shape: (N, 4)
    
    # Apply the transformation (pose matrix) to the 3D points
    transformed_points_3d = (pose @ points_3d_homogeneous.T).T  # Shape: (N, 4)

    # Drop the homogeneous coordinate and return (N, 3)
    return transformed_points_3d[:, :3]

def project_to_2d(points_3d, intrinsics):
    # Project 3D points back into 2D image space using the intrinsics
    points_2d = intrinsics @ points_3d.T  # Shape: (3, N)

    # Normalize by z to get (x, y) in image space
    points_2d[:2, :] /= points_2d[2, :]
    
    return points_2d[:2, :]  # Shape: (2, N)
