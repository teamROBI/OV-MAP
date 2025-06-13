import open3d as o3d
import numpy as np
import torch
from utils.util import *
import os
import glob
from os.path import join
import ovmap
import cv2
import pyrender
import trimesh
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from utils.depth_pc_util import *

def remove_ceiling(point_cloud, points, point_colors):
            # Identify points above the z_threshold to focus on detecting the ceiling
            high_points_indices = points[:, 2] > 0.5 #np.median(points[:, 2])
            print(np.mean(points[:, 2]), np.median(points[:, 2]))

            # Create a point cloud for the high points (above z_threshold) only for ceiling detection
            high_point_cloud = point_cloud.select_by_index(np.where(high_points_indices)[0])

            # Detect the dominant plane (likely the ceiling) using RANSAC on high points
            distance_threshold=0.1
            ransac_n=3
            num_iterations=1000
            plane_model, inliers = high_point_cloud.segment_plane(distance_threshold=distance_threshold,
                                                                ransac_n=ransac_n,
                                                                num_iterations=num_iterations)
            [a, b, c, d] = plane_model
            print(f"Detected plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

            # Get the z-values of the points identified as ceiling
            ceiling_points = np.asarray(high_point_cloud.points)[inliers]
            min_ceiling_z = np.min(ceiling_points[:, 2])
            
            # Filter out all points above the minimum z-value of the ceiling
            below_ceiling_indices = points[:, 2] < min_ceiling_z

            # Filter the point cloud to keep only the points below the ceiling
            filtered_points = points[below_ceiling_indices]
            filtered_colors = point_colors[below_ceiling_indices]

            # Create a new point cloud without the ceiling and points above it
            filtered_point_cloud = o3d.geometry.PointCloud()
            filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
            filtered_point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)

            # Voxelize the point cloud with a voxel size of 0.03
            # voxelized_cloud = point_cloud.voxel_down_sample(voxel_size=0.05)
            voxelized_cloud = filtered_point_cloud.voxel_down_sample(voxel_size=0.04)
            
            return voxelized_cloud

def visualize_all_ovmap(ovmap_output_path, data_pcd_path, data_type, rm_ceiling=False, color_floor=False):
    print(ovmap_output_path)
    output_path_list = sorted(glob.glob(os.path.join(ovmap_output_path, '*.pth')))
    print(output_path_list)
    
    for output_path in output_path_list:        
        if "raw" in os.path.basename(output_path):
            continue
        
        # if os.path.basename(output_path) not in ["scene_414.pth", "scene_2002_00.pth", "scene_2002_01.pth"]:
        #     continue
        if "scene0011" not in os.path.basename(output_path):
            continue
        
        print(f"[INFO] {os.path.basename(output_path)}")
        
        output_name = os.path.basename(output_path)
        output_3d_masks = torch.load(output_path)
        scene_path = os.path.join(data_pcd_path, output_name)
        try:
            scene = torch.load(scene_path)
        except:
            scene = o3d.io.read_point_cloud(scene_path.replace("pth", "ply"))
            # no_ceiling_scene = remove_ceiling(scene, np.asarray(scene.points), np.asarray(scene.colors))
            # o3d.visualization.draw_geometries([no_ceiling_scene])
            
        print("Total pixels:", len(output_3d_masks))
        print("Predicted instance segmentation len:", len(np.unique(output_3d_masks)))
        print("Negative prediction pixels:", len(np.where(output_3d_masks < 0)[0]))

        # OVMap result. Unique 3D mask and a color map
        unique_labels, counts = np.unique(output_3d_masks, return_counts=True)
        
        if data_type == "scannetv2":
            labels = scene['semantic_gt200']  # Semantic labels
            instance_ids = scene['instance_gt']  # Ground truth instance IDs

            # Generate random colors for each unique label in output_3d_masks
            colors = np.random.rand(len(unique_labels), 3)

            # Create a color map for each unique label in output_3d_masks
            color_map = {label: color for label, color in zip(unique_labels, colors)}

            # Assign colors based on output_3d_masks
            point_colors = np.array([
                color_map[output_3d_masks[i]] for i in range(len(output_3d_masks))
            ])

            # Define the gray color for labels -1, 0, and 2
            gray_color = (0.9, 0.9, 0.9)

            # Recolor parts where the GT semantic label is -1, 0, or 2 to gray
            point_colors = np.array([
                gray_color if labels[i] in (-1, 0, 2) else point_colors[i]
                for i in range(len(instance_ids))
            ])
        else:
            # Generate random colors for each unique label in output_3d_masks
            colors = np.random.rand(len(unique_labels), 3)

            # Create an array to map output_3d_masks to the corresponding color
            point_colors = np.array([
                colors[np.where(unique_labels == mask)[0][0]] for mask in output_3d_masks
            ])

        # # Find the label with the largest count
        # largest_label = unique_labels[np.argmax(counts)]

        # # Color the largest label with a specific color, e.g., [0.9, 0.9, 0.9]
        # # Find the index in unique_labels corresponding to the largest_label
        # largest_label_idx = np.where(unique_labels == largest_label)[0][0]
        # colors[largest_label_idx] = [0.9, 0.9, 0.9]  # Assign the gray color

        # Assign colors to each point in the point cloud using vectorized indexing

        # Create an Open3D point cloud object
        point_cloud = o3d.geometry.PointCloud()

        # Assign points based on dataset type
        if data_type == "scannetv2":
            point_cloud.points = o3d.utility.Vector3dVector(scene['coord'])
        elif data_type == "replica":
            point_cloud.points = o3d.utility.Vector3dVector(scene[0])
        elif data_type == "real_world":
            point_cloud.points = scene.points
        else:
            raise ValueError("[ERROR] Wrong dataset name")

        # Assign the corresponding colors to the point cloud
        point_cloud.colors = o3d.utility.Vector3dVector(point_colors)
        
        # Convert point cloud to numpy array for processing
        points = np.asarray(point_cloud.points)
        
        if rm_ceiling:
            voxelized_cloud = remove_ceiling(point_cloud, points, point_colors)
        else:
            voxelized_cloud = point_cloud.voxel_down_sample(voxel_size=0.04)
        
        if color_floor:
            # Detect the floor plane using RANSAC on the voxelized point cloud
            floor_model, floor_inliers = voxelized_cloud.segment_plane(distance_threshold=0.1,
                                                                    ransac_n=3,
                                                                    num_iterations=1000)
            [a, b, c, d] = floor_model
            print(f"Detected floor plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

            # Color the floor points with gray [0.9, 0.9, 0.9]
            voxelized_colors = np.asarray(voxelized_cloud.colors)
            voxelized_colors[floor_inliers] = [0.9, 0.9, 0.9]

            # Update the voxelized cloud colors with the new floor colors
            voxelized_cloud.colors = o3d.utility.Vector3dVector(voxelized_colors)

        # Visualize the voxelized point cloud with the floor colored gray
        o3d.visualization.draw_geometries([voxelized_cloud])

        # # Post process
        # if data_type == "scannetv2":
        #     pc_pred = np.concatenate((scene['coord'], output_3d_masks.reshape(-1, 1)), axis=1)
        # elif data_type == "replica":
        #     pc_pred = np.concatenate((scene[0], output_3d_masks.reshape(-1, 1)), axis=1)
        # labels = nearest_neighbor_smoothing(pc_pred[:, :3], pc_pred[:, 3], k=15)
        # print("Post processed label len:", len(np.unique(output_3d_masks)))

        # # Post processed result. Unique labels and a color map
        # # colors = assign_colormap(len(np.unique(labels))) / 255.0
        # colors = np.random.rand(len(unique_labels), 3)
        # print("Len colors:", len(colors))
        # point_colors = np.array([colors[label] for label in labels])
        # # point_cloud = o3d.geometry.PointCloud()
        # # point_cloud.points = o3d.utility.Vector3dVector(scene['coord'])
        # point_cloud.colors = o3d.utility.Vector3dVector(point_colors)
        # o3d.visualization.draw_geometries([point_cloud])


def visualize_all_ensemble(model):
    # ply_dir_path = f'output/scannetv2/save_bivi/ensemble/{model}'
    # ply_dir_path = f'output/scannetv2/save_bivi/save_pcd/{model}'
    # ply_dir_path = "/home/robi/PycharmProjects/SegmentAnything3D/output/scannetv2/MCl_OCso5_VX0.03_IT10/ensemble/sam_raw_depth"
    ply_dir_path = "/home/robi/PycharmProjects/SegmentAnything3D/output/scannetv2/CDIS_VX0.02_IT1_Q5_ITH0.9/ensemble/cropformer_sup_depth"
    
    ply_path_list = sorted(glob.glob(os.path.join(ply_dir_path, 'scene*.pth')))
    choosen_scene = ["scene0011_00", "scene0011_01", 'scene0084_02', 'scene0549_00', 'scene0552_00', 'scene0552_01', 'scene0616_00', 'scene0655_00', 'scene0655_01', "scene0655_02", 'scene0686_02']
    # "scene0011_01", 'scene0084_02', 'scene0549_00', 'scene0552_00', 'scene0552_01', 'scene0616_00', 'scene0655_00', 'scene0655_01', "scene0655_02", 'scene0686_02'

    for ply_path in ply_path_list:
        scene_name = os.path.basename(ply_path).split('.')[0]
        
        if scene_name not in choosen_scene:
            continue
        
        print(scene_name)
        def pcd_ensemble(org_path, new_path, pcd_path, vis_path):
            new_pcd = torch.load(new_path)
            new_pcd = num_to_natural(remove_small_group(new_pcd, 20))
            with open(org_path) as f:
                segments = json.load(f)
                org_pcd = np.array(segments['segIndices'])
            match_inds = [(i, i) for i in range(len(new_pcd))]
            print(len(match_inds))
            new_group = cal_group(dict(group=new_pcd), dict(group=org_pcd), match_inds)
            print(new_group.shape)
            data = torch.load(pcd_path)

            basename = os.path.splitext(os.path.basename(pcd_path))[0]
            visualize_partition(data["coord"], new_group, vis_path, basename)

        gt_ply_path = os.path.join("data/scannetv2/input/pointcept_process/val", scene_name + ".ply")
        # gt_ply_path = os.path.join("data/scannetv2/input/pointcept_process/train", scene_name + ".ply")
        gt_pc = o3d.io.read_point_cloud(gt_ply_path)
        print(len(gt_pc.points))
        # o3d.visualization.draw_geometries([gt_pc])
        gt_pth_path = os.path.join("data/scannetv2/input/pointcept_process/val", scene_name + ".pth")
        # gt_pth_path = os.path.join("data/scannetv2/input/pointcept_process/train", scene_name + ".pth")
        gt_pth = torch.load(gt_pth_path)
        print(len(gt_pth["coord"]))
        gt_pc.points = o3d.utility.Vector3dVector(gt_pth["coord"])
        labels = gt_pth['semantic_gt200']
        instance_ids = gt_pth['instance_gt']
        unique_ids = np.unique(instance_ids)
        colors = {id: color for id, color in zip(unique_ids, np.random.rand(len(unique_ids), 3))}
        
        nc = 0.9
        point_colors = np.array([colors[instance_ids[i]] if label not in (-1, 0, 2) else (nc, nc, nc) for i, label in enumerate(labels)])
        gt_pc.colors = o3d.utility.Vector3dVector(point_colors)
        # o3d.visualization.draw_geometries([gt_pc])

        # pred_ids = torch.load("output/scannetv2/save_bivi/save_pcd/cropformer_sup_depth/scene0011_01.pth")
        # print(len(pred_ids))
        # unique_ids = np.unique(pred_ids)
        # colors = {id: color for id, color in zip(unique_ids, np.random.rand(len(unique_ids), 3))}
        # point_colors = np.array([colors[id] if np.average(point_colors[i]) != nc else (nc, nc, nc) for i, id in enumerate(pred_ids)])
        # gt_pc.colors = o3d.utility.Vector3dVector(point_colors)
        # o3d.visualization.draw_geometries([gt_pc])

        pred_ids = torch.load(ply_path)
        print(len(pred_ids))
        unique_ids = np.unique(pred_ids)
        colors = {id: color for id, color in zip(unique_ids, np.random.rand(len(unique_ids), 3))}
        point_colors = np.array([colors[id] if np.average(point_colors[i]) != nc else (nc, nc, nc) for i, id in enumerate(pred_ids)])
        gt_pc.colors = o3d.utility.Vector3dVector(point_colors)
        o3d.visualization.draw_geometries([gt_pc])


def project_point_cloud_to_images(point_cloud, pose, intrinsic, image_shape):
    # Invert the pose to transform points from world to camera coordinates
    pose_inv = np.linalg.inv(pose)

    # Extract the coordinates and colors from the point cloud
    coords_world = point_cloud['coord']  # Shape: (N, 3)
    colors = point_cloud['color']  # Shape: (N, 3)
    normals = point_cloud['normal'] # Shape: (N, 3)

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

    # Initialize the depth and RGB images
    depth_image = np.zeros(image_shape, dtype=np.float64)
    rgb_image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)

    # Fill in the depth and RGB images
    for i in range(len(u)):
        x, y = int(round(u[i])), int(round(v[i]))
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            depth_image[y, x] = depth_values[i]
            rgb_image[y, x] = colors[i]


    print(depth_image.min(), depth_image.max())
    np.savetxt("output/scannetv2/save/point_depth.csv", depth_image)

    depth_img = cv2.imread("data/scannetv2/input/scannetv2_images/val/scene0011_00/depth/0.png", -1)
    print(depth_img.min(), depth_img.max())

    pcd = backproject_rgbd_to_pointcloud(rgb_image, depth_image, intrinsic)
    rgb_image, depth_image = render_point_cloud_to_image(pcd, pose_inv, intrinsic, image_shape)

    return rgb_image, depth_image


def backproject_rgbd_to_pointcloud(rgb_image, depth_image, intrinsic):
    depth_image_uint16 = (depth_image * 1000).astype(np.uint16)

    # Create Open3D RGBDImage from the RGB and depth images
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb_image),
        o3d.geometry.Image(depth_image_uint16),
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

    # Translate the point cloud so that it is centered at the origin
    # This step is optional and depends on how you want to visualize the point cloud
    # point_cloud.transform([[1, 0, 0, 0],
    #                        [0, 1, 0, 0],
    #                        [0, 0, 1, 0],
    #                        [0, 0, 0, 1]])

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])
    return point_cloud


def render_point_cloud_to_image(pcd, camera_pose, camera_intrinsics, image_shape):
    # Make pcd to mesh for pyrender
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)[0]
    o3d.visualization.draw_geometries([poisson_mesh])
    pyrender_mesh = convert_open3d_mesh_to_pyrender_mesh(poisson_mesh)

    # Create a Pyrender scene
    scene = pyrender.Scene()
    scene.add(pyrender_mesh)

    # Set up the camera
    camera = pyrender.IntrinsicsCamera(fx=camera_intrinsics[0, 0], fy=camera_intrinsics[1, 1],
                                       cx=camera_intrinsics[0, 2], cy=camera_intrinsics[1, 2])
    scene.add(camera, pose=camera_pose)

    # Set up the offscreen renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=image_shape[1], viewport_height=image_shape[0])

    # Render the scene
    rgb_image, depth_image = renderer.render(scene)

    return rgb_image, depth_image


def convert_open3d_mesh_to_pyrender_mesh(open3d_mesh):
    # Extract vertices and faces from Open3D mesh
    vertices = np.asarray(open3d_mesh.vertices)
    faces = np.asarray(open3d_mesh.triangles)

    # Create a Trimesh object from vertices and faces
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    pyrender_mesh = pyrender.Mesh.from_trimesh(trimesh_mesh)

    return pyrender_mesh


if __name__ == '__main__':
    print('======== SELECT ========')
    print('1. Visualize one pointcloud file\n'
          '2. Visualize OVMap Result(No Ensemble)\n'
          '3. Visualize OVMap Result(Ensemble)\n'
          '4. Ensemble Result\n'
          '5. Point cloud to 2d image')
    select = int(input(">>> "))
    
    data_type = "scannetv2" # scannetv2, replica, real_world
    mask_model = "cropformer"
    depth_type = "raw_depth"
    
    OVERLAP_CRITERION="lo3" #large_overlap(lo3), small overlap(so5)
    VOXEL_SIZE=0.03
    IMAGE_ITER=10
    HYPER_PARAMETERS=f"OVM_OC{OVERLAP_CRITERION}_VX{VOXEL_SIZE}_IT{IMAGE_ITER}"
    print(HYPER_PARAMETERS)
    
    ovmap_output_path = f'output/{data_type}/{HYPER_PARAMETERS}/save_pcd/{mask_model}_{depth_type}' #large_overlap_0.3, small_overlap_0.5, save_bivi, save
    
    # ovmap_output_path = f"/home/robi/PycharmProjects/SegmentAnything3D/output/real_world/OVM_MCh_OClo3_VX0.03_IT5/save_pcd/cropformer_sup_depth"
    # ovmap_output_path = f"/home/robi/PycharmProjects/SegmentAnything3D/output/real_world/OVM_MCh_OCiou4_VX0.03_IT1/save_pcd/cropformer_sup_depth"
    # ovmap_output_path = f"/home/robi/PycharmProjects/SegmentAnything3D/output/real_world/DEBUG_CDIS_VX0.02_IT1_Q5_ITH0.9/save_pcd/cropformer_sup_depth"
    # ovmap_output_path = "/home/robi/PycharmProjects/SegmentAnything3D/output/real_world/CDIS_MCh_OClo3_VX0.03_IT5/save_pcd/cropformer_sup_depth"
    
    data_pcd_path = f'data/{data_type}/input/pointcept_process/val'
    # data_pcd_path = f'data/{data_type}/input/replica_processed/replica_3d'
    # data_pcd_path = f'data/{data_type}/input/real_world_ply'

    if select == 1:
        # Load the point cloud from a PLY file
        print('>> Enter the pointcloud file path (extension: .ply)')
        ply_file_path = "/home/robi/PycharmProjects/SegmentAnything3D/data/real_world/input/real_world/real_world_ply/scene_414.ply"
        pc = o3d.io.read_point_cloud(ply_file_path)

        # Use RANSAC to segment the floor (plane model)
        plane_model, inliers = pc.segment_plane(distance_threshold=0.01,
                                                ransac_n=3,
                                                num_iterations=1000)

        # Extract the floor points based on inlier indices
        floor_pc = pc.select_by_index(inliers)

        # Get the points from the point cloud
        points = np.asarray(pc.points)

        # Filter non-floor points (i.e., points not in inliers)
        non_inliers = list(set(range(len(points))) - set(inliers))

        # Assign random colors to the rest of the points based on predicted instance IDs
        pth_dir_path = "output/real_world/save_real/save_pcd/cropformer_sup_depth/scene_414.pth"
        pred_ids = torch.load(pth_dir_path)

        # Ensure that the length of pred_ids matches non-inliers
        non_floor_pred_ids = pred_ids[non_inliers]
        unique_ids = np.unique(non_floor_pred_ids)
        colors = {id: color for id, color in zip(unique_ids, np.random.rand(len(unique_ids), 3))}
        non_floor_colors = np.array([colors[id] for i, id in enumerate(non_floor_pred_ids)])

        # Create a color array and assign colors to non-floor points first
        final_colors = np.zeros((len(points), 3))
        final_colors[non_inliers] = non_floor_colors  # Assign colors to non-floor points

        # Color the floor points with gray (RGB: 0.9, 0.9, 0.9) after non-floor points
        floor_colors = np.full((len(inliers), 3), [0.9, 0.9, 0.9])
        final_colors[inliers] = floor_colors  # Assign gray to floor points

        # Apply the new colors to the point cloud
        pc.colors = o3d.utility.Vector3dVector(final_colors)

        # Voxelize the point cloud (0.03 voxel size)
        voxel_size = 0.03
        pc = pc.voxel_down_sample(voxel_size=voxel_size)

        # Visualize the colored point cloud with the floor highlighted
        o3d.visualization.draw_geometries([pc])

        # Print features of the point cloud
        print("Number of points:", len(pc.points))
        if pc.colors:
            print("Contains color information")
        else:
            print("Does not contain color information")

        if pc.normals:
            print("Contains normals")
        else:
            print("Does not contain normals")

        print(pc.colors)
    elif select == 2:
        visualize_all_ovmap(ovmap_output_path, data_pcd_path, data_type)
    elif select == 3:
        visualize_all_ensemble(ovmap_output_path)
    elif select == 4:
        org_path = 'scene0011_00_vh_clean_2.0.020000.segs.json'
        new_path = f'output/scannetv2/save/save_pcd/{model}/scene0011_00.pth'
        data_path = 'data/scannetv2/input/pointcept_process/val/scene0011_00.pth'
        vis_path = f'output/scannetv2/save/ensemble/{model}'
        ovmap.pcd_ensemble(org_path, new_path, data_path, vis_path)

        ensemble_path = join(vis_path, 'scene0011_00.ply')
        pc = o3d.io.read_point_cloud(ensemble_path)
        o3d.visualization.draw_geometries([pc])
    else:
        # Adjust the image dimensions if necessary based on expected output size
        image_shape = (480, 640)

        # Project the point cloud to 2D image planes
        data_path = "data/scannetv2/input/scannetv2_images/val/scene0011_00"
        intrinsic_path = join(data_path, 'intrinsics', 'intrinsic_depth.txt')
        depth_intrinsic = np.loadtxt(intrinsic_path)
        pose = join(data_path, 'pose', '0' + '.txt')
        pose_matrix = np.loadtxt(pose)

        ply_file_path = "data/scannetv2/input/pointcept_process/val/scene0011_00.ply"
        pcd = o3d.io.read_point_cloud(ply_file_path)
        # points = np.asarray(pcd.points)
        ply_file_path = "data/scannetv2/input/pointcept_process/val/scene0011_00.pth"
        pth = torch.load(ply_file_path)

        # Create a coordinate frame to represent the camera pose
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)  # Size adjusted for visibility
        # Apply the transformation to the camera frame
        camera_frame.transform(pose_matrix)
        # Visualize the mesh and the transformed camera frame
        o3d.visualization.draw_geometries([pcd, camera_frame], window_name="Pose Visualization with Mesh")

        rgb_image, depth_img = project_point_cloud_to_images(pth, pose_matrix, depth_intrinsic, image_shape)
        depth_image_visual = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        cv2.imwrite('output/scannetv2/save/rgb_image.png', rgb_image)
        cv2.imwrite('output/scannetv2/save/depth_image_visual.png', depth_image_visual)


