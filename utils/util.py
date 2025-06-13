import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import open3d as o3d
from scipy.spatial import KDTree
import os, glob
import copy, random
from PIL import Image
import cv2
import json
from os.path import join
# import clip
import colorsys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from numba import jit
from scipy.spatial.distance import cdist
# import open_clip
import imageio


SCANNET_COLOR_MAP_20 = {-1: (0., 0., 0.), 0: (174., 199., 232.), 1: (152., 223., 138.), 2: (31., 119., 180.), 3: (255., 187., 120.), 4: (188., 189., 34.), 5: (140., 86., 75.),
                        6: (255., 152., 150.), 7: (214., 39., 40.), 8: (197., 176., 213.), 9: (148., 103., 189.), 10: (196., 156., 148.), 11: (23., 190., 207.), 12: (247., 182., 210.), 
                        13: (219., 219., 141.), 14: (255., 127., 14.), 15: (158., 218., 229.), 16: (44., 160., 44.), 17: (112., 128., 144.), 18: (227., 119., 194.), 19: (82., 84., 163.)}
# ADE20K_DF = pd.read_csv('output/scannetv2/save2/mask_test/ADE20K.tsv', delimiter='\t', index_col='Idx')
# ADE20K_COLOR_MAP_150 = {row.Name: tuple(map(int, row['Color_Code (R,G,B)'].strip('()').split(','))) for idx, row in ADE20K_DF.iterrows()}
# COLOR_MAP_150 = {idx: tuple(map(int, row['Color_Code (R,G,B)'].strip('()').split(','))) for idx, row in ADE20K_DF.iterrows()}


class Voxelize(object):
    def __init__(self,
                 voxel_size=0.02,
                 hash_type="fnv",
                 mode='train',
                 keys=("coord", "normal", "color", "label"),
                 return_discrete_coord=False,
                 return_min_coord=False):
        self.voxel_size = voxel_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.keys = keys
        self.return_discrete_coord = return_discrete_coord
        self.return_min_coord = return_min_coord

    def __call__(self, data_dict, custom_voxel_size=None):
        # print(">>> Start Voxelize")
        assert "coord" in data_dict.keys()

        if custom_voxel_size:
            print(">>> Use custom voxel size")
            discrete_coord = np.floor(data_dict["coord"] / np.array(custom_voxel_size)).astype(int)
            min_coord = discrete_coord.min(0) * np.array(custom_voxel_size)
        else:
            discrete_coord = np.floor(data_dict["coord"] / np.array(self.voxel_size)).astype(int)
            min_coord = discrete_coord.min(0) * np.array(self.voxel_size)

        discrete_coord -= discrete_coord.min(0)
        key = self.hash(discrete_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == 'train':  # train mode
            # idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1])
            idx_unique = idx_sort[idx_select]
            if self.return_discrete_coord:
                data_dict["discrete_coord"] = discrete_coord[idx_unique]
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            for key in self.keys:
                data_dict[key] = data_dict[key][idx_unique]
            return data_dict

        elif self.mode == 'test':  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = dict(index=idx_part)
                for key in data_dict.keys():
                    if key in self.keys:
                        data_part[key] = data_dict[key][idx_part]
                    else:
                        data_part[key] = data_dict[key]
                if self.return_discrete_coord:
                    data_part["discrete_coord"] = discrete_coord[idx_part]
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr


def overlap_percentage(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    area_intersection = np.sum(intersection)

    area_mask1 = np.sum(mask1)
    area_mask2 = np.sum(mask2)

    smaller_area = min(area_mask1, area_mask2)

    return area_intersection / smaller_area


def remove_samll_masks(masks, ratio=0.8):
    filtered_masks = []
    skip_masks = set()

    for i, mask1_dict in enumerate(masks):
        if i in skip_masks:
            continue

        should_keep = True
        for j, mask2_dict in enumerate(masks):
            if i == j or j in skip_masks:
                continue
            mask1 = mask1_dict["segmentation"]
            mask2 = mask2_dict["segmentation"]
            overlap = overlap_percentage(mask1, mask2)
            if overlap > ratio:
                if np.sum(mask1) < np.sum(mask2):
                    should_keep = False
                    break
                else:
                    skip_masks.add(j)

        if should_keep:
            filtered_masks.append(mask1)

    return filtered_masks


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().cpu().numpy()
    assert isinstance(x, np.ndarray)
    return x


def save_point_cloud(coord, color=None, file_path="pc.ply", logger=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    coord = to_numpy(coord)
    if color is not None:
        color = to_numpy(color)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(np.ones_like(coord) if color is None else color)
    o3d.io.write_point_cloud(file_path, pcd)
    if logger is not None:
        logger.info(f"Save Point Cloud to: {file_path}")


def remove_small_group(group_ids, th):
    unique_elements, counts = np.unique(group_ids, return_counts=True)
    result = group_ids.copy()
    for i, count in enumerate(counts):
        if count < th:
            result[group_ids == unique_elements[i]] = -1
    
    return result


def pairwise_indices(length):
    return [[i, i + 1] if i + 1 < length else [i] for i in range(0, length, 2)]

def pairwise_frames(frame_list):
    return [(frame_list[i], frame_list[i + 1]) if i + 1 < len(frame_list) else (frame_list[i], None) for i in range(0, len(frame_list), 2)]


def num_to_natural(group_ids):
    '''
    Change the group number to natural number arrangement
    '''
    if np.all(group_ids == -1):
        return group_ids
    array = copy.deepcopy(group_ids)
    unique_values = np.unique(array[array != -1])
    mapping = np.full(np.max(unique_values) + 2, -1)
    mapping[unique_values + 1] = np.arange(len(unique_values))
    array = mapping[array + 1]
    return array


def get_matching_indices(source, pcd_tree, search_voxel_size, K=None):
    match_inds = []
    for i, point in enumerate(source.points):
        try:
            [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        except:
            print("[WARNING] No near points searched.")
            print(point)
            continue
            
            # # Start with the default radius and increment if needed
            # radius = 5
            # while True:
            #     print(f"[WARNING] Current search radius: {radius}")
            #     try:
            #         [_, idx, _] = pcd_tree.search_radius_vector_3d(point, radius)
            #         break  # Exit the loop if successful
            #     except:
            #         print(f"[INFO] Increasing search radius to {radius}")
            #         radius += 5  # Increment the search radius value
            
        if K is not None:
            idx = idx[:K]
        for j in idx:
            # match_inds[i, j] = 1
            match_inds.append((i, j))
    return match_inds


def visualize_3d(data_dict, text_feat_path, save_path):
    text_feat = torch.load(text_feat_path)
    group_logits = np.einsum('nc,mc->nm', data_dict["group_feat"], text_feat)
    group_labels = np.argmax(group_logits, axis=-1)
    labels = group_labels[data_dict["group"]]
    labels[data_dict["group"] == -1] = -1
    visualize_pcd(data_dict["coord"], data_dict["color"], labels, save_path)


def visualize_pcd(coord, pcd_color, labels, save_path):
    # alpha = 0.5
    label_color = np.array([SCANNET_COLOR_MAP_20[label] for label in labels])
    # overlay = (pcd_color * (1-alpha) + label_color * alpha).astype(np.uint8) / 255
    label_color = label_color / 255
    save_point_cloud(coord, label_color, save_path)


def visualize_2d(img_color, labels, img_size, save_path):
    import matplotlib.pyplot as plt
    # from skimage.segmentation import mark_boundaries
    # from skimage.color import label2rgb
    label_names = ["wall", "floor", "cabinet", "bed", "chair",
           "sofa", "table", "door", "window", "bookshelf",
           "picture", "counter", "desk", "curtain", "refridgerator",
           "shower curtain", "toilet", "sink", "bathtub", "other"]
    colors = np.array(list(SCANNET_COLOR_MAP_20.values()))[1:]
    segmentation_color = np.zeros((img_size[0], img_size[1], 3))
    for i, color in enumerate(colors):
        segmentation_color[labels == i] = color
    alpha = 1
    overlay = (img_color * (1-alpha) + segmentation_color * alpha).astype(np.uint8)
    fig, ax = plt.subplots()
    ax.imshow(overlay)
    patches = [plt.plot([], [], 's', color=np.array(color)/255, label=label)[0] for label, color in zip(label_names, colors)]
    plt.legend(handles=patches, bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=4, fontsize='small')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def visualize_partition(coord, group_id, save_path, basename):
    group_id = group_id.reshape(-1)
    num_groups = group_id.max() + 1
    group_colors = np.random.rand(num_groups, 3)
    group_colors = np.vstack((group_colors, np.array([0,0,0])))
    color = group_colors[group_id]
    save_point_cloud(coord, color, join(save_path, basename+".ply"))
    torch.save(group_id, join(save_path, basename+".pth"))


def delete_invalid_group(group, group_feat):
    indices = np.unique(group[group != -1])
    group = num_to_natural(group)
    group_feat = group_feat[indices]
    return group, group_feat


def generate_distinguishable_colors(n, colormaps=['tab20', 'Set3', 'Accent', 'Dark2']):
    # Initialize an empty list to store colors
    colors = []

    # Determine how many colors are needed from each colormap
    num_colormaps = len(colormaps)
    colors_per_cmap = np.ceil(n / num_colormaps).astype(int)

    # Generate colors from each colormap
    for cmap_name in colormaps:
        cmap = plt.get_cmap(cmap_name)
        # Extract colors and append to the list
        colors.extend(cmap(np.linspace(0, 1, colors_per_cmap))[:, :3])
    print(len(colors))
    # If more colors are needed, adjust brightness and saturation
    while len(colors) < n:
        # Take an existing color and adjust its brightness and saturation
        for base_color in colors[:n]:
            hsv_color = mcolors.rgb_to_hsv(base_color[:3])
            # Adjust brightness and saturation
            hsv_mod = (hsv_color[0], max(hsv_color[1] - 0.2, 0), min(hsv_color[2] + 0.2, 1))
            mod_color = mcolors.hsv_to_rgb(hsv_mod)
            colors.append(mod_color)
            if len(colors) >= n:
                break

    return np.array(colors[:n])


def generate_unique_colors(n):
    colors = []
    for i in range(n):
        # Vary hue, keep saturation and lightness constant
        hue = i / n
        saturation = 0.7  # Adjust saturation (0 to 1)
        lightness = 0.5  # Adjust lightness (0 to 1)

        # Convert HSL to RGB
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(rgb)

    return np.array(colors)


def assign_colormap(n):
    # Calculate how many additional colors are needed
    additional_colors_needed = max(0, n - len(COLOR_MAP_150))

    # If additional colors are needed, generate them randomly
    if additional_colors_needed > 0:
        # Generate random RGB colors
        additional_colors = np.random.randint(0, 256, size=(additional_colors_needed, 3))
    else:
        additional_colors = np.array([])

    # Combine the predefined colormap with the additional colors
    extended_colors = list(COLOR_MAP_150.values()) + additional_colors.tolist()

    # Ensure the result is in the desired format, e.g., list of tuples
    extended_colors = [tuple(color) for color in extended_colors]

    return np.array(extended_colors)


def save_sam_result(image, anns, filename):
    vis_filename = filename.replace("2d_seg", "2d_seg_vis")
    
    if not os.path.exists(os.path.dirname(vis_filename)):
        os.makedirs(os.path.dirname(vis_filename))
    
    if len(anns) == 0:
        return

    plt.figure(figsize=(20, 20))
    plt.imshow(image)

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0  # Set transparency to 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])  # Random color with alpha value 0.35
        img[m] = color_mask
    ax.imshow(img)
    plt.axis('off')
    plt.savefig(vis_filename, bbox_inches='tight', pad_inches=0, transparent=True)  # Save the figure
    plt.close()


def save_oneformer_result(image, masks, masks_info, filename):
    if len(masks_info) == 0:
        return

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    # Assuming 'segmentation' is a numpy array with label_id for each pixel
    masks = masks.numpy()  # If it's a tensor, convert to numpy

    # Create an RGB image based on the segmentation mask
    segmentation_rgb = np.ones((*masks.shape, 4))
    segmentation_rgb[:, :, 3] = 0  # Set transparency to 0

    for info in masks_info:
        mask = masks == info['id']
        if info['label_id'] == 0:
            label_name = 'others'
        else:
            label_name = ADE20K_DF.loc[info['label_id']]['Name']
        color = np.array(ADE20K_COLOR_MAP_150.get(label_name, (0, 0, 0))) / 255  # Default to black if label_id not found
        segmentation_rgb[mask] = np.concatenate([color, [0.5]])

    # Visualize the result
    ax.imshow(segmentation_rgb)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True)  # Save the figure
    plt.close()


def save_cropformer_result(image, sorted_anns, filename):
    vis_filename = filename.replace("2d_seg", "2d_seg_vis")
    if vis_filename[-3:] != "jpg":
        vis_filename = vis_filename.replace(vis_filename[-3:], "jpg")
    
    if not os.path.exists(os.path.dirname(vis_filename)):
        os.makedirs(os.path.dirname(vis_filename))

    if len(sorted_anns) == 0:
        return

    plt.figure(figsize=(20, 20))
    plt.imshow(image)

    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0].shape[0], sorted_anns[0].shape[1], 4))
    img[:,:,3] = 0  # Set transparency to 0
    for ann in sorted_anns:
        m = ann
        color_mask = np.concatenate([np.random.random(3), [0.5]])  # Random color with alpha value 0.35
        img[m.astype(bool)] = color_mask
    ax.imshow(img)
    plt.axis('off')
    plt.savefig(vis_filename, bbox_inches='tight', pad_inches=0, transparent=True)  # Save the figure
    plt.close()

    return img


def visualize_segmentation(rgb_img, seg_img, alpha=0.5):
    """
    Visualize the segmentation results on top of the RGB image with random colors.
    
    If seg_img is a binary mask, overlay the True parts with a green color.
    
    :param rgb_img: RGB image (HxWx3)
    :param seg_img: Segmentation image (HxW) with instance IDs or binary mask
    :param alpha: Transparency factor for overlay (default is 0.5)
    :return: Visualized image with segmentation overlay
    """
    
    # Check if seg_img is a binary mask (either [False, True] or [0, 1])
    is_binary_mask = np.array_equal(np.unique(seg_img), [False, True]) or np.array_equal(np.unique(seg_img), [0, 1])
    
    # Prepare an empty RGBA image for overlay
    img_overlay = np.ones((*seg_img.shape, 4))
    img_overlay[:, :, 3] = 0  # Set transparency to 0

    if is_binary_mask:
        # Binary mask: Apply green color to the True parts
        color_mask = [0, 1, 0, alpha]  # Green color (R=0, G=1, B=0) with alpha transparency
        img_overlay[seg_img == True] = color_mask  # Apply to True values
    else:
        # Not a binary mask: Get unique segmentation IDs
        unique_ids = np.unique(seg_img[seg_img > -1])
        
        # Assign a random color for each unique instance ID
        for seg_id in unique_ids:
            random_color_mask = np.concatenate([np.random.random(3), [alpha]])  # Random color with alpha
            img_overlay[seg_img == seg_id] = random_color_mask

    # Plot the RGB image and overlay the segmentation with transparency
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_img)

    ax = plt.gca()
    ax.set_autoscale_on(False)  # Turn off autoscale to prevent overlay from scaling
    ax.imshow(img_overlay, interpolation='none')  # Overlay the segmentation mask

    plt.axis('off')  # Hide the axis
    plt.show()

    return img_overlay


def visualize_segmentation_subplots(rgb_img, seg_dict, intensity_dict, alpha=0.5, max_rows=5, image_size=5):
    """
    Visualize multiple segmentation binary masks on top of the RGB image with subplots, sorted by intensity.
    
    :param rgb_img: RGB image (HxWx3)
    :param seg_dict: Dictionary containing {id: binary_mask (HxW)} pairs
    :param intensity_dict: Dictionary containing {id: intensity} pairs (how often each id was assigned)
    :param alpha: Transparency factor for overlay (default is 0.5)
    :param max_rows: Maximum number of rows for the subplot grid
    :param image_size: Size of each subplot (default is 5x5)
    :return: Visualized image with segmentation overlays in subplots
    """
    
    num_masks = len(seg_dict)

    # Sort seg_dict by intensity values from intensity_dict (descending order)
    sorted_seg_items = sorted(seg_dict.items(), key=lambda item: intensity_dict.get(item[0], 0), reverse=True)

    # Define the number of rows and columns based on the number of masks
    rows = min(max_rows, num_masks)  # Limit rows to `max_rows` or less
    cols = (num_masks + rows - 1) // rows  # Calculate the required number of columns

    # Create subplots with a grid layout (rows x cols)
    fig, axs = plt.subplots(rows, cols, figsize=(image_size * cols, image_size * rows))

    # Flatten the axs to easily iterate over it, as we may have multiple rows and columns
    axs = axs.flatten()

    # Loop over the sorted items (by intensity) to visualize each mask
    for idx, (seg_id, binary_mask) in enumerate(sorted_seg_items):
        # Check if the mask is binary (just to ensure it's valid)
        is_binary_mask = np.array_equal(np.unique(binary_mask), [False, True]) or np.array_equal(np.unique(binary_mask), [0, 1])
        
        # Count the number of 'True' pixels in the binary mask
        true_pixel_count = np.sum(binary_mask)  # Number of True pixels
        
        # Prepare an empty RGBA image for overlay
        img_overlay = np.ones((*binary_mask.shape, 4))
        img_overlay[:, :, 3] = 0  # Set transparency to 0

        if is_binary_mask:
            # Apply green color to the True parts of the binary mask
            color_mask = [0, 1, 0, alpha]  # Green color (R=0, G=1, B=0) with alpha transparency
            img_overlay[binary_mask == True] = color_mask
        else:
            # Just in case there's a non-binary mask, handle it with random colors
            random_color_mask = np.concatenate([np.random.random(3), [alpha]])  # Random color with alpha
            img_overlay[binary_mask == True] = random_color_mask
        
        # Plot the original RGB image in the background
        axs[idx].imshow(rgb_img)
        
        # Overlay the segmentation mask with transparency
        axs[idx].imshow(img_overlay, interpolation='none')

        # Get intensity from intensity_dict for the current ID
        intensity = intensity_dict.get(seg_id, 0)

        # Set the title with seg_id, the count of True pixels, and the intensity
        axs[idx].set_title(f"IT: {intensity}, ID: {seg_id}, PX: {true_pixel_count}")
        axs[idx].axis('off')

    # Turn off any remaining empty subplots (if there are more grid spaces than masks)
    for i in range(idx + 1, len(axs)):
        axs[i].axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


def visualize_3d_instances(pcd_list, intensity_dict):
    # Load the scene point cloud
    scene_path = "/home/robi/PycharmProjects/SegmentAnything3D/data/scannetv2/input/pointcept_process/val/scene0011_00.ply"
    scene_pcd = o3d.io.read_point_cloud(scene_path)
    
    coords_world, groups_world = [], []
    for pcd_dict in tqdm(pcd_list):
        coords_world.extend(pcd_dict["coord"])
        groups_world.extend(pcd_dict["group"])
    coords_world, groups_world = np.array(coords_world), np.array(groups_world)

    print(f"Coords len: {len(coords_world)}")

    unique_group_ids = np.unique(groups_world)
    print(f"Unique group ids: {unique_group_ids}")

    # Sort the unique group IDs by their intensity in descending order
    sorted_group_ids_by_intensity = sorted(unique_group_ids, key=lambda x: intensity_dict.get(x, 0), reverse=True)

    # Visualize each instance one by one in green color, with larger voxel size for instance
    for group_id in sorted_group_ids_by_intensity:
        print(f"Visualize id: {group_id}")
        # Create the instance point cloud for the current group
        group_mask = (groups_world == group_id)
        instance_coords = coords_world[group_mask]

        # Create a point cloud for the instance
        instance_pcd = o3d.geometry.PointCloud()
        instance_pcd.points = o3d.utility.Vector3dVector(instance_coords)

        # Color the instance point cloud green
        instance_colors = np.tile([0, 1, 0], (instance_coords.shape[0], 1))  # Green color
        instance_pcd.colors = o3d.utility.Vector3dVector(instance_colors)

        # Visualize the scene point cloud and the instance point cloud
        print(f"Instance ID: {group_id}, Intensity: {intensity_dict.get(group_id, 0)}, with {len(instance_coords)} points")
        o3d.visualization.draw_geometries([scene_pcd, instance_pcd], window_name=f"Instance ID: {group_id}, Intensity: {intensity_dict.get(group_id, 0)}, with {len(instance_coords)} points")


def calculate_blur_score(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    print(laplacian_var)
    return laplacian_var


def calculate_cluster_overlap(cluster_0, cluster_1, voxel_size):
    # Calculate all pairwise distances between points in the two clouds
    if len(cluster_0) < len(cluster_1):
        distances = cdist(cluster_0, cluster_1, metric='euclidean')
    else:
        distances = cdist(cluster_1, cluster_0, metric='euclidean')

    threshold_distance = voxel_size * 2

    overlapping_mask = distances < threshold_distance
    overlapping_indices_cluster = np.any(overlapping_mask, axis=1)
    overlapping_indices = np.where(overlapping_indices_cluster)[0]

    # # Find the minimum distance to any point in points2 for each point in points1
    # min_distances = np.min(distances, axis=1)
    # overlapping_indices = np.where(min_distances < threshold_distance)[0]
    
    # Calculate the overlap ratio as the proportion of points1 that are close to points2
    # small_overlap_ratio = len(overlapping_indices) / min(len(cluster_0), len(cluster_1))
    large_overlap_ratio = len(overlapping_indices) / max(len(cluster_0), len(cluster_1))

    return large_overlap_ratio, overlapping_indices


@jit(nopython=True)
def calculate_cluster_overlap_numba(cluster_0, cluster_1, voxel_size):
    threshold_distance = voxel_size * 2
    distances = cdist(cluster_0, cluster_1)
    overlapping = distances < threshold_distance
    overlapping_indices = np.where(overlapping.any(axis=1))[0]
    large_overlap_ratio = len(overlapping_indices) / max(len(cluster_0), len(cluster_1))
    return large_overlap_ratio, overlapping_indices


def find_duplicate_coordinates(coords):
    # Convert the (N, 3) array to a structured array to make each row hashable
    structured_coords = np.core.records.fromarrays(coords.transpose(), 
                                                   names='x, y, z', 
                                                   formats='f8, f8, f8')
    
    # Use numpy unique function to find unique rows and their indices
    _, inverse_indices, counts = np.unique(structured_coords, return_inverse=True, return_counts=True)
    
    # Find indices of the first occurrences of the duplicates
    duplicate_first_indices = np.where(counts[inverse_indices] > 1)[0]
    
    # Optional: Find the actual duplicate coordinates (if needed)
    duplicate_coords = coords[duplicate_first_indices]

    print(">>> Duplicates:", len(duplicate_first_indices))

    return duplicate_first_indices


class ClipCapture(object):
    def __init__(self,
                 model_name='ViT-bigG-14',
                 pretrained='laion2b_s39b_b160k'):
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.object_texts = ['wall', 'chair', 'floor', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk', 'office chair', 'bed', 'pillow', 'sink', 'picture', 'window', 'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair', 'coffee table', 'box',
                            'refrigerator', 'lamp', 'kitchen cabinet', 'towel', 'clothes', 'tv', 'nightstand', 'counter', 'dresser', 'stool', 'cushion', 'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard', 'bag', 'backpack', 'toilet paper',
                            'printer', 'tv stand', 'whiteboard', 'blanket', 'shower curtain', 'trash can', 'closet', 'stairs', 'microwave', 'stove', 'shoe', 'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board', 'washing machine', 'mirror', 'copier',
                            'basket', 'sofa chair', 'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person', 'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard', 'piano', 'suitcase', 'rail', 'radiator', 'recycling bin', 'container',
                            'wardrobe', 'soap dispenser', 'telephone', 'bucket', 'clock', 'stand', 'light', 'laundry basket', 'pipe', 'clothes dryer', 'guitar', 'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle', 'ladder', 'bathroom stall', 'shower wall',
                            'cup', 'jacket', 'storage bin', 'coffee maker', 'dishwasher', 'paper towel roll', 'machine', 'mat', 'windowsill', 'bar', 'toaster', 'bulletin board', 'ironing board', 'fireplace', 'soap dish', 'kitchen counter', 'doorframe',
                            'toilet paper dispenser', 'mini fridge', 'fire extinguisher', 'ball', 'hat', 'shower curtain rod', 'water cooler', 'paper cutter', 'tray', 'shower door', 'pillar', 'ledge', 'toaster oven', 'mouse', 'toilet seat cover dispenser',
                            'furniture', 'cart', 'storage container', 'scale', 'tissue box', 'light switch', 'crate', 'power outlet', 'decoration', 'sign', 'projector', 'closet door', 'vacuum cleaner', 'candle', 'plunger', 'stuffed animal', 'headphones', 'dish rack',
                            'broom', 'guitar case', 'range hood', 'dustpan', 'hair dryer', 'water bottle', 'handicap bar', 'purse', 'vent', 'shower floor', 'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock', 'music stand', 'projector screen', 'divider',
                            'laundry detergent', 'bathroom counter', 'object', 'bathroom vanity', 'closet wall', 'laundry hamper', 'bathroom stall door', 'ceiling light', 'trash bin', 'dumbbell', 'stair rail', 'tube', 'bathroom cabinet', 'cd case', 'closet rod',
                            'coffee kettle', 'structure', 'shower head', 'keyboard piano', 'case of water bottles', 'coat rack', 'storage organizer', 'folded chair', 'fire alarm', 'power strip', 'calendar', 'poster', 'potted plant', 'luggage', 'mattress']
        self.object_token = self.tokenizer(self.object_texts)
        self.text_features = self.clip_model.encode_text(self.object_token)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        
    def __call__(self, image, group_ids):
        img_size = image.shape[0] * image.shape[1]

        unique_labels = np.unique(group_ids[group_ids > -1])
        object_ids = {}

        for label in unique_labels:
            rows, cols = np.where(group_ids == label)
            
            top_row = np.min(rows)
            bottom_row = np.max(rows)
            left_col = np.min(cols)
            right_col = np.max(cols)

            cropped_img = cv2.cvtColor(image[top_row:bottom_row+1, left_col:right_col+1], cv2.COLOR_BGR2RGB)
            # print(cropped_img.shape, cropped_img.shape[0] * cropped_img.shape[1] / img_size * 100)
            crop_per = cropped_img.shape[0] * cropped_img.shape[1] / img_size
            if 0.01 < crop_per and crop_per < 0.95:
                cropped_img = Image.fromarray(cropped_img)
                # cropped_img.show(title=f"Cropped object: {label}")
                with torch.no_grad(), torch.cuda.amp.autocast():
                    image_features = self.clip_model.encode_image(self.preprocess(cropped_img).unsqueeze(0))
                    
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                    text_probs = np.array((100.0 * image_features @ self.text_features.T).softmax(dim=-1))[0]
                    pred_idx = text_probs.argsort()[-1] #[::-1][:5]
                    object_ids[label] = pred_idx

            else:
                print(crop_per)
                group_ids[np.where(group_ids == label)] = -1

            # print(f"{label} Label prob: {text_probs[pred_idx]}")
            print(f"{label} Label pred:", np.array(self.object_texts)[pred_idx])
        
        return object_ids, group_ids

    def crop_mask_objects(seg_image):
        # Find unique labels in the mask, ignoring non assigned pixels (label -1)
        unique_labels = np.unique(seg_image[seg_image > -1])
        
        # Dictionary to hold cropped images for each label
        cropped_objects = []
        
        # Iterate over each label to find its bounding box and crop it
        for label in unique_labels:
            # Find rows and columns where this label appears
            rows, cols = np.where(seg_image == label)
            
            # Determine the bounding box of the current label
            top_row = np.min(rows)
            bottom_row = np.max(rows)
            left_col = np.min(cols)
            right_col = np.max(cols)
            
            # Crop the bounding box from the original mask
            cropped = seg_image[top_row:bottom_row+1, left_col:right_col+1]
            
            # Store the cropped image in the dictionary
            cropped_objects.append(cropped)
        
        return cropped_objects
    

def remove_group_ids_near_edge(group_ids, percentage=0.02):
    """
    Set mask IDs to -1 for masks too close to the edge of the image.

    Parameters:
    - group_ids: A 2D numpy array where each unique non-zero value represents a different mask ID.
    - percentage: The percentage of the image size to consider as the boundary. Masks within this percentage of the edge will be set to -1.

    Returns:
    - A modified mask with IDs set to -1 for those masks too close to the edge.
    """
    height, width = group_ids.shape
    
    # Calculate the number of pixels for the boundary based on the percentage
    n_top_bottom = int(height * percentage)  # For the top and bottom edges
    n_left_right = int(width * percentage)   # For the left and right edges
    
    # Create a border mask where edge pixels are True
    border_mask = np.zeros_like(group_ids, dtype=bool)
    border_mask[:n_top_bottom, :] = True  # Top edge
    border_mask[-n_top_bottom:, :] = True  # Bottom edge
    border_mask[:, :n_left_right] = True  # Left edge
    border_mask[:, -n_left_right:] = True  # Right edge

    # Find unique mask IDs, excluding -1
    unique_ids = np.unique(group_ids)
    unique_ids = unique_ids[unique_ids != -1]
    
    for group in unique_ids:
        mask_id_area = np.sum(group_ids == group)
        edge_close_area = np.sum(border_mask & (group_ids == group))

        # Calculate the percentage of the mask close to the edge
        percentage_close_to_edge = edge_close_area / mask_id_area.astype(np.float64) if mask_id_area > 0 else 0
        # Check if the percentage is above the threshold
        if percentage_close_to_edge > 0.7:
            # print("[INFO] Border line mask:", percentage_close_to_edge)
            group_ids[group_ids == group] = -1  # Set mask ID to -1

    return group_ids


def count_files(directory):
    # Ensure the directory exists
    if not os.path.exists(directory):
        print("[WARNING] depth_from_pc directory does not exist.")
        return -1
    
    # Initialize a count variable
    file_count = 0
    
    # Loop through files in the directory
    for filename in os.listdir(directory):
        # Check if the item is a file (not a directory)
        if os.path.isfile(os.path.join(directory, filename)):
            file_count += 1
    
    return file_count

def tuple_type(strings):
    return tuple(map(int, strings.split(',')))


def manage_temp_directory(dir_path, max_files=12):
    """
    Checks if a directory has more than max_files. If so, deletes the oldest file.

    Parameters:
        dir_path (str): The path to the directory to manage.
        max_files (int): The maximum number of files allowed in the directory.
                         Default is 10.
    """
    # Get list of all files in the directory, sorted by modification time (oldest first)
    files = sorted(glob.glob(os.path.join(dir_path, '*')), key=os.path.getmtime)

    # Check if the number of files exceeds max_files
    if len(files) > max_files:
        # Delete the oldest file
        oldest_file = files[0]
        os.remove(oldest_file)
        print(f"[INFO] Directory '{dir_path}' has {len(files)} files, deleted oldest file: {oldest_file}")
    else:
        print(f"[INFO] Directory '{dir_path}' has {len(files)} files, no need to delete any files.")
        
        
def get_data_file_path(args, scene_name, file_name, iter=0):
    color = join(args.rgb_path, scene_name, 'color', str(int(file_name[0:-4]) + iter) + '.jpg')
    depth = join(args.rgb_path, scene_name, 'depth', str(int(file_name[0:-4]) + iter) + '.png')
    pose = join(args.rgb_path, scene_name, 'pose', str(int(file_name[0:-4]) + iter) + '.txt')
    
    return color, depth, pose


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


def visualize_warped_mask_in_t_frame(warped_mask_t1_to_t, frame_data_dict_new):
    # Step 5: Overlay warped mask on RGB image for frame t
    unique_labels = np.unique(warped_mask_t1_to_t)
    colored_mask = np.zeros_like(frame_data_dict_new["color"])
    
    # Assign random colors for each label, excluding -1
    for label in unique_labels:
        if label == -1:
            continue  # Skip the label -1
        color = np.random.randint(0, 255, (1, 3), dtype=np.uint8).tolist()[0]
        colored_mask[warped_mask_t1_to_t == label] = color

    # Blend the colored mask with the original RGB image
    alpha = 0.5  # Transparency factor
    overlaid_image_t = cv2.addWeighted(frame_data_dict_new["color"], 1 - alpha, colored_mask, alpha, 0)
    
    return overlaid_image_t


def visualize_mask(frame_data_dict):
    # Step 5: Overlay warped mask on RGB image for frame t
    unique_labels = np.unique(frame_data_dict["mask_2d"])
    colored_mask = np.zeros_like(frame_data_dict["color"])
    
    # Assign random colors for each label, excluding -1
    for label in unique_labels:
        if label == -1:
            continue  # Skip the label -1
        color = np.random.randint(0, 255, (1, 3), dtype=np.uint8).tolist()[0]
        colored_mask[frame_data_dict["mask_2d"] == label] = color

    # Blend the colored mask with the original RGB image
    alpha = 0.5  # Transparency factor
    overlaid_image_t = cv2.addWeighted(frame_data_dict["color"], 1 - alpha, colored_mask, alpha, 0)
    
    return overlaid_image_t


# A dictionary to save and track assigned colors for each instance ID
color_map = {-1: [10, 10, 10]}

def get_color_for_id(id_):
    """Retrieve the color for a specific instance ID, assign a random color if not already assigned."""
    global color_map
    
    # If the color for this ID is not yet assigned, assign it
    if id_ not in color_map:
        # Ensure the colors are not red or black
        while True:
            color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            
            if color != [0, 0, 0] and color != [0, 0, 255]:  # Avoid black and red                
                color_map[id_] = color
                break
    
    return color_map[id_]


def visualize_matching_results_over_time(current_mask, warped_masks):
    """Visualize the matching results by putting the current mask on the left and sequentially adding t-1, t-2, ... masks on the right."""
    
    # Generate a consistent color map for instance IDs in all masks
    def assign_consistent_colors(mask):
        # Apply the color map to the mask
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
        for id_ in np.unique(mask):
            color = get_color_for_id(id_)
            colored_mask[mask == id_] = color
        
        return colored_mask
    
    # Start with the current mask (on the left)
    visualizations = [assign_consistent_colors(current_mask)]

    # Add each warped mask (sequentially) to the right
    for warped_mask in warped_masks:
        colored_warped_mask = assign_consistent_colors(warped_mask)
        visualizations.append(colored_warped_mask)
    
    # Concatenate all the visualizations side by side
    concatenated_result = np.concatenate(visualizations, axis=1)
    
    return concatenated_result


def visualize_matching_result(warped_mask, current_mask):
    """Visualize the matching result between warped_mask and current_mask by assigning random colors to each instance."""

    # Assign consistent colors to matching IDs and mark non-common IDs in red
    colored_warped_mask, colored_current_mask = assign_consistent_colors(warped_mask, current_mask)
    
    # Concatenate the images side by side for comparison
    concatenated_result = np.concatenate((colored_warped_mask, colored_current_mask), axis=1)
    
    return concatenated_result


# Generate a consistent color map for instance IDs present in both masks
def assign_consistent_colors(mask1, mask2):
    # Find unique IDs in each mask
    unique_ids_mask1 = np.unique(mask1)
    unique_ids_mask2 = np.unique(mask2)

    # Find common and non-common IDs
    common_ids = np.intersect1d(unique_ids_mask1, unique_ids_mask2)
    mask1_only_ids = np.setdiff1d(unique_ids_mask1, common_ids)
    mask2_only_ids = np.setdiff1d(unique_ids_mask2, common_ids)
    
    # Apply the color map to both masks
    colored_mask1 = np.zeros((mask1.shape[0], mask1.shape[1], 3), dtype=np.uint8)
    colored_mask2 = np.zeros((mask2.shape[0], mask2.shape[1], 3), dtype=np.uint8)
    
    # Assign colors for common IDs
    for id_ in common_ids:
        color = get_color_for_id(id_)
        colored_mask1[mask1 == id_] = color
        colored_mask2[mask2 == id_] = color
    
    # Assign red color for IDs that are in only one mask
    for id_ in mask1_only_ids:
        colored_mask1[mask1 == id_] = [0, 0, 255]  # Red color for non-common IDs in mask1
    for id_ in mask2_only_ids:
        colored_mask2[mask2 == id_] = [0, 0, 255]  # Red color for non-common IDs in mask2

    return colored_mask1, colored_mask2


# Generate random colors for each unique instance in the mask
def assign_random_colors(mask):
    unique_ids = np.unique(mask)
    color_map = {id_: [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for id_ in unique_ids if id_ != -1}
    color_map[-1] = [0, 0, 0]  # Assign black to background or unassigned area
    
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for id_ in unique_ids:
        colored_mask[mask == id_] = color_map[id_]
    
    return colored_mask


def update_intensity(intensity_dict, pcd_dict):
    for id in np.unique(pcd_dict["group"]):
            if id == -1:
                continue
            if id in intensity_dict:
                intensity_dict[id] += 1
            else:
                intensity_dict[id] = 1
                
    return intensity_dict