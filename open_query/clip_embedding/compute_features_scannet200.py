import hydra
from omegaconf import DictConfig
import numpy as np
from data.load import Camera, InstanceMasks3D, Images, PointCloud, get_number_of_images
from utils import get_free_gpu, create_out_folder
from mask_features_computation.features_extractor import FeaturesExtractor
import torch
import os
from glob import glob
import pickle

# TIP: add version_base=None to the arguments if you encounter some error  
@hydra.main(config_path="configs", config_name="openmask3d_scannet200_eval")
def main(ctx: DictConfig):
    device = "cpu"  # "mps" if torch.backends.mps.is_available() else "cpu"
    device = get_free_gpu(7000, ctx.gpu.device_num) if torch.cuda.is_available() else device
    print(f"[INFO] Using device: {device}")
    out_folder = ctx.output.output_directory
    os.chdir(hydra.utils.get_original_cwd())
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    print(f"[INFO] Saving feature results to {out_folder}")
    
    if ctx.debug.save_subs:
        subs_out_folder = out_folder.replace('mask_features', 'sub_mask_features')
        if not os.path.exists(subs_out_folder):
            os.makedirs(subs_out_folder)
        print(f"[INFO] Saving sub feature results to {subs_out_folder}")
    
    masks_paths = sorted(glob(os.path.join(ctx.data.masks.masks_path, ctx.data.masks.masks_suffix)))
    
    print("[INFO] Total scenes:", len(masks_paths))
    part_length = len(masks_paths) // ctx.multiprocess.subprocess_num
    start = ctx.multiprocess.part * part_length
    if ctx.multiprocess.part == ctx.multiprocess.subprocess_num - 1:
        end = len(masks_paths)
    else:
        end = start + part_length
    print(f"Subprocess {start} : {end}")

    for masks_path in masks_paths[start:end]:        
        scene_num_str = masks_path.split('/')[-1][5:12]
        
        filename = f"scene{scene_num_str}.npy"
        output_path = os.path.join(out_folder, filename)
        if os.path.isfile(output_path):
            print(f"[INFO] {output_path} exists, skipping scene {scene_num_str}")
            continue
        
        print(f"[INFO] Processing: scene{scene_num_str}")
        path = ctx.data.scans_path
        point_cloud_path = glob(os.path.join(path, 'scene'+ scene_num_str+'.ply'))[0]

        process_path = os.path.join(ctx.data.scans_2d_path, 'scene'+ scene_num_str)
        poses_path = os.path.join(process_path,ctx.data.camera.poses_path)
        intrinsic_path = os.path.join(process_path, ctx.data.camera.intrinsic_path)
        images_path = os.path.join(process_path, ctx.data.images.images_path)
        depths_path = os.path.join(process_path, ctx.data.depths.depths_path)
        syn_depths_path = os.path.join(ctx.data.synthetic_depth_path, 'scene'+ scene_num_str)

        # 1. Load the masks
        masks = InstanceMasks3D(masks_path) 

        # 2. Load the images
        total_images = get_number_of_images(poses_path)
        indices = np.arange(0, total_images, step = ctx.openmask3d.frequency)
        print(f"[INFO] Total image count: {total_images}. Selected {len(indices)} images with frequency {ctx.openmask3d.frequency}")
        images = Images(images_path=images_path, 
                        extension=ctx.data.images.images_ext, 
                        indices=indices)

        # 3. Load the pointcloud
        pointcloud = PointCloud(point_cloud_path)

        # 4. Load the camera configurations
        camera = Camera(images=images,
                        intrinsic_path=intrinsic_path, 
                        intrinsic_resolution=ctx.data.camera.intrinsic_resolution, 
                        poses_path=poses_path, 
                        depths_path=depths_path,
                        extension_depth=ctx.data.depths.depths_ext, 
                        depth_scale=ctx.data.depths.depth_scale,
                        syn_depths_path=syn_depths_path,
                        depth_supplemented=ctx.data.depths.depth_supplemented)

        # 5. Run extractor
        features_extractor = FeaturesExtractor(camera=camera, 
                                                clip_model=ctx.external.clip_model, 
                                                images=images, 
                                                masks=masks,
                                                pointcloud=pointcloud, 
                                                sam_model_type=ctx.external.sam_model_type,
                                                sam_checkpoint=ctx.external.sam_checkpoint,
                                                vis_threshold=ctx.openmask3d.vis_threshold,
                                                device=device,
                                                debug_vis_mode = ctx.debug.vis_mode)

        features, sub_features = features_extractor.extract_features(topk=ctx.openmask3d.top_k, 
                                                        multi_level_expansion_ratio = ctx.openmask3d.multi_level_expansion_ratio,
                                                        num_levels=ctx.openmask3d.num_of_levels, 
                                                        num_random_rounds=ctx.openmask3d.num_random_rounds,
                                                        num_selected_points=ctx.openmask3d.num_selected_points,
                                                        save_crops=ctx.output.save_crops,
                                                        out_folder=out_folder,
                                                        save_subs=ctx.debug.save_subs,
                                                        optimize_gpu_usage=ctx.gpu.optimize_gpu_usage)
        
        # 6. Save features
        np.save(output_path, features)
        print(f"[INFO] Mask features for scene {scene_num_str} saved to {output_path}.")
        
        if ctx.debug.save_subs:
            subs_output_path = os.path.join(subs_out_folder, filename.replace('.npy', '.pkl'))
            with open(subs_output_path, 'wb') as f:
                pickle.dump(sub_features, f)
            print(f"[INFO] Sub mask features for scene {scene_num_str} saved to {subs_output_path}.")
    
    
    
if __name__ == "__main__":
    main()