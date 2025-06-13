import sys
sys.path.append("./openmask3d/openmask3d/visualization")

import numpy as np
import open3d as o3d
import os
from os.path import join

import torch
import clip
import pdb
import matplotlib.pyplot as plt
from constants import *
from scannet_constants import SCANNET_COLOR_MAP_40, CLASS_IDS_40, CLASS_LABELS_40

class QuerySimilarityComputation():
    def __init__(self,):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.clip_model, _ = clip.load('ViT-L/14@336px', self.device)

    def get_query_embedding(self, text_query):
        text_input_processed = clip.tokenize(text_query).to(self.device)
        with torch.no_grad():
            sentence_embedding = self.clip_model.encode_text(text_input_processed)

        sentence_embedding_normalized =  (sentence_embedding/sentence_embedding.norm(dim=-1, keepdim=True)).float().cpu()
        return sentence_embedding_normalized.squeeze().numpy()

    def compute_similarity_scores(self, mask_features, text_query):
        text_emb = self.get_query_embedding(text_query)

        scores = np.zeros(len(mask_features))
        for mask_idx, mask_emb in enumerate(mask_features):
            mask_norm = np.linalg.norm(mask_emb)
            if mask_norm < 0.001:
                continue
            normalized_emb = (mask_emb/mask_norm)
            scores[mask_idx] = normalized_emb@text_emb

        return scores

    def get_per_point_colors_for_similarity(self, 
                                            per_mask_scores, 
                                            masks, 
                                            normalize_based_on_current_min_max=False, 
                                            normalize_min_bound=0.16, #only used for visualization if normalize_based_on_current_min_max is False
                                            normalize_max_bound=0.26, #only used for visualization if normalize_based_on_current_min_max is False
                                            background_color=(0.77, 0.77, 0.77)
                                        ):
        # get colors based on the openmask3d per mask scores
        non_zero_points = per_mask_scores!=0
        openmask3d_per_mask_scores_rescaled = np.zeros_like(per_mask_scores)
        pms = per_mask_scores[non_zero_points]

        # in order to be able to visualize the score differences better, we can use a normalization scheme
        if normalize_based_on_current_min_max: # if true, normalize the scores based on the min. and max. scores for this scene
            openmask3d_per_mask_scores_rescaled[non_zero_points] = (pms-pms.min()) / (pms.max() - pms.min())
        else: # if false, normalize the scores based on a pre-defined color scheme with min and max clipping bounds, normalize_min_bound and normalize_max_bound.
            new_scores = np.zeros_like(openmask3d_per_mask_scores_rescaled)
            new_indices = np.zeros_like(non_zero_points)
            new_indices[non_zero_points] += pms>normalize_min_bound
            new_scores[new_indices] = ((pms[pms>normalize_min_bound]-normalize_min_bound)/(normalize_max_bound-normalize_min_bound))
            openmask3d_per_mask_scores_rescaled = new_scores

        new_colors = np.ones((masks.shape[1], 3))*0 + background_color
        
        for mask_idx, mask in enumerate(masks[::-1, :]):
            # get color from matplotlib colormap
            new_colors[mask>0.5, :] = plt.cm.jet(openmask3d_per_mask_scores_rescaled[len(masks)-mask_idx-1])[:3]

        return new_colors


class InstSegEvaluator():
    def __init__(self, sentence_structure):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.clip_model, _ = clip.load('ViT-L/14@336px', self.device)
        self.query_sentences = self.get_query_sentences(sentence_structure)
        self.feature_size = 768
        self.text_query_embeddings = self.get_text_query_embeddings().numpy()  # torch.Size([20, 768])
        self.set_label_and_color_mapper()

    def set_label_and_color_mapper(self):
        self.label_mapper = np.vectorize({idx: el for idx, el in enumerate(CLASS_IDS_40)}.get)
        self.color_mapper = np.vectorize(SCANNET_COLOR_MAP_40.get)

    def get_query_sentences(self, sentence_structure="a {} in a scene"):
        label_list = list(CLASS_LABELS_40)
        label_list[-1] = 'other'  # replace otherfurniture with other, following OpenScene
        return [sentence_structure.format(label) for label in label_list]

    def get_text_query_embeddings(self):
        # ViT_L14_336px for OpenSeg, clip_model_vit_B32 for LSeg
        text_query_embeddings = torch.zeros((len(self.query_sentences), self.feature_size))

        for label_idx, sentence in enumerate(self.query_sentences):
            # print(label_idx, sentence) #CLASS_LABELS_20[label_idx],
            text_input_processed = clip.tokenize(sentence).to(self.device)
            with torch.no_grad():
                sentence_embedding = self.clip_model.encode_text(text_input_processed)

            sentence_embedding_normalized = (
                        sentence_embedding / sentence_embedding.norm(dim=-1, keepdim=True)).float().cpu()
            text_query_embeddings[label_idx, :] = sentence_embedding_normalized

        return text_query_embeddings

    def compute_classes_per_mask_diff_scores(self, masks_path, mask_features_path, keep_first=None):
        pred_masks = torch.load(masks_path)
        mask_features = np.load(mask_features_path)
        keep_mask = np.asarray([True for el in range(pred_masks.shape[1])])
        if keep_first:
            keep_mask[keep_first:] = False

        # normalize mask features
        mask_features_normalized = mask_features / np.linalg.norm(mask_features, axis=1)[..., None]
        mask_features_normalized[np.isnan(mask_features_normalized) | np.isinf(mask_features_normalized)] = 0.0

        per_class_similarity_scores = mask_features_normalized @ self.text_query_embeddings.T  # (177, 20)
        max_ind = np.argmax(per_class_similarity_scores, axis=1)
        max_ind_remapped = self.label_mapper(max_ind)

        pred_masks = pred_masks[:, keep_mask]
        pred_classes = max_ind_remapped[keep_mask]
        pred_scores = np.ones(pred_classes.shape)

        return pred_masks, pred_classes, pred_scores

    def visualize(self, scene_pcd, color_list):
        instance_scene = o3d.geometry.PointCloud()
        instance_scene.points = scene_pcd.points
        instance_scene.colors = o3d.utility.Vector3dVector(color_list/255.0)
        o3d.visualization.draw_geometries([instance_scene])
        # alternatively, you can save the scene_pcd_w_sim_colors as a .ply file
        # o3d.io.write_point_cloud("data/scene_pcd_w_sim_colors_{}.ply".format('_'.join(query_text.split(' '))), scene_pcd_w_sim_colors)

def ScannetDataMode(scene_path, scene_per_mask_feature_path, save_pth, keep_first, visualize=False, save=False):
    base_name = os.path.splitext(os.path.basename(scene_per_mask_feature_path))[0]
    scene_pcd = o3d.io.read_point_cloud(scene_path)

    label_dict_path = join(save_pth, 'label_dict')
    label_dict = np.load(join(label_dict_path, base_name+'.npy'), allow_pickle=True)
    label_dict = label_dict.item()

    label_path = join(save_pth, 'ensemble')
    labels = torch.load(join(label_path, base_name+'.pth'))

    mask_array_path = join(save_pth, 'mask_array')
    mask_array_path = join(mask_array_path, base_name+'.pt')
    sentence_structure = "a {} in a scene"
    evaluator = InstSegEvaluator(sentence_structure)
    print('Computing Mask Classes Score...')
    print(mask_array_path)
    print(scene_per_mask_feature_path)

    pred_masks, pred_classes, pred_scores = evaluator.compute_classes_per_mask_diff_scores(masks_path=mask_array_path,
                                                                                            mask_features_path=scene_per_mask_feature_path,
                                                                                            keep_first=keep_first)

    for idx, key in enumerate(label_dict):
        label_dict[key] = pred_classes[idx]

    if save:
        scan_mask_label_path = join(save_pth, 'scan_mask_label')
        np.save(join(scan_mask_label_path, base_name + '.npy'), pred_classes)

    if visualize:
        mapped_colors = []
        for idx, label in enumerate(labels):
            mapped_colors.append(evaluator.color_mapper(label_dict[label]))

        # mapped_colors = evaluator.color_mapper(labels)
        mapped_colors = np.array(mapped_colors)
        evaluator.visualize(scene_pcd, np.array(mapped_colors))
def InputQueryMode(path_scene_pcd, path_openmask3d_features):
    # --------------------------------
    # Load data
    # --------------------------------
    base_name = os.path.splitext(os.path.basename(path_openmask3d_features))[0]
    save_pth = '/home/tidy/PycharmProjects/SegmentAnything3D/save'
    path_pred_masks = join(save_pth, 'mask_array')
    path_pred_masks = join(path_pred_masks, base_name+'.pt')

    # load the scene pcd
    scene_pcd = o3d.io.read_point_cloud(path_scene_pcd)

    # load the predicted masks
    pred_masks = np.asarray(torch.load(path_pred_masks)).T  # (num_instances, num_points)

    # load the openmask3d features
    openmask3d_features = np.load(path_openmask3d_features)  # (num_instances, 768)

    # initialize the query similarity computer
    query_similarity_computer = QuerySimilarityComputation()

    # --------------------------------
    # Set the query text
    # --------------------------------
    while True:
        query_text = input("Input text query: ")  # "sofa" # change the query text here
        if query_text == 'q':
            break

        # --------------------------------
        # Get the similarity scores
        # --------------------------------
        # get the per mask similarity scores, i.e. the cosine similarity between the query embedding and each openmask3d mask-feature for each object instance
        per_mask_query_sim_scores = query_similarity_computer.compute_similarity_scores(openmask3d_features, query_text)

        # --------------------------------
        # Visualize the similarity scores
        # --------------------------------
        # get the per-point heatmap colors for the similarity scores
        per_point_similarity_colors = query_similarity_computer.get_per_point_colors_for_similarity(
            per_mask_query_sim_scores,
            pred_masks)  # note: for normalizing the similarity heatmap colors for better clarity, you can check the arguments for the function get_per_point_colors_for_similarity

        # visualize the scene with the similarity heatmap
        scene_pcd_w_sim_colors = o3d.geometry.PointCloud()
        scene_pcd_w_sim_colors.points = scene_pcd.points
        scene_pcd_w_sim_colors.colors = o3d.utility.Vector3dVector(per_point_similarity_colors)
        scene_pcd_w_sim_colors.estimate_normals()
        o3d.visualization.draw_geometries([scene_pcd_w_sim_colors])
        # alternatively, you can save the scene_pcd_w_sim_colors as a .ply file
        # o3d.io.write_point_cloud("data/scene_pcd_w_sim_colors_{}.ply".format('_'.join(query_text.split(' '))), scene_pcd_w_sim_colors)


def main():
    # --------------------------------
    # Set the paths
    # # --------------------------------
    path_scene_pcd = "/home/tidy/PycharmProjects/SegmentAnything3D/pointcept_process/val/scene0644_00.ply"
    path_mask_features = "/home/tidy/PycharmProjects/SegmentAnything3D/openmask3d/output/ours/mask_features/scene0644_00.npy" #mask_openmask3d_features.npy
    save_pth = '/home/tidy/PycharmProjects/SegmentAnything3D/save'
    print('SELECT EVAL MODE ==========================')
    print('1: Input text query\n'
          '2: Use Scannet Dataset labels')
    num = int(input())

    if num == 1:
        InputQueryMode(path_scene_pcd, path_mask_features)
    elif num == 2:
        ScannetDataMode(path_scene_pcd, path_mask_features, save_pth, keep_first=None, visualize=True, save=True)
    else:
        print('Invalid number!')

if __name__ == "__main__":
    # pass
    main()
