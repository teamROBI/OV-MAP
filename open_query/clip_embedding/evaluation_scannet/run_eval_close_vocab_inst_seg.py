import os
import numpy as np
import clip
import torch
import pdb
from eval_semantic_instance import evaluate
from scannet_constants import SCANNET_COLOR_MAP_40, VALID_CLASS_IDS_20, CLASS_LABELS_20, SCANNET_COLOR_MAP_200, VALID_CLASS_IDS_200, CLASS_LABELS_200
import tqdm
import argparse

import multiprocessing
from functools import partial

class InstSegEvaluator():
    def __init__(self, dataset_type, clip_model_type, sentence_structure, class_agnostic=False):  # Add class_agnostic flag
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_type = dataset_type
        self.clip_model_type = clip_model_type
        self.clip_model = self.get_clip_model(clip_model_type)
        self.query_sentences = self.get_query_sentences(dataset_type, sentence_structure)
        self.feature_size = self.get_feature_size(clip_model_type)
        self.text_query_embeddings = self.get_text_query_embeddings().numpy() # torch.Size([20, 768])
        self.set_label_and_color_mapper(dataset_type)
        self.class_agnostic = class_agnostic  # Store the class_agnostic flag

    def get_query_sentences(self, dataset_type, sentence_structure="a {} in a scene"):
        if dataset_type == 'scannet':
            label_list = list(CLASS_LABELS_20)
            label_list[-1] = 'other' # replace otherfurniture with other, following OpenScene
        elif dataset_type == 'scannet200':
            label_list = list(CLASS_LABELS_200)
        else:
            raise NotImplementedError
        return [sentence_structure.format(label) for label in label_list]

    def get_clip_model(self, clip_model_type):
        clip_model, _ = clip.load(clip_model_type, self.device)
        return clip_model

    def get_feature_size(self, clip_model_type):
        if clip_model_type == 'ViT-L/14' or clip_model_type == 'ViT-L/14@336px':
            return 768
        elif clip_model_type == 'ViT-B/32':
            return 512
        else:
            raise NotImplementedError

    def get_text_query_embeddings(self):
        # ViT_L14_336px for OpenSeg, clip_model_vit_B32 for LSeg
        text_query_embeddings = torch.zeros((len(self.query_sentences), self.feature_size))

        for label_idx, sentence in enumerate(self.query_sentences):
            #print(label_idx, sentence) #CLASS_LABELS_20[label_idx],
            text_input_processed = clip.tokenize(sentence).to(self.device)
            with torch.no_grad():
                sentence_embedding = self.clip_model.encode_text(text_input_processed)

            sentence_embedding_normalized =  (sentence_embedding/sentence_embedding.norm(dim=-1, keepdim=True)).float().cpu()
            text_query_embeddings[label_idx, :] = sentence_embedding_normalized

        return text_query_embeddings
    
    def set_label_and_color_mapper(self, dataset_type):
        if dataset_type == 'scannet':
            self.label_mapper = np.vectorize({idx: el for idx, el in enumerate(VALID_CLASS_IDS_20)}.get)
            self.color_mapper = np.vectorize(SCANNET_COLOR_MAP_20.get)
        elif dataset_type == 'scannet200':
            self.label_mapper = np.vectorize({idx: el for idx, el in enumerate(VALID_CLASS_IDS_200)}.get)
            self.color_mapper = np.vectorize(SCANNET_COLOR_MAP_200.get)
        else:
            raise NotImplementedError

    def compute_classes_per_mask(self, masks_path, mask_features_path, keep_first=None):
        masks = torch.load(masks_path)
        mask_features = np.load(mask_features_path)

        if keep_first is not None:
            masks = masks[:, 0:keep_first]
            mask_features = mask_features[0:keep_first, :]

        # Normalize mask features
        mask_features_normalized = mask_features/np.linalg.norm(mask_features, axis=1)[..., None]

        # For class-agnostic, skip the class computation and assign all to a default class
        if self.class_agnostic:
            pred_classes = np.zeros(mask_features.shape[0], dtype=int)  # Assign all instances to class 0
            max_class_similarity_scores = np.ones(mask_features.shape[0])  # Dummy scores
        else:
            similarity_scores = mask_features_normalized @ self.text_query_embeddings.T  # (N, 20)
            max_class_similarity_scores = np.max(similarity_scores, axis=1)
            max_ind = np.argmax(similarity_scores, axis=1)
            max_ind_remapped = self.label_mapper(max_ind)
            pred_classes = max_ind_remapped

        return masks, pred_classes, max_class_similarity_scores

    def compute_classes_per_mask_diff_scores(self, masks_path, mask_features_path, keep_first=None):
        pred_masks = torch.load(masks_path)
        
        keep_mask = np.asarray([True for _ in range(pred_masks.shape[1])])
        if keep_first:
            keep_mask[keep_first:] = False
        
        # For class-agnostic, skip the class computation and assign all to a default class
        if self.class_agnostic:
            pred_classes = np.zeros(pred_masks.shape[1], dtype=int)  # Assign all to class 0        
        else:
            mask_features = np.load(mask_features_path)
            
            # Normalize mask features
            mask_features_normalized = mask_features/np.linalg.norm(mask_features, axis=1)[..., None]
            mask_features_normalized[np.isnan(mask_features_normalized) | np.isinf(mask_features_normalized)] = 0.0

            per_class_similarity_scores = mask_features_normalized @ self.text_query_embeddings.T  # (N, 20)
            max_ind = np.argmax(per_class_similarity_scores, axis=1)
            max_ind_remapped = self.label_mapper(max_ind)
            pred_classes = max_ind_remapped

        pred_masks = pred_masks[:, keep_mask]
        pred_scores = np.ones(pred_classes.shape)  # Keep the scores as 1 for simplicity

        return pred_masks, pred_classes, pred_scores

    def evaluate_full(self, preds, scene_gt_dir, dataset, output_file='temp_evaluation_output.txt'):
        #pred_masks.shape, pred_scores.shape, pred_classes.shape #((237360, 177), (177,), (177,))

        inst_AP = evaluate(preds, scene_gt_dir, output_file=output_file, dataset=dataset, class_agnostic=self.class_agnostic)
        # read .txt file: scene0000_01.txt has three parameters each row: the mask file for the instance, the id of the instance, and the score. 

        return inst_AP
    
# # Define the function to process a single scene
# def process_scene(scene_name, pred_root_dir, masks_template, mask_features_dir, feature_file_template, evaluator, keep_first):
#     masks_path = os.path.join(pred_root_dir, scene_name + masks_template)
#     scene_per_mask_feature_path = os.path.join(mask_features_dir, feature_file_template.format(scene_name))

#     if not os.path.exists(scene_per_mask_feature_path):
#         print('--- SKIPPING ---', scene_per_mask_feature_path)
#         return scene_name, None

#     pred_masks, pred_classes, pred_scores = evaluator.compute_classes_per_mask_diff_scores(
#         masks_path=masks_path,
#         mask_features_path=scene_per_mask_feature_path,
#         keep_first=keep_first
#     )
    
#     return scene_name, {
#         'pred_masks': pred_masks,
#         'pred_scores': pred_scores,
#         'pred_classes': pred_classes
#     }


def test_pipeline_full_scannet200(mask_features_dir,
                                  gt_dir,
                                  pred_root_dir,
                                  sentence_structure,
                                  feature_file_template,
                                  dataset_type='scannet200',
                                  clip_model_type='ViT-L/14@336px',
                                  keep_first=None,
                                  class_agnostic=False,  # New argument for class-agnostic evaluation
                                  scene_list_file='evaluation/val_scenes_scannet200.txt',
                                  masks_template='.pt',
                                  output_file='temp_evaluation_output.txt'):

    evaluator = InstSegEvaluator(dataset_type, clip_model_type, sentence_structure, class_agnostic=class_agnostic)  # Pass class_agnostic to evaluator
    print('[INFO]', dataset_type, clip_model_type, sentence_structure, f"Class-agnostic: {class_agnostic}")

    with open(scene_list_file, 'r') as f:
        scene_names = f.read().splitlines()

    preds = {}

    for scene_name in tqdm.tqdm(scene_names[:]):
        masks_path = os.path.join(pred_root_dir, scene_name + masks_template)
        
        if class_agnostic:
            scene_per_mask_feature_path = None
        else:
            scene_per_mask_feature_path = os.path.join(mask_features_dir, feature_file_template.format(scene_name))            
    
            if not os.path.exists(scene_per_mask_feature_path):
                print('--- SKIPPING ---', scene_per_mask_feature_path)
                continue
        
        pred_masks, pred_classes, pred_scores = evaluator.compute_classes_per_mask_diff_scores(
            masks_path=masks_path, 
            mask_features_path=scene_per_mask_feature_path,
            keep_first=keep_first
        )

        preds[scene_name] = {
            'pred_masks': pred_masks,
            'pred_scores': pred_scores,
            'pred_classes': pred_classes}
        
    # with multiprocessing.Pool() as pool:
    #     # Use tqdm to track the progress
    #     process_func = partial(process_scene, pred_root_dir=pred_root_dir, masks_template=masks_template, 
    #                            mask_features_dir=mask_features_dir, feature_file_template=feature_file_template, 
    #                            evaluator=evaluator, keep_first=keep_first)

    #     results = list(tqdm.tqdm(pool.imap(process_func, scene_names), total=len(scene_names)))

    # # Gather results in a dictionary
    # preds = {scene_name: result for scene_name, result in results if result is not None}

    inst_AP = evaluator.evaluate_full(preds, gt_dir, dataset=dataset_type, output_file=output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, help='path to directory of GT .txt files')
    parser.add_argument('--mask_pred_dir', type=str, help='path to the saved class agnostic masks')
    parser.add_argument('--mask_features_dir', type=str, default=None, help='path to the saved mask features')
    parser.add_argument('--feature_file_template', type=str, default="{}.npy")
    parser.add_argument('--sentence_structure', type=str, default="a {} in a scene", help='sentence structure for 3D closed-set evaluation')
    parser.add_argument('--scene_list_file', type=str, default="evaluation_scannet/val_scenes_scannet200.txt")
    parser.add_argument('--masks_template', type=str, default=".pt")
    parser.add_argument('--evaluation_output_dir', type=str, default="temp_evaluation_output.txt")
    parser.add_argument('--class_agnostic', action='store_true', help='flag to enable class-agnostic evaluation')  # Add the flag

    opt = parser.parse_args()

    # ScanNet200, "a {} in a scene", all masks are assigned 1.0 as the confidence score
    test_pipeline_full_scannet200(opt.mask_features_dir, opt.gt_dir, opt.mask_pred_dir, opt.sentence_structure,
                                  opt.feature_file_template, dataset_type='scannet200', clip_model_type='ViT-L/14@336px',
                                  keep_first=None, class_agnostic=opt.class_agnostic,  # Pass the flag
                                  scene_list_file=opt.scene_list_file, masks_template=opt.masks_template,
                                  output_file=opt.evaluation_output_dir)