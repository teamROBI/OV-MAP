CONFIG='/home/robi/PycharmProjects/SegmentAnything3D/detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/cropformer_hornet_3x.yaml'
# INPUT='/home/robi/PycharmProjects/SegmentAnything3D/scannetv2_images/scene0000_00/color/2000.jpg'
INPUT='/home/robi/PycharmProjects/SegmentAnything3D/detectron2/projects/CropFormer/demo_cropformer/input/*.png'
# OUTPUT='/home/robi/PycharmProjects/SegmentAnything3D/save2/save_crop_seg'
OUTPUT='/home/robi/PycharmProjects/SegmentAnything3D/detectron2/projects/CropFormer/demo_cropformer/output'
WEIGHTS='/home/robi/PycharmProjects/SegmentAnything3D/detectron2/projects/CropFormer/weights/CropFormer_hornet_3x_03823a.pth'

python3 demo_from_dirs.py --config-file $CONFIG --input $INPUT --output $OUTPUT --model-weights $WEIGHTS