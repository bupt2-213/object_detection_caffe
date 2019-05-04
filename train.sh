#!/usr/bin/env bash

SAVE_DIR="multi_scales"
MODEL_NAME="multi_scales"
KITTI_ROOT=/media/yangshun/0008EB70000B0B9F/roi_roi_new/kitti

MODEL_DIR=${KITTI_ROOT}/${SAVE_DIR}/kittivoc_train/vgg16_faster_rcnn_iter_70000.caffemodel
PROTOTXT=models/kitti/VGG16/${MODEL_NAME}/test.prototxt
TXT_SAVE_DIR=/media/yangshun/0008EB70000B0B9F/roi_roi_new/kitti/txt/${SAVE_DIR}/data


#./experiments/scripts/faster_rcnn_end2end_kitti.sh 0 VGG16 kitti_voc ${MODEL_NAME} \
#    --set EXP_DIR ${KITTI_ROOT}/${SAVE_DIR}

python tools/demo_kitti_on_one_image.py \
    --prototxt ${PROTOTXT} \
    --model ${MODEL_DIR} \
    --save_root ${TXT_SAVE_DIR}