#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import glob
import pdb

# CLASSES = ('__background__',
#            'pedestrian', 'car', 'cyclist')

CLASSES = ('__background__',
           'car')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
               'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, save_path, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    # print("inds:{}".format(inds))
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    # plt.savefig(save_path)
    # plt.close()
    plt.show()


def plot_detections_on_image(im, dets, savename=None):
    """
    dets : [bbox,score, cls_ind]
    class_name = CLASSES[cls_ind].
    """
    import matplotlib.pyplot as plt
    inds = np.where(dets[:, -1] > 0.1)[0]
    if len(inds) == 0:
        if savename is not None:
            cv2.imwrite(savename, im)
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -2]
        classname = CLASSES[int(dets[i, -1])]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )

        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(classname, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    plt.axis('off')
    plt.tight_layout()
    plt.draw()

    # save and close
    if savename is not None:
        plt.savefig(savename)
    plt.close()


def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_name)
    path, name = os.path.split(image_name)
    save_path = os.path.join('/home/yangshun/PycharmProjects/faster_rcnn/data/demo/kitti2', name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.001
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        # print("cls_ind:{}, cls:{}.".format(cls_ind, cls))
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, save_path, thresh=CONF_THRESH)
        # plot_detections_on_image(im, dets)


def generate_evaluate_txt(net, image_name, save_root):
    im = cv2.imread(image_name)
    scores, boxes = im_detect(net, im)

    # save_root = '/media/yangshun/0008EB70000B0B9F/roi_roi_new/kitti/txt/faster_rcnn_context/data'
    if not (os.path.exists(save_root)):
        os.makedirs(save_root)
    save_dir = os.path.join(save_root, os.path.split(image_name)[1].split('.')[0] + '.txt')
    result_file = open(save_dir, 'w')

    CONF_THRESH = 0.5
    NMS_THRESH = 0.3

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]

        if len(inds) == 0:
            continue
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            result_file.write("%s -1 -1 -10 %.3f %.3f %.3f %.3f -1 -1 -1 -1000 -1000 -1000 -10 %.8f\n" %
                              (cls.capitalize(), bbox[0], bbox[1], bbox[2], bbox[3], score))
            # result_file.close()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]', default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode', help='Use CPU mode (overrides --gpu)', action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]', choices=NETS.keys(), default='vgg16')
    parser.add_argument('--prototxt', dest='prototxt', help='test prototxt file path', default='')
    parser.add_argument('--model', dest='model', help='model path', default='')
    parser.add_argument('--save_root', dest='save_root', help='test result save root', default='')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    # prototxt = '/media/yangshun/0008EB70000B0B9F/PycharmProjects/faster_rcnn/models/kitti/VGG16/' \
    #            'faster_rcnn_end2end_context/test.prototxt'
    # caffemodel = '/media/yangshun/0008EB70000B0B9F/roi_roi_new/kitti/faster_rcnn_context/kittivoc_train/' \
    #              'vgg16_faster_rcnn_iter_70000.caffemodel'

    prototxt = args.prototxt
    caffemodel = args.model

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print('\n\nLoaded network {:s}'.format(caffemodel))

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = im_detect(net, im)

    # kitti_test_root = '/media/yangshun/0008EB70000B0B9F/IMG_9037'
    # test_txt = '/media/yangshun/0008EB70000B0B9F/PycharmProjects/faster_rcnn/data/KITTIVOC/ImageSets/Main/test.txt'
    #
    # with open(test_txt, 'rb') as f:
    #     lines = f.readlines()
    #
    # # im_names = ['car2.jpg', 'car3.jpg', 'person2.jpg', 'person4.jpg']
    # # im_names = [os.path.join('../data/demo', path) for path in im_names]
    # im_names = [os.path.join(kitti_test_root, '{}.png'.format(line.strip())) for line in lines]

    kitti_test_root_dir = '/media/yangshun/0008EB70000B0B9F/IMG_9037'
    im_names = [os.path.join(kitti_test_root_dir, name) for name in os.listdir(kitti_test_root_dir)]

    im_names = sorted(im_names)
    for im_name in im_names:
        # print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print('Demo for {}'.format(im_name))
        # demo(net, im_name)
        generate_evaluate_txt(net, im_name, args.save_root)

        # plt.show()
