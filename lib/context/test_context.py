import numpy as np
from PIL import Image
import cv2

w_scale = 2
h_scale = 2
x_rel_pos = 0.5
y_rel_pos = 0.5


def _get_context_rois(im_size, im_rois):

    context_rois = np.zeros_like(im_rois)
    center_x = (im_rois[:, 0] + im_rois[:, 2]) / 2
    center_y = (im_rois[:, 1] + im_rois[:, 3]) / 2
    half_width = (center_x - im_rois[:, 0]) * w_scale
    half_height = (center_y - im_rois[:, 1]) * h_scale
    context_rois[:, 0] = np.max(np.hstack(
        (np.zeros([center_x.shape[0], 1]), (center_x - half_width * x_rel_pos * 2)[:, np.newaxis])
    ), axis=1)
    context_rois[:, 1] = np.max(np.hstack(
        (np.zeros([center_x.shape[0], 1]), (center_y - half_height * y_rel_pos * 2)[:, np.newaxis])
    ), axis=1)
    context_rois[:, 2] = np.min(np.hstack(
        (im_size[0] * np.ones([center_x.shape[0], 1]), (context_rois[:, 0] + half_width * 2)[:, np.newaxis])
    ), axis=1)
    context_rois[:, 3] = np.min(np.hstack(
        (im_size[1] * np.ones([center_x.shape[0], 1]), (context_rois[:, 1] + half_height * 2)[:, np.newaxis])
    ), axis=1)

    return context_rois

im_rois = np.array([[457.2160,  185.4933,  514.3680,  225.6000], [554.4747,  180.4800,  579.5413,  199.5307], [962.5600,   183.4880,  1106.9440,   229.6107]])
im_size = (1245, 376)

rois = _get_context_rois(im_size, im_rois)
print(rois)

rois = [map(int, roi) for roi in rois]
im_rois = [map(int, im_roi) for im_roi in im_rois]

im = 128 * np.ones((376, 1245, 3), dtype=np.uint8)
for roi in im_rois:
    cv2.rectangle(im, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 1)
for roi in rois:
    cv2.rectangle(im, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 255), 1)
cv2.imshow('0', im)
cv2.waitKey(0)
