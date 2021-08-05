from os import path

import cv2
import numpy as np
import scipy.ndimage as ndimage

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import set_session

from ggcnn_timeit import TimeIt

MODEL_FILE = 'ggcnn_model/epoch_29_model.hdf5'
sess = tf.Session()
set_session(sess)
graph = tf.compat.v1.get_default_graph()
model = load_model(path.join(path.dirname(__file__), MODEL_FILE))


TimeIt.print_output = False  # For debugging/timing


def inpaint_depth_image(depth, out_size=300, return_mask=False):
    with TimeIt('Inpainting_Processing'):
        depth_bordered = cv2.copyMakeBorder(depth, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        depth_nan_mask = np.isnan(depth_bordered).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        depth_nan_mask = cv2.dilate(depth_nan_mask, kernel, iterations=1)

        depth_bordered[depth_nan_mask == 1] = 0

        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        depth_scale = np.abs(depth_bordered).max()
        depth_bordered = depth_bordered.astype(np.float32) / depth_scale  # Has to be float32, 64 not supported.

        with TimeIt('Inpainting'):
            depth_inpainted = cv2.inpaint(depth_bordered, depth_nan_mask, 1, cv2.INPAINT_NS)

        # Back to original size and value range.
        depth_inpainted = depth_inpainted[1:-1, 1:-1]
        depth_inpainted = depth_inpainted * depth_scale

    with TimeIt('Resizing'):
        # Resize
        depth_inpainted = cv2.resize(depth_inpainted, (out_size, out_size), cv2.INTER_AREA)

    if return_mask:
        with TimeIt('Return Mask'):
            depth_nan_mask = depth_nan_mask[1:-1, 1:-1]
            depth_nan_mask = cv2.resize(depth_nan_mask, (out_size, out_size), cv2.INTER_NEAREST)
        return depth_inpainted, depth_nan_mask
    else:
        return depth_inpainted


def predict(depth, inpaint_depth=True, out_size=300, depth_nan_mask=None, filters=(2.0, 1.0, 1.0)):
    global graph, sess
    if inpaint_depth:
        depth, depth_nan_mask = inpaint_depth_image(depth, out_size, True)

    # Inference
    depth = np.clip((depth - depth.mean()), -1, 1)
    set_session(sess)
    with graph.as_default():
        pred_out = model.predict(depth.reshape((1, 300, 300, 1)))

    points_out = pred_out[0].squeeze()
    points_out[depth_nan_mask] = 0

    # Calculate the angle map.
    cos_out = pred_out[1].squeeze()
    sin_out = pred_out[2].squeeze()
    ang_out = np.arctan2(sin_out, cos_out) / 2.0

    width_out = pred_out[3].squeeze() * 150.0  # Scaled 0-150:0-1

    # Filter the outputs.
    if filters[0]:
        points_out = ndimage.filters.gaussian_filter(points_out, filters[0])  # 3.0
    if filters[1]:
        ang_out = ndimage.filters.gaussian_filter(ang_out, filters[1])
    if filters[2]:
        width_out = ndimage.filters.gaussian_filter(width_out, filters[2])

    points_out = np.clip(points_out, 0.0, 1.0-1e-3)

    return points_out, ang_out, width_out, depth
