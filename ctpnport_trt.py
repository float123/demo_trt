from __future__ import print_function

import glob
import os
import shutil
import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.platform import gfile
import pycuda.autoinit
import pycuda.driver as cuda


def init():
    sys.path.insert(0, "./CTPN/")


init()

sys.path.append(os.getcwd())
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.text_connector.detectors import TextDetector

cfg_from_file('./CTPN/ctpn/text.yml')


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


def draw_boxes(boxes):
    for box in boxes:
        for i in range(8):
            box[i] = int(box[i])
            if box[i] < 0:
                box[i] = 0
    return boxes


def draw_boxes1(img, image_name, boxes):
    base_name = image_name.split('/')[-1]
    with open('result/' + 'res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:
        for box in boxes:
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
            cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

            min_x = min(int(box[0]), int(box[2]), int(box[4]), int(box[6]))
            min_y = min(int(box[1]), int(box[3]), int(box[5]), int(box[7]))
            max_x = max(int(box[0]), int(box[2]), int(box[4]), int(box[6]))
            max_y = max(int(box[1]), int(box[3]), int(box[5]), int(box[7]))

            line = ','.join([str(min_x), str(min_y), str(max_x), str(max_y)]) + '\r\n'
            f.write(line)

    cv2.imwrite(os.path.join("result/", base_name), img)
    return boxes


def ctpnSource():
    text_detector = TextDetector()
    return text_detector


def getCharBlock(text_detector, img, rois, im_scales):
    scores = rois[:, 0]
    boxes = rois[:, 1:5] / im_scales[0]
    boxes = text_detector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    text_recs = draw_boxes(boxes)
    return text_recs


def getCharBlock1(text_detector, img, rois, im_scales, image_path):
    scores = rois[:, 0]
    boxes = rois[:, 1:5] / im_scales[0]
    boxes = text_detector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    text_recs = draw_boxes1(img, image_path, boxes)
    return text_recs


def predict_ctpn(data, ctpn_model):
    print('data shape: ', data.shape)
    data = np.array(data, np.float32)
    # data = np.transpose(data, (0, 3, 1, 2))
    # data = np.reshape(data, [1, 3, 600, 1200])

    d_input = cuda.mem_alloc(data.nbytes)
    cuda.memcpy_htod(d_input, data)

    d_ctpn_box_pred = cuda.mem_alloc(37 * 75 * 512 * 4)
    d_ctpn_cls_prob = cuda.mem_alloc(37 * 75 * 256 * 4)
    box_pred = np.empty((37, 75, 512), dtype=np.float32)
    cls_prob = np.empty((37, 75, 256), dtype=np.float32)
    # box_pred = np.empty((37*75, 40, 1, 1), dtype=np.float32)
    # cls_prob = np.empty((1, 37, 75, 20), dtype=np.float32)

    # batch_size
    ctpn_model.infer(1, d_input, d_ctpn_box_pred, d_ctpn_cls_prob)

    cuda.Context.synchronize()

    cuda.memcpy_dtoh(box_pred, d_ctpn_box_pred)
    cuda.memcpy_dtoh(cls_prob, d_ctpn_cls_prob)

    return box_pred, cls_prob


