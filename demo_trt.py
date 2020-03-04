from ctpnport_trt import *
from ctpn_static import *
from CTPN.lib.fast_rcnn.test import _get_blobs
from CTPN.lib.rpn_msr.proposal_layer_tf import proposal_layer
# from CTPN.lib.text_connector.text_connect_cfg import Config as TextLineCfg
from CTPN.lib.utils.timer import Timer
import os
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cuda.Device(0).make_context()
text_detector = ctpnSource()
timer = Timer()

DEMO_IMAGE_DIR = "./img/"
RESULT_IMAGE_DIR = "./result_trt/"
ctpn_model = CTPN(b_fp16)

for im_name in os.listdir(DEMO_IMAGE_DIR):
    if os.path.isdir(os.path.join(DEMO_IMAGE_DIR, im_name)):
        continue
    result_path = RESULT_IMAGE_DIR + im_name
    im = cv2.imread(os.path.join(DEMO_IMAGE_DIR, im_name))
    if im is None:
        continue
    timer.tic()
    img = cv2.resize(im, (1200, 600))
    blobs, im_scales = _get_blobs(img, None)

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32)

    box_pred, cls_prob = predict_ctpn(blobs['data'], ctpn_model)
    box_pred = box_pred.reshape(1, 37, 75, 512)
    print('conv.shape', box_pred.shape)
    print('conv\n', box_pred)
    print('lstm.shape', cls_prob.shape)
    print('lstm\n', cls_prob)
    # rois, _ = proposal_layer(cls_prob, box_pred, blobs['im_info'], 'TEST', anchor_scales=cfg.ANCHOR_SCALES)
    # text_recs = getCharBlock(text_detector, img, rois, im_scales)
    print("Time: %f" % timer.toc())

cuda.Context.pop()
