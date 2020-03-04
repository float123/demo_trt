from ctpnport import *
from CTPN.lib.fast_rcnn.test import _get_blobs
from CTPN.lib.rpn_msr.proposal_layer_tf import proposal_layer
# from CTPN.lib.text_connector.text_connect_cfg import Config as TextLineCfg
from CTPN.lib.utils.timer import Timer
import os
import cv2
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEMO_IMAGE_DIR = "./img"
RESULT_IMAGE_DIR = "./result/"
text_detector = ctpnSource()
timer = Timer()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.20
# config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

with gfile.FastGFile('./CTPN/data/ctpn.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
sess.run(tf.global_variables_initializer())

input_img = sess.graph.get_tensor_by_name('Placeholder:0')
output_box_pred = sess.graph.get_tensor_by_name('rpn_conv/3x3/rpn_conv/3x3:0')  # The last layer of convolution
output_cls_prob = sess.graph.get_tensor_by_name('lstm_o/concat:0')  # bilstm output
# output_box_pred = sess.graph.get_tensor_by_name('rpn_bbox_pred/Reshape_1:0')
# output_cls_prob = sess.graph.get_tensor_by_name('Reshape_2:0')

for im_name in os.listdir(DEMO_IMAGE_DIR):
    if os.path.isdir(os.path.join(DEMO_IMAGE_DIR, im_name)):
        continue
    result_path = RESULT_IMAGE_DIR + im_name
    im = cv2.imread(os.path.join(DEMO_IMAGE_DIR, im_name))
    if im is None:
        continue
    timer.tic()
    # img, scale = resize_im(im, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    img = cv2.resize(im, (1200, 600))
    print('img shape', img.shape)
    blobs, im_scales = _get_blobs(img, None)
    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32)

    cls_prob, box_pred = sess.run([output_cls_prob, output_box_pred], feed_dict={input_img: blobs['data']})
    print('conv shape ', box_pred.size)
    print('conv\n', box_pred)
    print('lstm.shape\n', cls_prob.shape)
    print('lstm\n', cls_prob)

    # rois, _ = proposal_layer(cls_prob, box_pred, blobs['im_info'], 'TEST', anchor_scales=cfg.ANCHOR_SCALES)
    # text_recs = getCharBlock(text_detector, img, rois, im_scales)

    print("Time: %f" % timer.toc())
