# get tf weights and bias
from __future__ import print_function

import os
import sys

import tensorflow as tf
sys.path.append(os.getcwd())
from tensorflow.python.platform import gfile
import numpy as np

pb_file = "./CTPN/data/ctpn.pb"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
# config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
with gfile.FastGFile(pb_file, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
sess.run(tf.global_variables_initializer())

# tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
# for tensor_name in tensor_name_list:
#     print(tensor_name, '\n')

list = ('conv1_1/weights:0', 'conv1_1/biases:0', 'conv1_2/weights:0', 'conv1_2/biases:0', 'conv2_1/weights:0',
            'conv2_1/biases:0', 'conv2_2/weights:0', 'conv2_2/biases:0', 'conv3_1/weights:0', 'conv3_1/biases:0',
            'conv3_2/weights:0', 'conv3_2/biases:0', 'conv3_3/weights:0', 'conv3_3/biases:0', 'conv4_1/weights:0',
            'conv4_1/biases:0', 'conv4_2/weights:0', 'conv4_2/biases:0', 'conv4_3/weights:0', 'conv4_3/biases:0',
            'conv5_1/weights:0', 'conv5_1/biases:0', 'conv5_2/weights:0', 'conv5_2/biases:0', 'conv5_3/weights:0',
            'conv5_3/biases:0', 'rpn_conv/3x3/weights:0', 'rpn_conv/3x3/biases:0', 'lstm_o/bidirectional_rnn/fw/lstm_cell/kernel:0',
            'lstm_o/bidirectional_rnn/fw/lstm_cell/bias:0', 'lstm_o/bidirectional_rnn/bw/lstm_cell/kernel:0',
            'lstm_o/bidirectional_rnn/bw/lstm_cell/bias:0', 'lstm_o/weights:0', 'lstm_o/biases:0', 'rpn_bbox_pred/weights:0',
            'rpn_bbox_pred/biases:0', 'rpn_cls_score/weights:0', 'rpn_cls_score/biases:0')

tf_args = {}
for i in list:
    tf_args[i] = sess.run(i)
    print(i)
    np.savez('tf_args.npz', **tf_args)