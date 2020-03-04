import tensorrt as trt
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import os
cuda.Device(0).make_context()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

b_fp16 = True
max_time_step = 75
logger = trt.Logger(trt.Logger.INFO)


class CTPN:
    def __init__(self, b_fp16):
        self.b_fp16 = b_fp16

        trt_file_path = "ctpn_1.trt"
        # if os.path.isfile(trt_file_path):
        #     with open(trt_file_path, 'rb') as f:
        #         engine_str = f.read()
        # else:
        if True:
            engine_str = self.build_engine_str(b_fp16)
            print("create new trt engine: ", trt_file_path)
            with open(trt_file_path, 'wb') as f:
                f.write(engine_str)

        with trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_str)
        self.context = self.engine.create_execution_context()

    def __del__(self):
        self.context = None
        self.engine = None

    def create_engine(self, builder, b_fp16):
        params = np.load('tf_args.npz')
        bag = []
        network = builder.create_network()
        # data = network.add_input("data", trt.DataType.FLOAT, (1, 3, 600, 1200))
        data = network.add_input("data", trt.DataType.FLOAT, (1, 600, 1200, 3))

        data0 = network.add_shuffle(data)
        data0.first_transpose = (0, 3, 1, 2)
        print('data0 shape', data0.get_output(0).shape)

        # VGG16....
        w = params['conv1_1/weights:0'].transpose((3, 2, 0, 1)).reshape(-1)
        b = params['conv1_1/biases:0']
        bag += [w, b]
        conv1_1 = network.add_convolution(data0.get_output(0), 64, (3, 3), w, b)
        conv1_1.stride = (1, 1)
        conv1_1.padding = (1, 1)
        print('conv1_1', conv1_1.get_output(0).shape)
        actv1_1 = network.add_activation(conv1_1.get_output(0), trt.ActivationType.RELU)

        w = params['conv1_2/weights:0'].transpose((3, 2, 0, 1)).reshape(-1)
        b = params['conv1_2/biases:0']
        bag += [w, b]
        conv1_2 = network.add_convolution(actv1_1.get_output(0), 64, (3, 3), w, b)
        conv1_2.stride = (1, 1)
        conv1_2.padding = (1, 1)
        actv1_2 = network.add_activation(conv1_2.get_output(0), trt.ActivationType.RELU)
        print('conv1_2', conv1_2.get_output(0).shape)

        pool1 = network.add_pooling(actv1_2.get_output(0), trt.PoolingType.MAX, (2, 2))
        pool1.stride = (2, 2)

        ##################################################
        w = params['conv2_1/weights:0'].transpose((3, 2, 0, 1)).reshape(-1)
        b = params['conv2_1/biases:0']
        bag += [w, b]
        conv2_1 = network.add_convolution(pool1.get_output(0), 128, (3, 3), w, b)
        conv2_1.stride = (1, 1)
        conv2_1.padding = (1, 1)
        print('conv2_1', conv2_1.get_output(0).shape)
        actv2_1 = network.add_activation(conv2_1.get_output(0), trt.ActivationType.RELU)

        w = params['conv2_2/weights:0'].transpose((3, 2, 0, 1)).reshape(-1)
        b = params['conv2_2/biases:0']
        bag += [w, b]
        conv2_2 = network.add_convolution(actv2_1.get_output(0), 128, (3, 3), w, b)
        conv2_2.stride = (1, 1)
        conv2_2.padding = (1, 1)
        print('conv2_2', conv2_2.get_output(0).shape)
        actv2_2 = network.add_activation(conv2_2.get_output(0), trt.ActivationType.RELU)

        pool2 = network.add_pooling(actv2_2.get_output(0), trt.PoolingType.MAX, (2, 2))
        pool2.stride = (2, 2)

        #######################################################
        w = params['conv3_1/weights:0'].transpose((3, 2, 0, 1)).reshape(-1)
        b = params['conv3_1/biases:0']
        bag += [w, b]
        conv3_1 = network.add_convolution(pool2.get_output(0), 256, (3, 3), w, b)
        conv3_1.stride = (1, 1)
        conv3_1.padding = (1, 1)
        print('conv3_1', conv3_1.get_output(0).shape)
        actv3_1 = network.add_activation(conv3_1.get_output(0), trt.ActivationType.RELU)

        w = params['conv3_2/weights:0'].transpose((3, 2, 0, 1)).reshape(-1)
        b = params['conv3_2/biases:0']
        bag += [w, b]
        conv3_2 = network.add_convolution(actv3_1.get_output(0), 256, (3, 3), w, b)
        conv3_2.stride = (1, 1)
        conv3_2.padding = (1, 1)
        print('conv3_2', conv3_2.get_output(0).shape)
        actv3_2 = network.add_activation(conv3_2.get_output(0), trt.ActivationType.RELU)

        w = params['conv3_3/weights:0'].transpose((3, 2, 0, 1)).reshape(-1)
        b = params['conv3_3/biases:0']
        bag += [w, b]
        conv3_3 = network.add_convolution(actv3_2.get_output(0), 256, (3, 3), w, b)
        conv3_3.stride = (1, 1)
        conv3_3.padding = (1, 1)
        print('conv3_3', conv3_3.get_output(0).shape)
        actv3_3 = network.add_activation(conv3_3.get_output(0), trt.ActivationType.RELU)

        pool3 = network.add_pooling(actv3_3.get_output(0), trt.PoolingType.MAX, (2, 2))
        pool3.stride = (2, 2)

        ####################################################
        w = params['conv4_1/weights:0'].transpose((3, 2, 0, 1)).reshape(-1)
        b = params['conv4_1/biases:0']
        bag += [w, b]
        conv4_1 = network.add_convolution(pool3.get_output(0), 512, (3, 3), w, b)
        conv4_1.stride = (1, 1)
        conv4_1.padding = (1, 1)
        print('conv4_1', conv4_1.get_output(0).shape)
        actv4_1 = network.add_activation(conv4_1.get_output(0), trt.ActivationType.RELU)

        w = params['conv4_2/weights:0'].transpose((3, 2, 0, 1)).reshape(-1)
        b = params['conv4_2/biases:0']
        bag += [w, b]
        conv4_2 = network.add_convolution(actv4_1.get_output(0), 512, (3, 3), w, b)
        conv4_2.stride = (1, 1)
        conv4_2.padding = (1, 1)
        print('conv4_2', conv4_2.get_output(0).shape)
        actv4_2 = network.add_activation(conv4_2.get_output(0), trt.ActivationType.RELU)

        w = params['conv4_3/weights:0'].transpose((3, 2, 0, 1)).reshape(-1)
        b = params['conv4_3/biases:0']
        bag += [w, b]
        conv4_3 = network.add_convolution(actv4_2.get_output(0), 512, (3, 3), w, b)
        conv4_3.stride = (1, 1)
        conv4_3.padding = (1, 1)
        print('conv4_3', conv4_3.get_output(0).shape)
        actv4_3 = network.add_activation(conv4_3.get_output(0), trt.ActivationType.RELU)

        pool4 = network.add_pooling(actv4_3.get_output(0), trt.PoolingType.MAX, (2, 2))
        pool4.stride = (2, 2)

        ###################################################################
        w = params['conv5_1/weights:0'].transpose((3, 2, 0, 1)).reshape(-1)
        b = params['conv5_1/biases:0']
        bag += [w, b]
        conv5_1 = network.add_convolution(pool4.get_output(0), 512, (3, 3), w, b)
        conv5_1.stride = (1, 1)
        conv5_1.padding = (1, 1)
        print('conv5_1', conv5_1.get_output(0).shape)
        actv5_1 = network.add_activation(conv5_1.get_output(0), trt.ActivationType.RELU)

        w = params['conv5_2/weights:0'].transpose((3, 2, 0, 1)).reshape(-1)
        b = params['conv5_2/biases:0']
        bag += [w, b]
        conv5_2 = network.add_convolution(actv5_1.get_output(0), 512, (3, 3), w, b)
        conv5_2.stride = (1, 1)
        conv5_2.padding = (1, 1)
        print('conv5_2', conv5_2.get_output(0).shape)
        actv5_2 = network.add_activation(conv5_2.get_output(0), trt.ActivationType.RELU)

        w = params['conv5_3/weights:0'].transpose((3, 2, 0, 1)).reshape(-1)
        b = params['conv5_3/biases:0']
        bag += [w, b]
        conv5_3 = network.add_convolution(actv5_2.get_output(0), 512, (3, 3), w, b)
        conv5_3.stride = (1, 1)
        conv5_3.padding = (1, 1)
        print('conv5_3', conv5_3.get_output(0).shape)
        actv5_3 = network.add_activation(conv5_3.get_output(0), trt.ActivationType.RELU)

        # RPN #######################################
        w = params['rpn_conv/3x3/weights:0'].transpose((3, 2, 0, 1)).reshape(-1)
        b = params['rpn_conv/3x3/biases:0']
        bag += [w, b]
        convRPN = network.add_convolution(actv5_3.get_output(0), 512, (3, 3), w, b)
        convRPN.stride = (1, 1)
        convRPN.padding = (1, 1)
        print('convRPN', convRPN.get_output(0).shape)
        actvRPN = network.add_activation(convRPN.get_output(0), trt.ActivationType.RELU)

        shuf0 = network.add_shuffle(actvRPN.get_output(0))
        shuf0.first_transpose = (0, 2, 3, 1)
        shuf0.reshape_dims = (37, -1, 512)
        print('shuf0', shuf0.get_output(0).shape)

        # RPN-BiLSTM
        n_input = 512
        n_hidden = 128
        lstm = network.add_rnn_v2(shuf0.get_output(0), 1, n_hidden, max_time_step, trt.RNNOperation.LSTM)
        lstm.direction = trt.RNNDirection.BIDIRECTION
        zero_bias = np.zeros(n_hidden, dtype=np.float32)
        for i in range(4):
            layer = i // 2  # 0 ,0 , 1 ,1
            iW = i % 2
            isW = True if iW == 0 else False
            param_name = 'lstm_o/bidirectional_rnn/{}/lstm_cell/'.format("fw" if layer % 2 == 0 else "bw")
            all_w = [w.transpose((1, 0)).reshape(-1) for w in
                    np.split(np.split(params[param_name + 'kernel:0'], [n_input])[iW], 4, axis=1)]  # (640, 512)
            all_b = [w if isW else zero_bias for w in np.split(params[param_name + 'bias:0'], 4)]  # (512,)
            for t, w, b in zip(
                    [trt.RNNGateType.INPUT, trt.RNNGateType.CELL, trt.RNNGateType.FORGET, trt.RNNGateType.OUTPUT],
                    all_w, all_b):
                lstm.set_weights_for_gate(layer, t, isW, w)
                lstm.set_bias_for_gate(layer, t, isW, b)
        print('lstm', lstm.get_output(0).shape)

        shuf1 = network.add_shuffle(lstm.get_output(0))
        shuf1.reshape_dims = (37*75, 1, 1, 256)
        print("shuf1", shuf1.get_output(0).shape)

        # (256, 512)
        w = params['lstm_o/weights:0'].transpose((1, 0)).reshape(-1)  # (out_c, in_c, h, w)
        b = params['lstm_o/biases:0'].reshape(-1)
        bag += [w, b]
        fc0 = network.add_fully_connected(shuf1.get_output(0), 512, w, b)
        print('fc0', fc0.get_output(0).shape)  # (2775, 512, 1, 1)

        shuf2 = network.add_shuffle(fc0.get_output(0))
        shuf2.reshape_dims = (37*75, 1, 1, 512)

        w = params['rpn_bbox_pred/weights:0'].transpose((1, 0)).reshape(-1)
        b = params['rpn_bbox_pred/biases:0'].reshape(-1)
        bag += [w, b]
        fc1 = network.add_fully_connected(shuf2.get_output(0), 40, w, b)
        print('fc1', fc1.get_output(0).shape)  # fc1 (2775, 40, 1, 1)

        w = params['rpn_cls_score/weights:0'].transpose((1, 0)).reshape(-1)
        b = params['rpn_cls_score/biases:0'].reshape(-1)
        bag += [w, b]
        fc2 = network.add_fully_connected(shuf2.get_output(0), 20, w, b)
        print('fc2', fc2.get_output(0).shape)  # fc2 (2775, 20, 1, 1)

        reshape1 = network.add_shuffle(fc2.get_output(0))
        # reshape1.reshape_dims = (1, 37, -1, 2)
        reshape1.first_transpose = (0, 3, 1, 2)
        reshape1.reshape_dims = (2, -1)
        print('reshape1: ', reshape1.get_output(0).shape)

        # softmax
        Isoftmax = network.add_softmax(reshape1.get_output(0))
        print('Isoftmax', Isoftmax.get_output(0).shape)

        reshape2 = network.add_shuffle(Isoftmax.get_output(0))
        # # shape is (1, H, WxA, 2) -> (1, H, W, Ax2)
        # reshape2.first_transpose = (1, 0)
        # reshape2.reshape_dims = (2, 27750)
        reshape2.reshape_dims = (1, 37, 75, 20)
        print('reshape2: ', reshape2.get_output(0).shape)

        network.mark_output(shuf0.get_output(0))
        network.mark_output(lstm.get_output(0))

        builder.max_workspace_size = 2 << 30
        builder.fp16_mode = b_fp16
        return builder.build_cuda_engine(network)

    def build_engine_str(self, b_fp16):
        with trt.Builder(logger) as builder, self.create_engine(builder, b_fp16) as engine:
            return engine.serialize()

    def infer(self, batch_size, d_input, d_ctpn_box_pred, d_ctpn_cls_prob, ):
        bindings = [int(d_input), int(d_ctpn_box_pred), int(d_ctpn_cls_prob)]
        self.context.execute_async(batch_size, bindings, 0)

cuda.Context.pop()
