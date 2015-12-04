from runit import RnnLayer
import numpy as np


class RnnNet(object):
    def __init__(self, x_len, h_len, y_len, inner_activation, out_activation, sequence_len):
        self.head = RnnLayer(x_len, h_len, y_len, inner_activation, out_activation)
        self.w_xh, self.w_hh, self.w_hy = self.head.create_w()
        # link together
        prev = self.head
        self.sequence_len = sequence_len
        for i in range(self.sequence_len - 1):
            layer = RnnLayer(x_len, h_len, y_len, inner_activation, out_activation, self.w_xh, self.w_hh, self.w_hy)
            prev.set_post(layer)
            layer.set_prev(prev)
            prev = layer

    def feed_forward(self, data):
        if len(data) != self.sequence_len:
            print '[Warning] Input length not match sequence number!'
        cur_layer = self.head
        for i in range(len(data)):
            cur_layer.tiny_forward(data[i])
            cur_layer = cur_layer.get_post()

    def back_propagate(self, teacher_data, lr, back_step):
        if len(teacher_data) != self.sequence_len:
            print '[Warning] Input teacher length not match sequence number!'
        cur_layer = self.head
        for i in range(len(teacher_data)):
            cur_layer.tiny_backward(back_step, teacher_data=teacher_data[i])
            cur_layer = cur_layer.get_post()
        # finally update weights
        self.head.update_weights(lr)

    def xentroty_err(self, teacher_data):
        cur_layer = self.head
        error = 0
        i = 0
        while cur_layer is not None and i < len(teacher_data):
            error += cur_layer.xentropy(teacher_data[i])
            cur_layer = cur_layer.post
            i += 1
        return error

    def get_outputs(self):
        cur_layer = self.head
        mat = np.zeros((self.sequence_len, len(cur_layer.y)))
        i = 0
        while cur_layer is not None:
            mat[i] = cur_layer.y
            cur_layer = cur_layer.post
            i += 1
        return mat