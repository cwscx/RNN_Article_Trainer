import numpy as np

class Weight(object):
    def __init__(self, in_len, out_len):
        self.w = np.random.rand(out_len, in_len) / np.sqrt(in_len)
        self.delta = np.zeros(self.w)

    def dot_prod(self, input_data):
        return np.dot(self.w, input_data)

    def accum_del(self, u_delta, z, lr):
        self.delta += np.dot(u_delta.reshape(len(u_delta), 1), z.reshape(1, len(z))) * lr

    def update(self):
        self.w += self.delta
        self.delta = np.zeros(self.w)


class RnnLayer(object):
    def __init__(self, x_len, h_len, y_len, inner_activation, out_activation, w_xh=None, w_hh=None, w_hy=None):
        self.x = np.zeros(x_len)
        self.h = np.zeros(h_len)
        self.y = np.zeros(y_len)
        self.w_xh = w_xh
        self.w_hh = w_hh
        self.w_hy = w_hy
        self.inner_activation = inner_activation
        self.out_activation = out_activation
        self.h_grad = np.zeros(h_len)
        self.prev = None
        self.post = None

    def create_w(self):
        if self.w_xh is None:
            self.w_xh = Weight(len(self.x), len(self.h))
        if self.w_hh is None:
            self.w_hh = Weight(len(self.h), len(self.h))
        if self.w_hy is None:
            self.w_hy = Weight(len(self.h), len(self.y))
        return self.w_xh, self.w_hh, self.w_hy

    def set_prev(self, rnnlayer):
        self.prev = rnnlayer

    def get_prev(self):
        return self.prev

    def set_post(self, rnnlayer):
        self.post = rnnlayer

    def get_post(self):
        return self.post

    def tiny_forward(self, input_data):
        if len(input_data) != len(self.x):
            raise ValueError('Input dimension not match x!')
        self.x = input_data
        if self.prev is not None:
            self.h += self.w_hh.dot_prod(self.prev.h)
        self.h += self.w_xh.dot_prod(input_data)
        self.h_grad = self.inner_activation.grad(self.h)
        self.h = self.inner_activation.calc(self.h)
        self.y = self.out_activation.calc(self.w_hy.dot_prod(self.h))

    def tiny_backward(self, lr, post_h_delta=None, teacher_data=None):
        if post_h_delta is None and teacher_data is None:
            raise ValueError('Cannot backprop, insufficient info given!')
        if teacher_data is not None:
            # backprop starting from this time frame
            self.w_hy.accum_del(teacher_data - self.y, self.h, lr)
            h_delta = self.w_hy.dot_prod(teacher_data - self.y) * self.h_grad
        else:
            # backprop from the next time frame
            h_delta = self.w_hh.dot_prod(post_h_delta) * self.h_grad
            self.w_hh.accum_del(post_h_delta, self.h, lr)
        self.w_xh.accum_del(h_delta, self.x, lr)
        if self.prev is not None:
            self.prev.tiny_backward(lr, post_h_delta=h_delta)

    def update_weights(self):
        self.w_xh.update()
        self.w_hh.update()
        self.w_hy.update()

