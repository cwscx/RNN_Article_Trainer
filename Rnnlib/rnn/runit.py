import numpy as np

class Weight(object):
    def __init__(self, in_len, out_len):
        self.w = np.random.rand(out_len, in_len) / in_len
        self.delta = np.zeros((out_len, in_len))
        self.historical_delta = 0

    def dot_prod(self, input_data, t=False):
        if t:
            return np.dot(self.w.transpose(), input_data)
        else:
            return np.dot(self.w, input_data)

    def accum_del(self, u_delta, z):
        self.delta += np.dot(u_delta.reshape(len(u_delta), 1), z.reshape(1, len(z)))

    def update(self, lr):
        # print np.max(self.delta)
        # print np.min(self.delta)
        self.historical_delta += self.delta ** 2
        adjusted_delta = self.delta / (1e-6 + np.sqrt(self.historical_delta))
        self.w += adjusted_delta * lr
        self.delta = np.zeros((len(self.w), len(self.w[0])))


class RnnLayer(object):
    def __init__(self, x_len, h_len, y_len, inner_activation, out_activation, w_xh=None, w_hh=None, w_hy=None):
        self.x = np.zeros(x_len)        # Input
        self.h = np.zeros(h_len)        # Store f(h_t)
        self.y = np.zeros(y_len)        # Output
        self.w_xh = w_xh
        self.w_hh = w_hh
        self.w_hy = w_hy
        self.inner_activation = inner_activation
        self.out_activation = out_activation
        self.h_grad = np.zeros(h_len)
        self.prev = None                # t-1
        self.post = None                # t+1

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
        
        # h = W_hh * f(h_t-1) + W_hx * x
        self.h += self.w_xh.dot_prod(input_data)
        if self.prev is not None:
            self.h += self.w_hh.dot_prod(self.prev.h)
        
        # y = W_hy * f(h_t)     // Before activated    
        self.h_grad = self.inner_activation.grad(self.h)
        self.h = self.inner_activation.calc(self.h)     # h = f(h_t)
        self.y = self.out_activation.calc(self.w_hy.dot_prod(self.h))

    def tiny_backward(self, backward_times, post_h_delta=None, teacher_data=None):
        if backward_times <= 0:
            return
        if post_h_delta is None and teacher_data is None:
            raise ValueError('Cannot backprop, insufficient info given!')
        if teacher_data is not None:
            # backprop starting from this time frame
            self.w_hy.accum_del(teacher_data - self.y, self.h)
            h_delta = self.w_hy.dot_prod(teacher_data - self.y, True) * self.h_grad
        else:
            # backprop from the next time frame
            h_delta = self.w_hh.dot_prod(post_h_delta) * self.h_grad
            self.w_hh.accum_del(post_h_delta, self.h)
        self.w_xh.accum_del(h_delta, self.x)
        if self.prev is not None:
            self.prev.tiny_backward(backward_times-1, post_h_delta=h_delta)

    def update_weights(self, lr):
        self.w_xh.update(lr)
        self.w_hh.update(lr)
        self.w_hy.update(lr)

    def xentropy(self, teacher_vector):
        return -np.dot(teacher_vector, np.log(self.y))
