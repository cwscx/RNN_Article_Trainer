from Read.article import *
from rnn import rnet
from rnn import runit
from pxxnet import pxxactivation
import sys
import matplotlib.pyplot as plt

def sample(h, seed_ix, n, input_size, Wxh, Whh, Why, bh, by, temp=1):
  """
  sample a sequence of integers from the model
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((input_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h).reshape(len(h), 1) + bh.reshape(len(bh), 1))
    y = np.dot(Why, h) + by
    y /= temp
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(input_size), p=p.ravel())
    x = np.zeros((input_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

# python main.py filepath sequence_length
if __name__ == '__main__':
    
    # The second arg in command line should be the file path
    if len(sys.argv) < 2:
        # raise ValueError('File path should be given!')
        sys.argv.append('holmes1')
    
    f = openFile(sys.argv[1])       # Open the file according to given path
    char_map = get_char_map(f)
    f.close()
    f = openFile(sys.argv[1])
    char_arr = ['a'] * len(char_map)
    for k in char_map:
        char_arr[char_map[k]] = k
    
    # If the sequence length is given and is positive, store it in seq_length.
    # Otherwise, set seq_length as 10 by default
    if len(sys.argv) == 3 and int(sys.argv[2]) > 0:
        seq_length = int(sys.argv[2])
    else:
        seq_length = 25
    
    
    length = seq_length
    x_len = y_len = len(char_arr)
    h_len = 100
    lr = 0.2
    network = rnet.RnnNet(x_len, h_len, y_len, pxxactivation.Tanh, pxxactivation.Softmax, seq_length)
    # Check input's length. If less then 10, means EOF is reached
    i = 1
    j = 0
    smooth_loss = -np.log(1.0/len(char_arr))*seq_length
    while True:
        inputs, teacher = getData(f, char_map, seq_length)
        if len(inputs) < seq_length:
            print 'another epoch'
            f = openFile(sys.argv[1])
            j += 1
            continue

        if i % 100 == 0:
            print 'Cross Entropy at iter {} epoch {} is {}'.format(i, j, smooth_loss)
            # result = sample(network.end.h, inputs[0], 200, len(char_arr), network.head.w_xh.w, network.head.w_hh.w,
            #                 network.head.w_hy.w, network.head.w_bh.w.reshape(len(network.head.w_bh.w), 1),
            #                 network.head.w_by.w.reshape(len(network.head.w_by.w), 1))
            # txt = ''.join(char_arr[ix] for ix in result)
            # print '----\n %s \n----' % (txt, )
        # print 'Predicted text is {}'.format(network.sample(inputs[0], char_arr, char_map))
        network.feed_forward(inputs)
        smooth_loss = smooth_loss * 0.999 + network.xentroty_err(teacher) * 0.001
        network.back_propagate(teacher, lr)
        # Update length by the latest input
        i += 1

    # Close file
    f.close()