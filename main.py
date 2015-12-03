from Read.article import *
from rnn import rnet
from rnn import runit
from pxxnet import pxxactivation
import sys
import matplotlib.pyplot as plt

# python main.py filepath sequence_length
if __name__ == '__main__':
    
    # The second arg in command line should be the file path
    if len(sys.argv) < 2:
        # raise ValueError('File path should be given!')
        sys.argv.append('holmes')
    
    f = openFile(sys.argv[1])       # Open the file according to given path
    
    # If the sequence length is given and is positive, store it in seq_length.
    # Otherwise, set seq_length as 10 by default
    if len(sys.argv) == 3 and int(sys.argv[2]) > 0:
        seq_length = int(sys.argv[2])
    else:
        seq_length = 100
    
    
    length = seq_length
    x_len = y_len = 256
    h_len = 256
    lr = 1e-3
    network = rnet.RnnNet(x_len, h_len, y_len, pxxactivation.Tanh, pxxactivation.Softmax, seq_length)
    # Check input's length. If less then 10, means EOF is reached
    i = 0
    while length >= seq_length:
        inputs, teacher = getData(f, seq_length)
        network.feed_forward(inputs)
        print 'Cross Entropy at iter {} is {}'.format(i, network.xentroty_err(teacher))
        network.back_propagate(teacher, lr)
        # Update length by the latest input
        length = len(inputs)
        i += 1

    # Close file
    f.close()