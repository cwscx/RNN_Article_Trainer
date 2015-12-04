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
    lr = 1e-3
    network = rnet.RnnNet(x_len, h_len, y_len, pxxactivation.Tanh, pxxactivation.Softmax, seq_length)
    # Check input's length. If less then 10, means EOF is reached
    i = 0
    j = 0
    while True:
        inputs, teacher = getData(f, char_map, seq_length)
        if len(inputs) < seq_length:
            print 'another epoch'
            f = openFile(sys.argv[1])
            j += 1
            continue
        network.feed_forward(inputs)
        print 'Cross Entropy at iter {} epoch {} is {}'.format(i, j, network.xentroty_err(teacher))
        print 'Predicted text is {}'.format(m2t(network.get_outputs(), char_arr))
        network.back_propagate(teacher, lr, 1)
        # Update length by the latest input
        i += 1

    # Close file
    f.close()