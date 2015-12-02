from Read.article import *
import rnn

import sys
import matplotlib.pyplot as plt

# python main.py filepath sequence_length
if __name__ == '__main__':
    
    # The second arg in command line should be the file path
    if len(sys.argv) < 2:
        raise ValueError('File path should be given!')
    
    f = openFile(sys.argv[1])       # Open the file according to given path
    
    # If the sequence length is given and is positive, store it in seq_length.
    # Otherwise, set seq_length as 10 by default
    if len(sys.argv) == 3 and int(sys.argv[2]) > 0:
        seq_length = int(sys.argv[2])
    else:
        seq_length = 10
    
    
    length = seq_length
    # Check input's length. If less then 10, means EOF is reached
    while length >= seq_length:
        inputs, teacher = getData(f, seq_length)
        
        #
        #
        
        # Update length by the latest input
        length = len(inputs)
        
    
    # Close file
    f.close()