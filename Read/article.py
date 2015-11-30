import numpy as np
    
# Transform a single char to a 256 length binary vector 
def tc2v(c):
    v = np.zeros(256, dtype = np.int8)
    v[ord(c)] = 1
    return v