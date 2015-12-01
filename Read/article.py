import numpy as np
import codecs


# Transform a single char to a 256 length binary vector 
def tc2v(c):
    v = np.zeros(256, dtype = np.int8)
    v[ord(c)] = 1
    return v
    

# Self defined openfile to get rid of UTF-8 BOM
def openFile(filepath, mood = 'r'):
    f = codecs.open(filepath, mood, "utf-8-sig")
    return f


# Read the next 10, and next 2-11 characters in the file.
# However the 11th character in the file should be put back to the stream
def getData(f, length = 10):
    inputs = f.read(length)
    teacher = inputs[1:] + f.read(1)
    f.seek(-1,1)
    
    return inputs, teacher
    