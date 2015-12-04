import numpy as np
import codecs


# Transform a single char to a self-designed length binary vector
def tc2v(c, unique_char):
    v = np.zeros(len(unique_char), dtype=np.int8)
    v[unique_char[c]] = 1
    return v
    

# Self defined openfile to get rid of UTF-8 BOM
def openFile(filepath, mood = 'r'):
    f = codecs.open(filepath, mood, errors="ignore")
    return f


# Read the next 10, and next 2-11 characters in the file.
# However the 11th character in the file should be put back to the stream
def getData(f, char_map, length=10):
    inputs = f.read(length)
    teacher = inputs[1:] + f.read(1)
    f.seek(-1,1)
    
    a = ts2nda(inputs, char_map)
    b = ts2nda(teacher, char_map)
    
    return a, b

def get_char_map(f):
    unique_char = {}
    index = 0
    inputs = f.read(1000)
    while len(inputs) == 1000:
        for char in inputs:
            if char not in unique_char:
                unique_char[char] = index
                index += 1
        inputs = f.read(1000)
    return unique_char


# Transfer a string to ndarray matrix
def ts2nda(str, char_map):
    l = map(lambda c: tc2v(c, char_map), str)
    return np.asarray(l, dtype=np.int8)

def m2t(matrix, char_arr):
    return "".join([char_arr[np.random.choice(len(char_arr), p=arr)] for arr in matrix])