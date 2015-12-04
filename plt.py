import matplotlib.pyplot as plt
import re
import sys

if __name__ == '__main__':
    
    f = open(sys.argv[1], 'r')
    data = f.read()
    f.close()
    
    iters = re.findall('Cross Entropy at iter (\d*) is', data)
    x_e = re.findall('Cross Entropy at iter \d* is (\d*.\d*)', data)
    
    iters = map(lambda x: int(x), iters)
    x_e = map(lambda x: float(x), x_e)
    
    plt.plot(iters, x_e)
    plt.show()