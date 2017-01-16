from __future__ import print_function
import sys
import numpy as np
from mdcgenpy.interface import json as js


if (sys.version_info > (3, 0)):
    stdout = sys.stdout.buffer
else:
    stdout = sys.stdout


if __name__ == '__main__':
    p = js.get_cluster_generator(sys.argv[1])

    try:
        batch_size = int(sys.argv[2])
    except IndexError:
        batch_size = 0
    data = p.generate_data(batch_size=batch_size)
    if batch_size == 0:  # received data, and not a generator
        np.savetxt(stdout, np.hstack(data), delimiter=',')
    else:
        for d in data:
            np.savetxt(stdout, np.hstack(d), delimiter=',')
