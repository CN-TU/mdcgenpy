from __future__ import print_function
from .utils import initialize


if __name__ == '__main__':
    p = None
    exec(open('config.py').read())

    initialize(p)
    
