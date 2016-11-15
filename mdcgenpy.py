from __future__ import print_function
from mdcgenutils import initialize
from clusters import DataConfig


if __name__ == '__main__':
    p = DataConfig()
    # exec(open('config.py').read())

    initialize(p)
    data = p.generate_data()
    data = next(data)
    print(data)
