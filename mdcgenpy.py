from __future__ import print_function
from mdcgenutils import initialize
from clusters import ClusterGenerator


if __name__ == '__main__':
    p = ClusterGenerator()
    # exec(open('config.py').read())

    initialize(p)
    data = p.generate_data()
    data = next(data)
    print(data)
    import matplotlib.pyplot as plt
    plt.scatter(data[0][:,0], data[0][:,1], c=data[1])
    plt.show()
