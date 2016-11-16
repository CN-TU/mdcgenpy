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
    # plt.scatter(data[0][:, 0], data[0][:, 1], c=data[1])
    # plt.show()
    plts = []
    from matplotlib.colors import cnames
    # colors = list(cnames.keys())
    import itertools
    colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    for lab in range(-1, p.n_clusters):
        indexes = data[1] == lab
        plts.append(plt.scatter(data[0][indexes,0], data[0][indexes,1], color=next(colors)))  #, color=colors[hash(lab) % len(colors)])) # , c=data[1]))
    plt.legend(plts, (str(l) for l in range(-1, p.n_clusters)))
    plt.show()
