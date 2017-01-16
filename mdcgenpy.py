from __future__ import print_function
import time
import itertools
import matplotlib.pyplot as plt
from mdcgenpy.clusters import ClusterGenerator


if __name__ == '__main__':
    p = ClusterGenerator(seed=100,
                         n_samples=3000,
                         n_feats=2,
                         k=20,
                         min_samples=0,
                         distributions='gap',
                         dflag=False,  # not implemented
                         mv=True,
                         corr=0.,
                         compactness_factor=0.1,
                         alpha_n=1,
                         scale=True,
                         outliers=50,
                         rotate=True,
                         add_noise=0,  # not implemented
                         n_noise=[],
                         ki_coeff=3.
                         )

    tic = time.time()
    cputic = time.process_time()
    data = p.generate_data(batch_size=0)
    print('Time to compute clusters: Real Time:',
          time.time() - tic,
          '; CPU Time:',
          time.process_time() - cputic)
    print(data)
    plts = []
    colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    for lab in range(-1, p.n_clusters):
        indexes = data[1] == lab
        plts.append(plt.scatter(data[0][indexes,0], data[0][indexes,1], color=next(colors)))
    # plt.legend(plts, (str(l) for l in range(-1, p.n_clusters)))
    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)
    plt.show()
