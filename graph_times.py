import matplotlib.pyplot as plt
import numpy as np


def get_mean_time(approach, method, sc):
    # approach = 1, 2
    # method = 'numpy', 'f77', 'f90'
    # sc = 1, 2, 4...
    times = np.loadtxt('./approach_%d/tests/%s/sc-%d.txt' % (approach, method, sc))
    return np.mean(times)


def add_point_labels(x, y):
    for tup in zip(x, y):
        plt.annotate('%.2f' % tup[1], xy=tup)


def graph(approach):
    methods = ['numpy', 'f77', 'f90']
    scs     = [1, 2, 4]
    y_data  = dict(zip(methods, [[], [], []]))

    for method in methods:
        for sc in scs:
            mean_time = get_mean_time(approach, method, sc)
            y_data[method].append(mean_time)

        plt.plot(scs, y_data[method], label=method)
        add_point_labels(scs, y_data[method])

    plt.xlim([0,  scs[-1] + 0.5])
    plt.xlabel('$sc$ ($N=128sc$, grid size is $N\\times N$ )')
    plt.ylabel('Average time (s)')
    plt.title('Average time of solving the Rotating Shallow Water equations in 2D')
    plt.legend(loc='best')
    plt.savefig('./approach_%d/graphs/times.pdf' % approach)
    # plt.show()


graph(2)
