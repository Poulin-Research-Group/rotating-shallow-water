import matplotlib.pyplot as plt
import numpy as np


def get_mean_time(approach, method, sc, opt=None):
    # approach = 1, 2
    # method = 'numpy', 'f77', 'f90'
    # sc = 1, 2, 4...
    if opt is None:
        filename = './approach_%d/tests/%s/sc-%d.txt' % (approach, method, sc)
    else:
        filename = './approach_%d/tests/%s/%s/sc-%d.txt' % (approach, method, opt, sc)
    times = np.loadtxt(filename)
    return np.mean(times)


def add_point_labels(x, y):
    for tup in zip(x, y):
        plt.annotate('%.2f' % tup[1], xy=tup)


def graph(approach, scs, opt):
    methods = ['numpy', 'f2py-f77', 'f2py-f90', 'hybrid77', 'hybrid90', 'f77']
    methodsFortran = methods[1:]
    y_data  = dict(zip(methods, [[] for i in xrange(len(methods))]))

    # handle numpy data
    for sc in scs:
        mean_time = get_mean_time(approach, 'numpy', sc)
        y_data['numpy'].append(mean_time)

    # handle Fortran data
    for method in methodsFortran:
        for sc in scs:
            mean_time = get_mean_time(approach, method, sc, opt)
            y_data[method].append(mean_time)

        plt.plot(scs, y_data[method], label=method)
        add_point_labels(scs, y_data[method])

    plt.xlim([0,  scs[-1] + 0.5])
    plt.xlabel('$sc$ ($N=128sc$, grid size is $N\\times N$ )')
    plt.ylabel('Average time (s)')
    plt.title('Average time of solving the Rotating Shallow Water equations in 2D\nCompiled using %s' % opt)
    plt.legend(loc='best')
    plt.savefig('./approach_%d/graphs/times_%s.pdf' % (approach, opt))
    plt.clf()


scs = [1, 2]
graph(2, scs, 'O0')
graph(2, scs, 'O3')
graph(2, scs, 'Ofast')
