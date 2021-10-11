import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def particleDistribution(file):

    colors = ['red', 'darkgreen', 'blue', 'black']

    f = h5py.File(file, 'r')
    print(list(f.keys()))

    ranges = list(f['hilbertRanges'].value)
    keys = np.array(list(f['hilbertKey'].value))
    indices = keys.argsort()
    keys = keys[indices].copy()
    pos = np.array(list(f['x'].value))[indices].copy()
    #vel = np.array(list(f['v'].value))[indices].copy()

    """
    keys = np.array(list(f['hilbertKey'].value))
    indices = keys.argsort()
    keys = keys[indices]
    keys = np.array([elem for i,elem in enumerate(keys) if i % 100 = 0])
    pos = np.array(list(f['x'].value))[indices]
    vel = np.array(list(f['v'].value))[indices]
    """

    min_x = min([x[0] for i, x in enumerate(pos)])
    max_x = max([x[0] for i, x in enumerate(pos)])
    min_y = min([x[1] for i, x in enumerate(pos)])
    max_y = max([x[1] for i, x in enumerate(pos)])
    min_z = min([x[2] for i, x in enumerate(pos)])
    max_z = max([x[2] for i, x in enumerate(pos)])

    global_min = min([min_x, min_y, min_z])
    global_max = max([max_x, max_y, max_z])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim3d(global_min, global_max)
    ax.set_ylim3d(global_min, global_max)
    ax.set_zlim3d(global_min, global_max)

    ax.set_axis_off()

    ind = [0]
    for i in range(len(ranges) - 2):
        ind.append(np.argwhere(keys > ranges[i + 1])[0][0])
    ind.append(-1)

    pos_x = [x[2] for x in pos]
    pos_y = [x[0] for x in pos]
    pos_z = [x[1] for x in pos]

    for i in range(len(ranges) - 1):
        print("from {} to {}\n".format(ind[0], ind[i+1]))
        ax.scatter(pos_x[ind[i]:ind[i+1]],
                   pos_y[ind[i]:ind[i+1]],
                   pos_z[ind[i]:ind[i+1]],
                   s=5, c=colors[i])

        # ax.plot(pos_x[ind[i]:ind[i+1]],
        #         pos_y[ind[i]:ind[i+1]],
        #         pos_z[ind[i]:ind[i+1]],
        #         c=colors[i], alpha=0.3)

    plt.show()

    #import tikzplotlib
    #tikzplotlib.save("test.tex")


if __name__ == '__main__':

    file = '../log/test.h5'
    particleDistribution(file)
