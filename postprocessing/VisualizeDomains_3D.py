import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DIM = 3

DirTable = [[8,  10, 3,  3,  4,  5,  4,  5],
            [2,  2,  11, 9,  4,  5,  4,  5],
            [7,  6,  7,  6,  8,  10, 1,  1],
            [7,  6,  7,  6,  0,  0,  11, 9],
            [0,  8,  1,  11, 6,  8,  6,  11],
            [10, 0,  9,  1,  10, 7,  9,  7],
            [10, 4,  9,  4,  10, 2,  9,  3],
            [5,  8,  5,  11, 2,  8,  3,  11],
            [4,  9,  0,  0,  7,  9,  2,  2],
            [1,  1,  8,  5,  3,  3,  8,  6],
            [11, 5,  0,  0,  11, 6,  2,  2],
            [1,  1,  4,  10, 3,  3,  7,  10]]

HilbertTable = [[0, 7, 3, 4, 1, 6, 2, 5],
                [4, 3, 7, 0, 5, 2, 6, 1],
                [6, 1, 5, 2, 7, 0, 4, 3],
                [2, 5, 1, 6, 3, 4, 0, 7],
                [0, 1, 7, 6, 3, 2, 4, 5],
                [6, 7, 1, 0, 5, 4, 2, 3],
                [2, 3, 5, 4, 1, 0, 6, 7],
                [4, 5, 3, 2, 7, 6, 0, 1],
                [0, 3, 1, 2, 7, 4, 6, 5],
                [2, 1, 3, 0, 5, 6, 4, 7],
                [4, 7, 5, 6, 3, 0, 2, 1],
                [6, 5, 7, 4, 1, 2, 0, 3]]


def lebesgue2Hilbert(lebesgue, maxLevel):
    direction = 0
    hilbert = 0
    for lvl in range(maxLevel, 0, -1):
        cell = (lebesgue >> ((lvl-1)*DIM)) & ((1 << DIM)-1)
        hilbert = hilbert << DIM
        if lvl > 0:
            hilbert += HilbertTable[direction][cell]
        direction = DirTable[direction][cell]
    return hilbert


def getCube():

    phi_ = np.arange(1, 10, 2) * np.pi/4
    phi, theta = np.meshgrid(phi_, phi_)

    x = 2 * (np.cos(phi) * np.sin(theta))
    y = 2 * (np.sin(phi) * np.sin(theta))
    z = 2 * (np.cos(theta)/np.sqrt(2))

    return x, y, z


def val2Path(val, max_level):
    val_path = []
    for i_level in range(max_level):
        val_path.append((val >> (3 * i_level)) & 7)
    val_path.reverse()
    return val_path


if __name__ == '__main__':

    plot_per_process = False
    hilbert = False

    f = h5py.File('test.h5', 'r')
    print(list(f.keys()))

    ranges = list(f['hilbertRanges'].value)
    for i in range(len(ranges)):
        print("range[{}]] = {}".format(i, ranges[i]))

    # colors = ['red', 'darkgreen', 'blue', 'black', 'yellow', 'orange', 'lightgreen', 'grey']
    colors = ['lightgrey', 'grey', 'black', 'white']

    quadrant = [(1, 1, 1),
                (-1, 1, 1),
                (1, -1, 1),
                (-1, -1, 1),
                (1, 1, -1),
                (-1, 1, -1),
                (1, -1, -1),
                (-1, -1, -1)]

    paths = []
    maxLevel = 21

    for i_range, range_i in enumerate(ranges):
        paths.append([])
        for i in range(maxLevel):
            paths[i_range].append((int(range_i) >> (3 * i)) & 7)
        paths[i_range].reverse()

    paths = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]]

    origins = [np.array([0., 0., 0.]) for i in range(len(paths) - 1)]

    maxLength = 0
    for i_path, path in enumerate(paths):
        if 0 < i_path < len(paths) - 1:
            while path[-1] == 0:
                path.pop(-1)
            if len(path) > maxLength:
                maxLength = len(path)

    for i_path, path in enumerate(paths):
        while len(path) < maxLength:
            path.append(0)
        paths[i_path] = path[0:maxLength]

    print("maxLength: {}".format(maxLength))

    pathValues = []  # [[] for i in range(len(paths))]
    for i_path, path in enumerate(paths):
        val = 0
        for i in range(maxLength):
            val += path[i] << (3 * (maxLength - (i + 1)))
        pathValues.append(val)
        # print("pathValues[{}]: {}".format(i_path, pathValues[i_path]))

    pathRanges = []
    for i_path in range(len(paths) - 2):
        pathRanges.append(np.arange(pathValues[i_path], pathValues[i_path+1]))
    pathRanges.append(np.arange(pathValues[-2], pathValues[-1] + 1))

    cubes = [[] for i in range(len(paths) - 1)]

    fig = plt.figure()
    axes = []
    if plot_per_process:
        for i in range(len(paths) - 1):
            print(i)
            axes.append(fig.add_subplot(1, len(paths) - 1, i + 1, projection='3d'))
    else:
        axes.append(fig.add_subplot(111, projection='3d'))

    edgeLength = 0.5**maxLength

    for i_path in range(len(paths) - 1):
        for i in range(len(pathRanges[i_path])):
            if hilbert:
                currentPath = val2Path(lebesgue2Hilbert(pathRanges[i_path][i], maxLength+1), maxLength)
            else:
                currentPath = val2Path(pathRanges[i_path][i], maxLength)
            currentOrigin = origins[i_path]
            for i_currentPath in range(len(currentPath)):
                currentOrigin = currentOrigin + np.array(quadrant[currentPath[i_currentPath]])*0.5**(i_currentPath+1)
            x, y, z = getCube()  # , edge_length=edgeLength)
            if plot_per_process:
                axes[i_path].plot_surface(x * edgeLength + currentOrigin[0],
                                          y * edgeLength + currentOrigin[1],
                                          z * edgeLength + currentOrigin[2],
                                          color=colors[i_path], shade=False)
            else:
                axes[0].plot_surface(x * edgeLength + currentOrigin[0],
                                     y * edgeLength + currentOrigin[1],
                                     z * edgeLength + currentOrigin[2],
                                     color=colors[i_path], shade=False)

    limit = 1.2
    for ax in axes:
        ax.set_xlim((-1) * limit, limit)
        ax.set_ylim((-1) * limit, limit)
        ax.set_zlim((-1) * limit, limit)

        ax.set_axis_off()

    plt.show()
