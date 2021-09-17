import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DIM = 2
POW_DIM = DIM**2

DirTable = [[1, 2, 0, 0],
            [0, 1, 3, 1],
            [2, 0, 2, 3],
            [3, 3, 1, 2]]

HilbertTable = [[0, 3, 1, 2],
                [0, 1, 3, 2],
                [2, 3, 1, 0],
                [2, 1, 3, 0]]


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


def val_to_path(val, max_level):
    val_path = []
    for i_level in range(max_level):
        val_path.append((val >> (DIM * i_level)) & (POW_DIM - 1))
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

    quadrant = [(1, 1),
                (-1, 1),
                (1, -1),
                (-1, -1)]

    paths = []
    maxLevel = 21

    for i_range, range_i in enumerate(ranges):
        paths.append([])
        for i in range(maxLevel):
            paths[i_range].append((int(range_i) >> (DIM * i)) & (POW_DIM - 1))
        paths[i_range].reverse()

    paths = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [2, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]

    # paths = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #          [1, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #          [2, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #          [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]

    origins = [np.array([0., 0.]) for i in range(len(paths) - 1)]

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
        print("path: {}".format(paths[i_path]))

    print("maxLength: {}".format(maxLength))

    pathValues = []
    for i_path, path in enumerate(paths):
        val = 0
        for i in range(maxLength):
            val += path[i] << (DIM * (maxLength - (i + 1)))
        pathValues.append(val)
        # print("pathValues[{}]: {}".format(i_path, pathValues[i_path]))

    pathRanges = []
    for i_path in range(len(paths) - 2):
        pathRanges.append(np.arange(pathValues[i_path], pathValues[i_path+1]))
    pathRanges.append(np.arange(pathValues[-2], pathValues[-1] + 1))

    cubes = [[] for i in range(len(paths) - 1)]

    fig = plt.figure(dpi=100)
    axes = []
    if plot_per_process:
        for i in range(len(paths) - 1):
            axes.append(fig.add_subplot(1, len(paths) - 1, i + 1))
    else:
        axes.append(fig.add_subplot(111))

    edgeLength = 0.5**maxLength

    for i_path in range(len(paths) - 1):
        for i in range(len(pathRanges[i_path])):
            if hilbert:
                currentPath = val_to_path(lebesgue2Hilbert(pathRanges[i_path][i], maxLength+1), maxLength)
            else:
                currentPath = val_to_path(pathRanges[i_path][i], maxLength)
            currentOrigin = origins[i_path]
            for i_currentPath in range(len(currentPath)):
                currentOrigin = currentOrigin + np.array(quadrant[currentPath[i_currentPath]])*0.5**(i_currentPath+1)
            if plot_per_process:
                axes[i_path].add_patch(patches.Rectangle(currentOrigin-edgeLength, 2*edgeLength, 2*edgeLength,
                                                         facecolor=colors[i_path]))
            else:
                axes[0].add_patch(patches.Rectangle(currentOrigin-edgeLength, 2*edgeLength, 2*edgeLength,
                                                    facecolor=colors[i_path]))

    limit = 1.2
    for ax in axes:
        ax.axis('equal')
        ax.set_axis_off()

    plt.show()
