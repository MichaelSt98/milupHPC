import h5py
import numpy as np
import collections

if __name__ == '__main__':

    file = '../log/SPH2Send.h5'

    f = h5py.File(file, 'r')
    print(list(f.keys()))

    ranges = list(f['hilbertRanges'].value)
    keys = np.array(list(f['hilbertKey'].value))
    indices = keys.argsort()
    keys = keys[indices].copy()
    pos = np.array(list(f['x'].value))[indices].copy()

    x_pos = [elem[0] for elem in pos]
    x_pos_set = set(x_pos)

    contains_duplicates = len(x_pos) != len(x_pos_set)
    print(contains_duplicates)
