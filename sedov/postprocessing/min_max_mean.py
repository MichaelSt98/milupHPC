import numpy as np
import h5py

def get_dataset_keys(f):
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys

if __name__ == '__main__':
    f = h5py.File('sedov.h5', 'r')

    keys = get_dataset_keys(f)
    print(keys)

    data = {}

    for key in keys:
        print("reading: {}".format(key))
        data[key] = np.array(f[key][:])

    f.close()

    min = {}
    max = {}
    mean = {}

    for key in data:
        min[key] = data[key].min()
        max[key] = data[key].max()
        mean[key] = data[key].mean()

    for key in data:
        print("key: {} min = {:e} | max = {:e} | mean = {:e}".format(key, min[key], max[key], mean[key]))
