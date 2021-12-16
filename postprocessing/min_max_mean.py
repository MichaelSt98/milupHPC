#!/usr/bin/env python3

import numpy as np
import h5py
import argparse
import csv


def get_dataset_keys(f):
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="min/max/mean for all entries")
    parser.add_argument("--input", "-i", metavar="str", type=str, help="input directory",
                        nargs="?", default="../output")
    parser.add_argument("--output", "-o", metavar="str", type=str, help="output directory",
                        nargs="?", default="-")
    args = parser.parse_args()

    f = h5py.File(args.input, 'r')

    keys = get_dataset_keys(f)
    # print(keys)

    data = {}

    for key in keys:
        # print("reading: {}".format(key))
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

    if args.output != "-":
        with open("{}min_max_mean.csv".format(args.output), 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=";")
            header = ["key", "min", "max", "mean"]
            csv_writer.writerow(header)
            for key in data:
                csv_writer.writerow([key, min[key], max[key], mean[key]])
