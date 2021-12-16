#!/usr/bin/env python3

import numpy as np
import h5py
import argparse
import csv
import glob


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

    directory = args.input
    files = glob.glob(directory + "ts*.h5")
    # print("files: {}".format(sorted(files)))
    sorted_files = sorted(files)

    vec_entries = ["x", "v"]

    with open("{}min_max_mean.csv".format(args.output), 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=";")
        # header = ["time", "quantiles_0", "quantiles_1", "quantiles_2"]
        # csv_writer.writerow(header)
        # csv_writer.writerow(time)
        # csv_writer.writerow(quantiles[:, 0])
        # csv_writer.writerow(quantiles[:, 1])
        # csv_writer.writerow(quantiles[:, 2])
        for i_file, file in enumerate(sorted_files):
            print("evaluating {} ...".format(file))
            data_dic = {}
            f = h5py.File(file, 'r')
            keys = get_dataset_keys(f)
            for key in keys:
                data = np.array(f[key][:])
                if key in vec_entries:
                    amplitude = [np.sqrt(elem[0]**2 + elem[1]**2 + elem[2]**2) for elem in data]
                    data = np.array(amplitude)
                data_dic[key] = [data.min(), data.mean(), data.max()]
            # write header
            if i_file == 0:
                header = []
                sorted_keys = sorted(keys)
                for key in sorted_keys:
                    header.append(key)
                csv_writer.writerow(header)
                header = []
                sorted_keys = sorted(keys)
                for key in sorted_keys:
                    header.extend(["{}_min".format(key), "{}_mean".format(key), "{}_max".format(key)])
                csv_writer.writerow(header)
            # write data
            csv_data = []
            for key in sorted_keys:
                csv_data.extend(data_dic[key])
            csv_writer.writerow(csv_data)

    """
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
    """
