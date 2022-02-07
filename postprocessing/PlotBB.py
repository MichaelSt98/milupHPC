#!/usr/bin/env python3

import numpy as np
import h5py
import glob
import matplotlib
import matplotlib.pyplot as plt
import argparse
import csv


def get_dataset_keys(f):
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Plot conservation of energy and angular momentum for Plummer test case.")
    parser.add_argument("--data", "-d", metavar="str", type=str, help="input directory",
                        nargs="?", default="../output")
    parser.add_argument("--output", "-o", metavar="str", type=str, help="output directory",
                        nargs="?", default="../output")
    # parser.add_argument("--angular_momentum", "-L", action="store_true", help="plot angular momentum (defaul: energy and mass)")

    args = parser.parse_args()

    directory = args.data
    files = glob.glob(directory + "ts*.h5")
    # print("files: {}".format(sorted(files)))
    sorted_files = sorted(files)

    max_density = []
    time = []
    for file in sorted_files:
        print("evaluating {} ...".format(file))
        f = h5py.File(file, 'r')
        max_density.append(np.array(f["rho"][:]).max())
        time.append(f["time"][0])

    # print("max densities: {}".format(max_density))
    time = np.array(time)
    t_f = 5.529e11  # free-fall time
    time_tf = time/t_f  # use free-fall time as unit

    bb_time = np.array([618528960000.0, 656398080000.0])/t_f
    bb_max_density = [5.7e-13, 9.3e-13]

    # font = {'family': 'normal', 'weight': 'bold', 'size': 18}
    # font = {'family': 'normal', 'size': 18}
    font = {'size': 14}
    matplotlib.rc('font', **font)

    fig, (ax1) = plt.subplots(nrows=1, sharex=True)
    ax1.plot(time_tf, max_density, color="k")
    ax1.scatter(bb_time, bb_max_density, color="darkblue", marker='x', label="Boss & Bodenheimer (1979)")
    # ax1.set_title("max(density) evolution")
    ax1.grid(color="darkgrey", alpha=0.5)
    ax1.set_ylim([1e-14, 1e-10])
    ax1.set_xlim([0, 1.2])
    ax1.set_yscale('log')
    ax1.set_xlabel(r"time $t$ $[t_f]$")
    ax1.set_ylabel(r"max($\rho$) $[\frac{kg}{m^3}]$")
    ax1.legend(loc='best')

    fig.tight_layout()
    fig.savefig("{}density_evolution.png".format(args.output))

    with open("{}density_evolution.csv".format(args.output), 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=";")
        header = ["time", "max_density"]
        csv_writer.writerow(header)
        csv_writer.writerow(time)
        csv_writer.writerow(max_density)


