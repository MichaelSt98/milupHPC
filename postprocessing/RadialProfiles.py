#!/usr/bin/env python3

import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse
import sys


def get_dataset_keys(f):
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Plotting Sedov simulation data and analytical solution.")
    parser.add_argument("--input", "-i",  metavar="str", type=str, help="input file", required=True)
    parser.add_argument("--output", "-o", metavar="str", type=str, help="output path/file", required=True)
    parser.add_argument("--plot_type", "-p", metavar="int", type=int,
                        help="plot type ([0]: rho; [1]: rho, p, e;[2]: rho, p, e, noi)", required=True)
    args = parser.parse_args()

    plot_type = args.plot_type

    f = h5py.File(args.input, 'r')

    keys = get_dataset_keys(f)
    print(keys)

    data = {}

    for key in keys:
        print("reading: {}".format(key))
        data[key] = np.array(f[key][:])

    f.close()

    data["x_flattened"] = data["x"].flatten()
    data["x_pos"] = data["x_flattened"][0:-2:3].copy()
    data["y_pos"] = data["x_flattened"][1:-1:3].copy()
    data["z_pos"] = data["x_flattened"][2::3].copy()

    print("length: x_pos = {}, y_pos {}, z_pos = {}".format(data["x_pos"].size, data["y_pos"].size, data["z_pos"].size))
    print("length: rho = {}".format(data["rho"].size))

    data["radii"] = np.sqrt(np.square(data["x_pos"]) + np.square(data["y_pos"]) + np.square(data["z_pos"]))

    s = 2

    if plot_type == 0:
        fig, (ax1) = plt.subplots(nrows=1, sharex=True)
        ax1.scatter(data["radii"], data["rho"], s=s, label=r"$\rho$")
        ax1.legend(loc='best')
        # ax1.set_ylim([0, 1])
    elif plot_type == 1:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
        ax1.scatter(data["radii"], data["rho"], s=s, label=r"$\rho$")
        ax2.scatter(data["radii"], data["p"], s=s, label=r"$p$")
        ax3.scatter(data["radii"], data["e"], s=s, label=r"$e$")
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        ax3.legend(loc='best')
        # ax1.set_ylim([0, 1])
        ax2.set_ylim([0, 25])
        # ax3.set_ylim([0, 1])
    elif plot_type == 2:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True)
        ax1.scatter(data["radii"], data["rho"], s=s, label=r"$\rho$")
        ax2.scatter(data["radii"], data["p"], s=s, label=r"$p$")
        ax3.scatter(data["radii"], data["e"], s=s, label=r"$e$")
        ax3.scatter(data["radii"], data["noi"], s=s, label="#interactions")
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        ax3.legend(loc='best')
        ax4.legend(loc='best')
        # ax1.set_ylim([0, 1])
        ax2.set_ylim([0, 25])
        # ax3.set_ylim([0, 1])
        ax4.set_ylim([0, 120])
    else:
        sys.exit(1)

    figure_file_name = args.output
    if ".png" not in figure_file_name:
        figure_file_name = "{}.csv".format(figure_file_name)
    fig.savefig(figure_file_name)
