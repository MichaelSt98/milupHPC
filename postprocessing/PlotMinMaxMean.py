#!/usr/bin/env python3

import numpy as np
import h5py
import argparse
import matplotlib
import matplotlib.pyplot as plt
import csv
import glob
import os


def get_dataset_keys(f):
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys


def plot(time, data, key, figure_name):
    os.system("mkdir -p {}min_max_mean/".format(figure_name))
    if "min" in key or "mean" in key or "max" in key:
        fig, (ax1) = plt.subplots(nrows=1, sharex=True)
        plt.subplots_adjust(hspace=0.1)
        ax1.plot(time, data[key], c="k")  # , s=0.1, alpha=0.3)
        ax1.set_title("Key: {}".format(key))
        # ax1.set_xlabel(r'$r$')
        # ax1.set_ylabel(r'$\rho$')
        # ax1.set_xlim(0, r_max)
        # ax1.set_ylim(0, 4.0)
        fig.tight_layout()
        plt.savefig("{}min_max_mean/min_max_mean_{}.png".format(figure_name, key))
    else:
        fig, (ax1) = plt.subplots(nrows=1, sharex=True)
        plt.subplots_adjust(hspace=0.1)
        # linestyle="dotted",
        # linestyle="dashed",
        # linestyle="dashdot"
        ax1.plot(time, data["{}_min".format(key)], c="k", linestyle="dotted", label="min")  # , s=0.1, alpha=0.3)
        ax1.plot(time, data["{}_max".format(key)], c="k", linestyle="dashed", label="max")
        ax1.plot(time, data["{}_mean".format(key)], c="k", linestyle="dashdot", label="mean")
        ax1.set_title("Key: {}".format(key))
        ax1.legend(loc="best")
        # ax1.set_xlabel(r'$r$')
        # ax1.set_ylabel(r'$\rho$')
        # ax1.set_xlim(0, r_max)
        # ax1.set_ylim(0, 4.0)
        fig.tight_layout()
        plt.savefig("{}min_max_mean/min_max_mean_{}.png".format(figure_name, key))


if __name__ == '__main__':

    plot_all = False

    parser = argparse.ArgumentParser(description="min/max/mean for all entries")
    parser.add_argument("--input", "-i", metavar="str", type=str, help="input file",
                        nargs="?", default="../output")
    parser.add_argument("--output", "-o", metavar="str", type=str, help="output directory",
                        nargs="?", default="-")
    parser.add_argument("--key", "-k", metavar="str", type=str, help="output directory",
                        nargs="?", default="-")
    parser.add_argument('--all', "-a", action='store_true')
    args = parser.parse_args()

    if args.all:
        plot_all = True

    header = []
    detailed_header = []
    data = {}

    with open(args.input, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for i_row, row in enumerate(reader):
            if i_row == 0:
                for elem in row:
                    header.append(elem)
            elif i_row == 1:
                for elem in row:
                    detailed_header.append(elem)
                    data[elem] = []
            else:
                for i_elem, elem in enumerate(row):
                    data[detailed_header[i_elem]].append(float(elem))

    # print("data[{}] = {}".format(args.key, data[args.key]))
    if args.key in header or args.key in detailed_header:
        plot(data["time_max"], data, args.key, args.output)

    if plot_all:
        for _header in header:
            print("plotting {} ...".format(_header))
            plot(data["time_max"], data, _header, args.output)
