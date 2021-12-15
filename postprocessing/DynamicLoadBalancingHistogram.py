#!/usr/bin/env python3

from statistics import mean
import numpy as np
import h5py
import argparse
import sys
import csv
import matplotlib.pyplot as plt


def get_dataset_keys(f):
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Performance postprocessing, generating csv (summary) files.")
    parser.add_argument("--file", "-f",  metavar="str", type=str, help="path to the h5 profiling data file", required=True)
    #parser.add_argument("--directory", "-d", metavar="str", type=str, help="output path", required=True)
    #parser.add_argument("--sim_type", "-s", metavar="int", type=int, help="simulation type", required=True)
    args = parser.parse_args()

    f = h5py.File(args.file, 'r')
    keys = get_dataset_keys(f)
    histogram = f["bins"][:]

    fig, (ax1) = plt.subplots(nrows=1, sharex=True)
    plt.subplots_adjust(hspace=0.1)
    ax1.hist(histogram, len(histogram))
    # ax1.set_title("Time t = %.2e" % time)
    # ax1.set_xlabel(r'$r$')
    # ax1.set_ylabel(r'$\rho$')
    # ax1.set_xlim(0, r_max)
    # ax1.set_ylim(0, 4.0)
    plt.show()

