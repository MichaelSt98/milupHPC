import argparse
import glob
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
import csv

if __name__ == '__main__':

    num_processes = [2, 4, 8, 16, 32]

    # N61
    # 0F_np1: 613.3204178665496
    N_61_1 = 613.3204178665496
    # 1D_np2: 363.9671125506992, 1D_np4: 257.3145906952212,
    N_61_1D = [363.9671125506992, 257.3145906952212]

    # N81
    # 0F_np1: 2147.5913492239542
    N_81_1 = 2147.5913492239542
    # 1D_np2: 1139.2567952985214, 1D_np4: 631.6445726317613, 1D_np8: 498.55545619806367, 1D_np16: 572.7053802361994, 1D_np32: 795.454250190379
    N_81_1D = [1139.2567952985214, 631.6445726317613, 498.55545619806367, 572.7053802361994, 795.454250190379]

    # N101
    # 0F_np1: 4018.944116530109
    N_101_1 = 4018.944116530109
    # 1D_np2: 2155.368274348747, 1D_np4: 1248.3803499674254, 1D_np8: 867.5506475550095, 1D_np16: 763.1568961139504,
    N_101_1D = [2155.368274348747, 1248.3803499674254, 867.5506475550095, 763.1568961139504]

    if True:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
        # N61
        color_61 = "darkblue"
        ax.scatter(1, N_61_1, marker="o", color=color_61, label="Single-GPU N=$61^3$")
        for i, elapsed in enumerate(N_61_1D):
            if i == 0:
                ax.scatter(num_processes[i], elapsed, marker="x", color=color_61, label="Multi-GPU, Hilbert, dynamic LB, N=$61^3$")
            else:
                ax.scatter(num_processes[i], elapsed, marker="x", color=color_61)
        # N81
        color_N81 = "darkred"
        ax.scatter(1, N_81_1, marker="o", color=color_N81, label="Single-GPU N=$81^3$")
        for i, elapsed in enumerate(N_81_1D):
            if i == 0:
                ax.scatter(num_processes[i], elapsed, marker="x", color=color_N81, label="Multi-GPU, Hilbert, dynamic LB, N=$81^3$")
            else:
                ax.scatter(num_processes[i], elapsed, marker="x", color=color_N81)
        # N101
        color_N101 = "darkgreen"
        ax.scatter(1, N_101_1, marker="o", color=color_N101, label="Single-GPU N=$101^3$")
        for i, elapsed in enumerate(N_101_1D):
            if i == 0:
                ax.scatter(num_processes[i], elapsed, marker="x", color=color_N101, label="Multi-GPU, Hilbert, dynamic LB, N=$101^3$")
            else:
                ax.scatter(num_processes[i], elapsed, marker="x", color=color_N101)
        ax.legend(loc="best")
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.set_ylabel(r"Time $t$ in ms")
        ax.set_xlabel(r"number of processes")
        fig.savefig("sedov_scaling.png", bbox_inches='tight')
        plt.show()

    if True:
        # N 81
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        ax.scatter(1, N_81_1, marker="o", color="black", label="Single-GPU")
        for i, elapsed in enumerate(N_81_1D):
            if i == 0:
                ax.scatter(num_processes[i], elapsed, marker="x", color="black", label="Multi-GPU, Hilbert, dynamic LB")
            else:
                ax.scatter(num_processes[i], elapsed, marker="x", color="black")
        ax.legend(loc="best")
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.set_ylabel(r"Time $t$ in ms")
        ax.set_xlabel(r"number of processes")
        plt.show()