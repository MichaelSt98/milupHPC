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

    # 1M particles
    # 0F_np1: 1044.3736899250007
    M1_1 = 1044.3736899250007
    # 0F_np2: 723.9651920500004,  0F_np4: 570.8093250349999
    M1_0F = [723.9651920500004, 570.8093250349999]
    # 0D_np2: 795.8073550699999,  0D_np4: 924.04091713,      0D_np8: 1402.6497670750005
    M1_0D = [795.8073550699999, 924.04091713, 1402.6497670750005]
    # 1F_np2: 762.9085601400008,  1F_np4: 589.5497498250002
    M1_1F = [762.9085601400008, 589.5497498250002]
    # 1D_np2: 840.1012932349986,  1D_np4: 962.3819560000001, 1D_np8: 1382.68151549
    M1_1D = [840.1012932349986, 962.3819560000001, 1382.68151549]

    # 2M particles
    # 0F_np1: 2034.9214373649977
    M2_1 = 2034.9214373649977
    # 1D_np2: 1581.194329465    , 1D_np4: 1782.9512564349984, 1D_np8: 2548.4241517850005
    M2_1D = [1581.194329465, 1782.9512564349984, 2548.4241517850005]

    # 5M particles
    # 0F_np1: 3942.782217583333
    M5_1 = 3942.782217583333
    # 1D_np2: 3148.0643469166675, 1D_np4: 3365.4041649166666, 1D_np8:
    M5_1D = [3148.0643469166675, 3365.4041649166666, 4584.946050416666]

    if True:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
        # fig.tight_layout()
        ax.scatter(1, M1_1, marker="o", color="black", label="Single-GPU")
        ax.hlines(y=M1_1, xmin=0.75, xmax=4.25, linewidth=1, color="black", alpha=0.3, linestyle="--")
        # 1 Million, Hilbert dynamic
        for i, elapsed in enumerate(M1_1D):
            if i == 0:
                ax.scatter(num_processes[i], elapsed, marker="x", color="darkblue", label="Multi-GPU, Hilbert, dynamic LB")
            else:
                ax.scatter(num_processes[i], elapsed, marker="x", color="darkblue")
        # 1 Million, Lebesgue dynamic
        for i, elapsed in enumerate(M1_0D):
            if i == 0:
                ax.scatter(num_processes[i], elapsed, marker="x", color="darkgreen", label="Multi-GPU, Lebesgue, dynamic LB")
            else:
                ax.scatter(num_processes[i], elapsed, marker="x", color="darkgreen")
        # 1 Million, Hilbert fixed
        for i, elapsed in enumerate(M1_1F):
            if i == 0:
                ax.scatter(num_processes[i], elapsed, marker="+", color="darkblue", label="Multi-GPU, Hilbert, fixed")
            else:
                ax.scatter(num_processes[i], elapsed, marker="+", color="darkblue")
        # 1 Million, Lebesgue fixed
        for i, elapsed in enumerate(M1_0F):
            if i == 0:
                ax.scatter(num_processes[i], elapsed, marker="+", color="darkgreen", label="Multi-GPU, Lebesgue, fixed")
            else:
                ax.scatter(num_processes[i], elapsed, marker="+", color="darkgreen")
        ax.legend(loc="best")
        ax.set_xticks([1, 2, 4]) #, 8, 16, 32])
        ax.set_xlim([0.6, 4.4])
        ax.set_ylabel(r"Time $t$ in ms")
        ax.set_xlabel(r"number of processes")
        fig.savefig("plummer_scaling_1.png", bbox_inches="tight")
        # plt.show()

    if False:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        ax.scatter(1, M2_1, marker="o", color="black", label="Single-GPU")
        for i, elapsed in enumerate(M2_1D):
            if i == 0:
                ax.scatter(num_processes[i], elapsed, marker="x", color="black", label="Multi-GPU, Hilbert, dynamic LB")
            else:
                ax.scatter(num_processes[i], elapsed, marker="x", color="black")
        ax.legend(loc="best")
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.set_ylabel(r"Time $t$ in ms")
        ax.set_xlabel(r"number of processes")
        plt.show()

    if True:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
        # fig.tight_layout()
        # 1 Million, Hilbert, dynamic
        color_1M = "darkgreen"
        ax.scatter(1, M1_1, marker="o", color=color_1M, label="Single-GPU 1M")
        ax.hlines(y=M1_1, xmin=0.75, xmax=8.25, linewidth=1, color=color_1M, alpha=0.3, linestyle="--")
        for i, elapsed in enumerate(M1_1D):
            if i == 0:
                ax.scatter(num_processes[i], elapsed, marker="x", color=color_1M, label="Multi-GPU 1M, Hilbert, dynamic LB")
            else:
                ax.scatter(num_processes[i], elapsed, marker="x", color=color_1M)
        # 2 Million, Hilbert, dynamic
        color_2M = "red"
        ax.scatter(1, M2_1, marker="o", color=color_2M, label="Single-GPU 2M")
        ax.hlines(y=M2_1, xmin=0.75, xmax=8.25, linewidth=1, color=color_2M, alpha=0.3, linestyle="--")
        for i, elapsed in enumerate(M2_1D):
            if i == 0:
                ax.scatter(num_processes[i], elapsed, marker="x", color=color_2M, label="Multi-GPU 2M, Hilbert, dynamic LB")
            else:
                ax.scatter(num_processes[i], elapsed, marker="x", color=color_2M)
        # 5 Million, Hilbert, dynamic
        color_5M = "darkblue"
        ax.scatter(1, M5_1, marker="o", color=color_5M, label="Single-GPU 5M")
        ax.hlines(y=M5_1, xmin=0.75, xmax=8.25, linewidth=1, color=color_5M, alpha=0.3, linestyle="--")
        for i, elapsed in enumerate(M5_1D):
            if i == 0:
                ax.scatter(num_processes[i], elapsed, marker="x", color=color_5M, label="Multi-GPU 5M, Hilbert, dynamic LB")
            else:
                ax.scatter(num_processes[i], elapsed, marker="x", color=color_5M)
        #ax.legend(loc="best")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), fancybox=True, shadow=False, ncol=3, prop={'size': 6})
        ax.set_xticks([1, 2, 4, 8])
        ax.set_ylabel(r"Time $t$ in ms")
        ax.set_xlabel(r"number of processes")
        fig.savefig("plummer_scaling_2.png", bbox_inches="tight")
        #plt.show()

