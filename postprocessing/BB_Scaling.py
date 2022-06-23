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

    # 30000
    # 0F_np1: 773.680761467499
    # 0F_np2: 773.680761467499

    # 100000 (short)
    # 0F_np1: 2332.3972542774986
    # 1D_np2: 1981.8337958975012, 1D_np4: 1991.8739401050004,
    # 100000 (long in milupHPC 2)
    # 0F_np1: 2584.9774381927127
    # 0F_np2: 2252.519718508292, 0F_np4: 1840.38367494199
    # 0D_np2: 2189.8912116902334, 0D_np4: 2295.8276118590215
    # 1F_np2: 2180.589345193249, 1F_np4: 2178.5231193212408
    # 1D_np2: 2262.6005279090473, 1D_np4: 2338.5032033177877

    # 500000 (from short)
    # 0F_np1: 18151.170311679984
    N500000_1 = 18151.170311679984
    # 1D_np2: 14289.716971402497, 1D_np4: 13182.13439080249, 1D_np8: 12288.852851947497, 1D_np16: 12997.052494097492, 1D_np32: 14069.475038902494
    N500000_1D = [14289.716971402497, 13182.13439080249, 12288.852851947497, 12997.052494097492, 14069.475038902494]

    # N 500000
    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    ax.scatter(1, N500000_1, marker="o", color="black", label="Single-GPU")
    for i, elapsed in enumerate(N500000_1D):
        if i == 0:
            ax.scatter(num_processes[i], elapsed, marker="x", color="black", label="Multi-GPU, Hilbert, dynamic LB")
        else:
            ax.scatter(num_processes[i], elapsed, marker="x", color="black")
    ax.legend(loc="best")
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    # ax.set_xticklabels([1,4,5], fontsize=12)
    ax.set_ylabel(r"Time $t$ in ms")
    ax.set_xlabel(r"number of processes")
    fig.savefig("bb_scaling.png", bbox_inches="tight")
    #plt.show()