#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':


    # Paralobstar
    """
    np = [8, 16, 20, 40, 50, 80, 100, 125, 160, 200]
    markers = [8, 20, 40, 50, 80, 100, 125, 160, 200, 400, 800]
    elapsed = [100994.86022, 61141.99020, 51101.47589, 29581.01308, 24346.25784, 18772.85896, 16259.64008,
               13853.33346, 12870.04571, 12448.04812]
    fig, ax1 = plt.subplots(figsize=(6, 4), dpi=100)
    ax1.scatter(np, elapsed, marker="x", color="blue")

    # Gadget
    # Cores     |    200  |     420 |     840 |
    # treegrav  |   3.881 |   1.942 |   1.041 |
    # domain    |   0.386 |   0.522 |   1.488 |
    # TOTAL     |   4.267 |   2.463 |   2.529 |

    ax1.set_xticks(markers)

    ax1.set_xlabel("number of processes/cores")
    ax1.set_ylabel("time in ms")

    plt.show()
    # fig.tight_layout()
    # plt.savefig("cluster_comparison/figures/paralobstar_binac_plummer_1e7.png")

    sys.exit(0)
    """

    # parser = argparse.ArgumentParser(description="Rhs performance evaluation")
    # parser.add_argument("--data", "-d", metavar="str", type=str, help="input directory",
    #                    nargs="?", default="../output")
    # parser.add_argument("--output", "-o", metavar="str", type=str, help="output directory",
    #                    nargs="?", default="../output")
    # args = parser.parse_args()

    key_to_investigate = "time/rhsElapsed"  # "time/IO", "time/sph_fixedRadiusNN"

    nbi_files = {
        "pl_N10000000_sfc0F_np1": {"file": "cluster_comparison/nbi/pl_N10000000_sfc0F_np1_performance.h5", "np": 1,
                                   "sfc": "0F", "n": 1e7, "cluster": "Tesla V100"},
        "pl_N10000000_sfc1D_np2": {"file": "cluster_comparison/nbi/pl_N10000000_sfc1D_np2_performance.h5", "np": 2,
                                   "sfc": "1D", "n": 1e7, "cluster": "Tesla V100"},
        "pl_N10000000_sfc1D_np4": {"file": "cluster_comparison/nbi/pl_N10000000_sfc1D_np4_performance.h5", "np": 4,
                                   "sfc": "1D", "n": 1e7, "cluster": "Tesla V100"},
    }

    bwUni_files = {
        "pl_N10000000_sfc0F_np1": {"file": "cluster_comparison/bwUni/pl_N10000000_sfc0F_np1_performance.h5", "np": 1,
                                   "sfc": "0F", "n": 1e7, "num": r"$10^{7}$", "cluster": "Tesla V100"},
        "pl_N10000000_sfc1D_np2": {"file": "cluster_comparison/bwUni/pl_N10000000_sfc1D_np2_performance.h5", "np": 2,
                                   "sfc": "1D", "n": 1e7, "num": r"$10^{7}$", "cluster": "Tesla V100"},
        "pl_N10000000_sfc1D_np4": {"file": "cluster_comparison/bwUni/pl_N10000000_sfc1D_np4_performance.h5", "np": 4,
                                   "sfc": "1D", "n": 1e7, "num": r"$10^{7}$", "cluster": "Tesla V100"},
    }

    naboo_files = {
        "pl_N10000000_sfc0F_np1": {"file": "cluster_comparison/naboo/pl_N10000000_sfc0F_np1_performance.h5", "np": 1,
                                   "sfc": "0F", "n": 1e7, "cluster": "naboo"},
        "pl_N10000000_sfc1D_np2": {"file": "cluster_comparison/naboo/pl_N10000000_sfc1D_np2_performance.h5", "np": 2,
                                   "sfc": "1D", "n": 1e7, "cluster": "naboo"},
    }


    binac_files = {
        "pl_N10000000_sfc0F_np1": {"file": "cluster_comparison/binac/pl_N10000000_sfc0F_np1_performance.h5", "np": 1,
                                   "sfc": "0F", "n": 1e7, "cluster": "binac", "np1": 2191.8553383168305},
        "pl_N10000000_sfc1D_np2": {"file": "cluster_comparison/binac/pl_N10000000_sfc1D_np2_performance.h5", "np": 2,
                                   "sfc": "1D", "n": 1e7, "cluster": "binac", "np1": 2191.8553383168305},
        "pl_N10000000_sfc1D_np4": {"file": "cluster_comparison/binac/pl_N10000000_sfc1D_np4_performance.h5", "np": 4,
                                   "sfc": "1D", "n": 1e7, "cluster": "binac", "np1": 2191.8553383168305},
        "pl_N10000000_sfc1D_np8": {"file": "cluster_comparison/binac/pl_N10000000_sfc1D_np8_performance.h5", "np": 8,
                                   "sfc": "1D", "n": 1e7, "cluster": "binac", "np1": 2191.8553383168305},
        "pl_N10000000_sfc1D_np16": {"file": "cluster_comparison/binac/pl_N10000000_sfc1D_np16_performance.h5", "np": 16,
                                   "sfc": "1D", "n": 1e7, "cluster": "binac", "np1": 2191.8553383168305},
    }


    new_binac_files = {
        "pl_N10000000_sfc0F_np1": {"file": "cluster_comparison/binac/pl_N10000000_sfc0F_np1_performance.h5", "np": 1,
                                   "sfc": "0F", "n": 1e7, "num": r"$10^{7}$", "cluster": "Tesla K80", "np1": 2191.8553383168305},
        "pl_N10000000_sfc1D_np2": {"file": "cluster_comparison/binac/new_pl_N10000000_sfc1D_np2_performance.h5", "np": 2,
                                   "sfc": "1D", "n": 1e7, "num": r"$10^{7}$", "cluster": "Tesla K80", "np1": 2191.8553383168305},
        "pl_N10000000_sfc1D_np4": {"file": "cluster_comparison/binac/new_pl_N10000000_sfc1D_np4_performance.h5", "np": 4,
                                   "sfc": "1D", "n": 1e7, "num": r"$10^{7}$", "cluster": "Tesla K80", "np1": 2191.8553383168305},
        "pl_N10000000_sfc1D_np8": {"file": "cluster_comparison/binac/new_pl_N10000000_sfc1D_np8_performance.h5", "np": 8,
                                   "sfc": "1D", "n": 1e7, "num": r"$10^{7}$", "cluster": "Tesla K80", "np1": 2191.8553383168305},
        "pl_N10000000_sfc1D_np16": {"file": "cluster_comparison/binac/new_pl_N10000000_sfc1D_np16_performance.h5", "np": 16,
                                    "sfc": "1D", "n": 1e7, "num": r"$10^{7}$", "cluster": "Tesla K80", "np1": 2191.8553383168305},
        "pl_N10000000_sfc1D_np32": {"file": "cluster_comparison/binac/new_pl_N10000000_sfc1D_np32_performance.h5", "np": 32,
                                    "sfc": "1D", "n": 1e7, "num": r"$10^{7}$", "cluster": "Tesla K80", "np1": 2191.8553383168305},
    }

    new_binac_files_many_particles = {
        #"pl_N10000000_sfc0F_np1": {"file": "cluster_comparison/binac/pl_N10000000_sfc0F_np1_performance.h5", "np": 1,
        #                           "sfc": "0F", "n": 1e7, "num": r"$10^{7}$", "cluster": "Tesla K80", "np1": 2191.8553383168305},
        #"pl_N10000000_sfc1D_np2": {"file": "cluster_comparison/binac/new_pl_N10000000_sfc1D_np2_performance.h5", "np": 2,
        #                           "sfc": "1D", "n": 1e7, "num": r"$10^{7}$", "cluster": "Tesla K80", "np1": 2191.8553383168305},
        "pl_N50000000_sfc1D_np4": {"file": "cluster_comparison/binac/new_pl_N50000000_sfc1D_np4_performance.h5", "np": 4,
                                   "sfc": "1D", "n": 5e7, "num": r"$5 \cdot 10^{7}$", "cluster": "Tesla K80", "np1": 0.0},
        "pl_N50000000_sfc1D_np8": {"file": "cluster_comparison/binac/new_pl_N50000000_sfc1D_np8_performance.h5", "np": 8,
                                   "sfc": "1D", "n": 5e7, "num": r"$5 \cdot 10^{7}$", "cluster": "Tesla K80", "np1": 0.0},
        "pl_N50000000_sfc1D_np16": {"file": "cluster_comparison/binac/new_pl_N50000000_sfc1D_np16_performance.h5", "np": 16,
                                    "sfc": "1D", "n": 5e7, "num": r"$5 \cdot 10^{7}$", "cluster": "Tesla K80", "np1": 0.0},
    }

    file_dic_collections = [new_binac_files] #, new_binac_files_many_particles] #, bwUni_files]  # [nbi_files] # , naboo_files, binac_files] [binac_files] #
    colors = ["k", "blue", "darkgreen", "red"]
    markers = ["+", "x", "*"]
    markersize = 60

    # fig, ax1 = plt.subplots(figsize=(12, 9), dpi=200)
    fig, ax1 = plt.subplots(figsize=(5, 3), dpi=250)

    for i_file_dic_collection, file_dic_collection in enumerate(file_dic_collections):
        for file_dic in file_dic_collection:
            # print(file_dic)
            # print(file_dic_collection[file_dic]["file"])
            f = h5py.File(file_dic_collection[file_dic]["file"], 'r')
            rhs_elapsed = np.array(f[key_to_investigate][:])
            rhs_elapsed_max = [np.array(elem).max() for elem in rhs_elapsed]
            mean_rhs_elapsed = np.array(rhs_elapsed_max).mean()
            file_dic_collection[file_dic]["rhs"] = mean_rhs_elapsed
            print("{}: {} average: {}".format(file_dic_collection[file_dic]["file"], key_to_investigate,
                                              mean_rhs_elapsed))

            """
            if file_dic_collection[file_dic]["np"] == 4:
                ax1.scatter(file_dic_collection[file_dic]["np"], file_dic_collection[file_dic]["rhs"],
                            c=colors[i_file_dic_collection],  # c=colors[i_file_dic_collection],
                            label=r"milupHPC on {}: n = {}".format(file_dic_collection[file_dic]["cluster"],
                                                                       file_dic_collection[file_dic]["num"]), marker=markers[i_file_dic_collection],
                            s=markersize)
            else:
                ax1.scatter(file_dic_collection[file_dic]["np"], file_dic_collection[file_dic]["rhs"],
                            c=colors[i_file_dic_collection], marker=markers[i_file_dic_collection],
                            s=markersize)  # c=colors[i_file_dic_collection],)
            """

            """
            if file_dic_collection[file_dic]["n"] == 101:
                if file_dic_collection[file_dic]["np"] == 1:
                    ax1.scatter(file_dic_collection[file_dic]["np"], file_dic_collection[file_dic]["rhs"],
                                c=colors[0],#c=colors[i_file_dic_collection],
                                label=r"milupHPC on {}: n = ${}^3$".format(file_dic_collection[file_dic]["cluster"],
                                                            file_dic_collection[file_dic]["n"]), marker="x", s=markersize)
                else:
                    ax1.scatter(file_dic_collection[file_dic]["np"], file_dic_collection[file_dic]["rhs"],
                                c=colors[0], marker="x", s=markersize)#c=colors[i_file_dic_collection],)
            else:
                if file_dic_collection[file_dic]["np"] == 1:
                    ax1.scatter(file_dic_collection[file_dic]["np"], file_dic_collection[file_dic]["rhs"],
                                c=colors[1],#c=colors[i_file_dic_collection],
                                label=r"milupHPC on {}: n = ${}^3$".format(file_dic_collection[file_dic]["cluster"],
                                                                           file_dic_collection[file_dic]["n"]), s=markersize)
                else:
                    ax1.scatter(file_dic_collection[file_dic]["np"], file_dic_collection[file_dic]["rhs"],
                                c=colors[1], s=markersize)#c=colors[i_file_dic_collection],
            """
            #########

            # parallel speedup
            """
            if file_dic_collection[file_dic]["np"] == 1:
                ax1.scatter(file_dic_collection[file_dic]["np"], file_dic_collection[file_dic]["np1"]/file_dic_collection[file_dic]["rhs"],
                            c=colors[0],#c=colors[i_file_dic_collection],
                            label=r"milupHPC on {}: n = {}".format(file_dic_collection[file_dic]["cluster"],
                                                                       file_dic_collection[file_dic]["num"]), marker="x", s=markersize)
            else:
                ax1.scatter(file_dic_collection[file_dic]["np"], file_dic_collection[file_dic]["np1"]/file_dic_collection[file_dic]["rhs"],
                            c=colors[0], marker="x", s=markersize)#c=colors[i_file_dic_collection],)
            """
            ##########

            ##########
            # parallel efficiency

            if file_dic_collection[file_dic]["np"] == 1:
                ax1.scatter(file_dic_collection[file_dic]["np"], file_dic_collection[file_dic]["np1"]/file_dic_collection[file_dic]["rhs"]/file_dic_collection[file_dic]["np"],
                            c=colors[0],#c=colors[i_file_dic_collection],
                            label=r"milupHPC on {}: n = {}".format(file_dic_collection[file_dic]["cluster"],
                                                                       file_dic_collection[file_dic]["num"]), marker="x", s=markersize)
            else:
                ax1.scatter(file_dic_collection[file_dic]["np"], file_dic_collection[file_dic]["np1"]/file_dic_collection[file_dic]["rhs"]/file_dic_collection[file_dic]["np"],
                            c=colors[0], marker="x", s=markersize)#c=colors[i_file_dic_collection],)

            ##########
            # ax1.text(file_dic_collection[file_dic]["np"] + 0.15, file_dic_collection[file_dic]["rhs"],
            #         "{} ms".format(round(file_dic_collection[file_dic]["rhs"], 2)), verticalalignment='center',
            #         fontsize=10)

    # fig.patch.set_facecolor("black")
    ax1.set_xlabel("number of processes/GPUs")

    # ax1.set_ylabel("time in ms")
    ax1.set_ylabel("parallel efficiency")
    # ax1.set_ylabel("parallel speedup")

    # milupHPC sedov N = 126 on de.NBI FRNN 0
    # ax1.scatter(1, 800.0, c="k", label=r"miluphcuda on de.NBI: n=$126^3$")
    # ax1.text(1 + 0.15, 800.0,
    #         "{} ms".format(800.0), verticalalignment='center',
    #         fontsize=10)

    # miluphcuda plummer N = 1e7 on de.NBI
    #ax1.scatter(1, 10000.0, c="darkred", label=r"miluphcuda on de.NBI: n = ${:.0e}$".format(1e7))
    #ax1.text(1 + 0.15, 10000.0,
    #         r"$\approx$ {} ms".format(10000.0), verticalalignment='center',
    #         fontsize=10)

    # milupHPC grav force v2 N = 1e7 on de.NBI
    #ax1.scatter(1, 5485.93, c="k", label=r"milupHPC with miluphcuda force", marker="+")
    #ax1.text(1 + 0.15, 5485.93,
    #         r"{} ms".format(5485.93), verticalalignment="center",
    #         fontsize=10)

    # Paralobstar 200 cores: 12448.048 ms = 12.4s
    #ax1.axhline(y=12448.048, color="k", linestyle="--")
    #ax1.text(1 + 0.15, 12000,
    #         r"Paralobstar on 200 cores: {} ms".format(12448), verticalalignment="center",
    #         fontsize=10)

    # Gadget 420 cores: 2463 ms
    #ax1.axhline(y=2463, color="k", linestyle="--")
    #ax1.text(2 + 0.15,  2333,
    #         r"Gadget on 420 cores: {} ms".format(2463), verticalalignment="center",
    #         fontsize=10)

    # ideal = np.linspace(1, 4, 200)
    # ax1.plot(ideal, ideal, c="red", label="ideal")

    # ax1.set_xlim([0.5, 4.5])
    #ax1.set_xlim([0.5, 16.5])
    ax1.set_xlim([0.5, 32.5])
    # ax1.set_ylim([0.2, 1.1])
    # ax1.set_xticks([1, 2, 4])
    # ax1.set_xticks([1, 2, 4, 8, 16])
    ax1.set_xticks([1, 2, 4, 8, 16, 32])
    y_max = ax1.get_ylim()
    # ax1.set_ylim([0, y_max[1]])

    ax1.grid(alpha=0.4, color='grey')

    ax1.legend(loc="best")
    # ax1.legend(loc='center right')
    # ax1.legend(loc='upper center')
    # plt.show()

    fig.tight_layout()
    plt.savefig("cluster_comparison/figures/plummer_parallel_efficiency_32.png")
