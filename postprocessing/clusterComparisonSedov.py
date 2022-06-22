#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description="Rhs performance evaluation")
    # parser.add_argument("--data", "-d", metavar="str", type=str, help="input directory",
    #                    nargs="?", default="../output")
    # parser.add_argument("--output", "-o", metavar="str", type=str, help="output directory",
    #                    nargs="?", default="../output")
    # args = parser.parse_args()

    key_to_investigate = "time/rhsElapsed"  # "time/IO", "time/sph_fixedRadiusNN"

    nbi_files = {
        #"nbi_sedov_N101_sfc0F_np1": {"file": "cluster_comparison/nbi/sedov_N101_sfc0F_np1_performance.h5", "np": 1,
        #                            "sfc": "0F", "n": 101, "cluster": "de.nbi"},
        #"nbi_sedov_N101_sfc1D_np2": {"file": "cluster_comparison/nbi/sedov_N101_sfc1D_np2_performance.h5", "np": 2,
        #                             "sfc": "1D", "n": 101, "cluster": "de.nbi"},
        #"nbi_sedov_N101_sfc1D_np4": {"file": "cluster_comparison/nbi/sedov_N101_sfc1D_np4_performance.h5", "np": 4,
        #                             "sfc": "1D", "n": 101, "cluster": "de.nbi"},
        "nbi_sedov_N126_sfc0F_np1": {"file": "cluster_comparison/nbi/sedov_N126_sfc0F_np1_performance.h5", "np": 1,
                                     "sfc": "0F", "n": 126, "cluster": "de.NBI"},
        "nbi_sedov_N126_sfc1D_np2": {"file": "cluster_comparison/nbi/sedov_N126_sfc1D_np2_performance.h5", "np": 2,
                                     "sfc": "1D", "n": 126, "cluster": "de.NBI"},
        "nbi_sedov_N126_sfc1D_np4": {"file": "cluster_comparison/nbi/sedov_N126_sfc1D_np4_performance.h5", "np": 4,
                                     "sfc": "1D", "n": 126, "cluster": "de.NBI"},
    }

    naboo_files = {
        "naboo_sedov_N126_sfc0F_np1": {"file": "cluster_comparison/naboo/sedov_N126_sfc0F_np1_performance.h5", "np": 1,
                                       "sfc": "0F", "n": 126, "cluster": "naboo"},
        "naboo_sedov_N126_sfc1D_np2": {"file": "cluster_comparison/naboo/sedov_N126_sfc1D_np2_performance.h5", "np": 2,
                                       "sfc": "1D", "n": 126, "cluster": "naboo"},
    }

    bwUni_files = {
        #"nbi_sedov_N101_sfc0F_np1": {"file": "cluster_comparison/nbi/sedov_N101_sfc0F_np1_performance.h5", "np": 1,
        #                            "sfc": "0F", "n": 101, "cluster": "de.nbi"},
        #"nbi_sedov_N101_sfc1D_np2": {"file": "cluster_comparison/nbi/sedov_N101_sfc1D_np2_performance.h5", "np": 2,
        #                             "sfc": "1D", "n": 101, "cluster": "de.nbi"},
        #"nbi_sedov_N101_sfc1D_np4": {"file": "cluster_comparison/nbi/sedov_N101_sfc1D_np4_performance.h5", "np": 4,
        #                             "sfc": "1D", "n": 101, "cluster": "de.nbi"},
        "bwUni_sedov_N126_sfc0F_np1": {"file": "cluster_comparison/bwUni/sedov_N126_sfc0F_np1_performance.h5", "np": 1,
                                     "sfc": "0F", "n": 126, "cluster": "Tesla V100"},
        "bwUni_sedov_N126_sfc1D_np2": {"file": "cluster_comparison/bwUni/sedov_N126_sfc1D_np2_performance.h5", "np": 2,
                                     "sfc": "1D", "n": 126, "cluster": "Tesla V100"},
        "bwUni_sedov_N126_sfc1D_np4": {"file": "cluster_comparison/bwUni/sedov_N126_sfc1D_np4_performance.h5", "np": 4,
                                     "sfc": "1D", "n": 126, "cluster": "Tesla V100"},
    }

    binac_files = {
        #"binac_sedov_N101_sfc0F_np1": {"file": "cluster_comparison/binac/sedov_N101_sfc0F_np1_performance.h5", "np": 1,
        #                              "sfc": "0F", "n": 101, "cluster": "binac"},
        #"binac_sedov_N101_sfc1D_np2": {"file": "cluster_comparison/binac/sedov_N101_sfc1D_np2_performance.h5", "np": 2,
        #                              "sfc": "1D", "n": 101, "cluster": "binac"},
        #"binac_sedov_N101_sfc1D_np4": {"file": "cluster_comparison/binac/sedov_N101_sfc1D_np4_performance.h5", "np": 4,
        #                              "sfc": "1D", "n": 101, "cluster": "binac"},
        #"binac_sedov_N101_sfc1D_np8": {"file": "cluster_comparison/binac/sedov_N101_sfc1D_np8_performance.h5", "np": 8,
        #                              "sfc": "1D", "n": 101, "cluster": "binac"},
        #"binac_sedov_N101_sfc1D_np16": {"file": "cluster_comparison/binac/sedov_N101_sfc1D_np16_performance.h5", "np": 16,
        #                               "sfc": "1D", "n": 101, "cluster": "binac"},
        "binac_sedov_N126_sfc0F_np1": {"file": "cluster_comparison/binac/sedov_N126_sfc0F_np1_performance.h5", "np": 1,
                                       "sfc": "0F", "n": 126, "cluster": "binac", "np1": 3056.739066647304},
        "binac_sedov_N126_sfc1D_np2": {"file": "cluster_comparison/binac/sedov_N126_sfc1D_np2_performance.h5", "np": 2,
                                       "sfc": "1D", "n": 126, "cluster": "binac", "np1": 3056.739066647304},
        "binac_sedov_N126_sfc1D_np4": {"file": "cluster_comparison/binac/sedov_N126_sfc1D_np4_performance.h5", "np": 4,
                                       "sfc": "1D", "n": 126, "cluster": "binac", "np1": 3056.739066647304},
        "binac_sedov_N126_sfc1D_np8": {"file": "cluster_comparison/binac/sedov_N126_sfc1D_np8_performance.h5", "np": 8,
                                       "sfc": "1D", "n": 126, "cluster": "binac", "np1": 3056.739066647304},
        "binac_sedov_N126_sfc1D_np16": {"file": "cluster_comparison/binac/sedov_N126_sfc1D_np16_performance.h5",
                                        "np": 16,
                                        "sfc": "1D", "n": 126, "cluster": "binac", "np1": 3056.739066647304},
    }


    new_binac_files_101 = {
        "binac_sedov_N101_sfc0F_np1": {"file": "cluster_comparison/binac/sedov_N101_sfc0F_np1_performance.h5", "np": 1,
                                      "sfc": "0F", "n": 101, "cluster": "Tesla K80"},
        "binac_sedov_N101_sfc1D_np2": {"file": "cluster_comparison/binac/new_sedov_N101_sfc1D_np2_performance.h5", "np": 2,
                                      "sfc": "1D", "n": 101, "cluster": "Tesla K80"},
        "binac_sedov_N101_sfc1D_np4": {"file": "cluster_comparison/binac/new_sedov_N101_sfc1D_np4_performance.h5", "np": 4,
                                      "sfc": "1D", "n": 101, "cluster": "Tesla K80"},
        "binac_sedov_N101_sfc1D_np8": {"file": "cluster_comparison/binac/new_sedov_N101_sfc1D_np8_performance.h5", "np": 8,
                                      "sfc": "1D", "n": 101, "cluster": "Tesla K80"},
        "binac_sedov_N101_sfc1D_np16": {"file": "cluster_comparison/binac/new_sedov_N101_sfc1D_np16_performance.h5", "np": 16,
                                       "sfc": "1D", "n": 101, "cluster": "Tesla K80"},
    }

    new_binac_files_126 = {
        #"binac_sedov_N101_sfc0F_np1": {"file": "cluster_comparison/binac/sedov_N101_sfc0F_np1_performance.h5", "np": 1,
        #                              "sfc": "0F", "n": 101, "cluster": "binac"},
        #"binac_sedov_N101_sfc1D_np2": {"file": "cluster_comparison/binac/sedov_N101_sfc1D_np2_performance.h5", "np": 2,
        #                              "sfc": "1D", "n": 101, "cluster": "binac"},
        #"binac_sedov_N101_sfc1D_np4": {"file": "cluster_comparison/binac/sedov_N101_sfc1D_np4_performance.h5", "np": 4,
        #                              "sfc": "1D", "n": 101, "cluster": "binac"},
        #"binac_sedov_N101_sfc1D_np8": {"file": "cluster_comparison/binac/sedov_N101_sfc1D_np8_performance.h5", "np": 8,
        #                              "sfc": "1D", "n": 101, "cluster": "binac"},
        #"binac_sedov_N101_sfc1D_np16": {"file": "cluster_comparison/binac/sedov_N101_sfc1D_np16_performance.h5", "np": 16,
        #                               "sfc": "1D", "n": 101, "cluster": "binac"},
        "binac_sedov_N126_sfc0F_np1": {"file": "cluster_comparison/binac/sedov_N126_sfc0F_np1_performance.h5", "np": 1,
                                       "sfc": "0F", "n": 126, "cluster": "Tesla K80", "np1": 3056.739066647304},
        "binac_sedov_N126_sfc1D_np2": {"file": "cluster_comparison/binac/new_sedov_N126_sfc1D_np2_performance.h5", "np": 2,
                                       "sfc": "1D", "n": 126, "cluster": "Tesla K80c", "np1": 3056.739066647304},
        "binac_sedov_N126_sfc1D_np4": {"file": "cluster_comparison/binac/new_sedov_N126_sfc1D_np4_performance.h5", "np": 4,
                                       "sfc": "1D", "n": 126, "cluster": "Tesla K80", "np1": 3056.739066647304},
        "binac_sedov_N126_sfc1D_np8": {"file": "cluster_comparison/binac/new_sedov_N126_sfc1D_np8_performance.h5", "np": 8,
                                       "sfc": "1D", "n": 126, "cluster": "Tesla K80", "np1": 3056.739066647304},
        "binac_sedov_N126_sfc1D_np16": {"file": "cluster_comparison/binac/new_sedov_N126_sfc1D_np16_performance.h5",
                                        "np": 16,
                                        "sfc": "1D", "n": 126, "cluster": "Tesla K80", "np1": 3056.739066647304},
        "binac_sedov_N126_sfc1D_np32": {"file": "cluster_comparison/binac/new_sedov_N126_sfc1D_np32_performance.h5",
                                        "np": 32,
                                        "sfc": "1D", "n": 126, "cluster": "Tesla K80", "np1": 3056.739066647304},
    }

    file_dic_collections = [new_binac_files_101, new_binac_files_126, bwUni_files] #[new_binac_files_126] #[new_binac_files_101, new_binac_files_126, bwUni_files] #[nbi_files] # [nbi_files] # , naboo_files, binac_files] [binac_files] #
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


            if file_dic_collection[file_dic]["np"] == 1:
                ax1.scatter(file_dic_collection[file_dic]["np"], file_dic_collection[file_dic]["rhs"],
                            c=colors[i_file_dic_collection],#c=colors[i_file_dic_collection],
                            label=r"milupHPC on {}: n = ${}^3$".format(file_dic_collection[file_dic]["cluster"],
                                                                       file_dic_collection[file_dic]["n"]), marker=markers[i_file_dic_collection], s=markersize)
            else:
                ax1.scatter(file_dic_collection[file_dic]["np"], file_dic_collection[file_dic]["rhs"],
                            c=colors[i_file_dic_collection], marker=markers[i_file_dic_collection], s=markersize)#c=colors[i_file_dic_collection],)


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
                            label=r"milupHPC on {}: n = ${}^3$".format(file_dic_collection[file_dic]["cluster"],
                                                                       file_dic_collection[file_dic]["n"]), marker="x", s=markersize)
            else:
                ax1.scatter(file_dic_collection[file_dic]["np"], file_dic_collection[file_dic]["np1"]/file_dic_collection[file_dic]["rhs"],
                            c=colors[0], marker="x", s=markersize)#c=colors[i_file_dic_collection],)
            """
            ##########

            ##########
            # parallel efficiency
            """
            if file_dic_collection[file_dic]["np"] == 1:
                ax1.scatter(file_dic_collection[file_dic]["np"], file_dic_collection[file_dic]["np1"]/file_dic_collection[file_dic]["rhs"]/file_dic_collection[file_dic]["np"],
                            c=colors[0],#c=colors[i_file_dic_collection],
                            label=r"milupHPC on {}: n = ${}^3$".format(file_dic_collection[file_dic]["cluster"],
                                                                       file_dic_collection[file_dic]["n"]), marker="x", s=markersize)
            else:
                ax1.scatter(file_dic_collection[file_dic]["np"], file_dic_collection[file_dic]["np1"]/file_dic_collection[file_dic]["rhs"]/file_dic_collection[file_dic]["np"],
                            c=colors[0], marker="x", s=markersize)#c=colors[i_file_dic_collection],)
            """
            ##########
            #ax1.text(file_dic_collection[file_dic]["np"] + 0.15, file_dic_collection[file_dic]["rhs"],
            #         "{} ms".format(round(file_dic_collection[file_dic]["rhs"], 2)), verticalalignment='center',
            #         fontsize=10)

    # fig.patch.set_facecolor("black")
    ax1.set_xlabel("number of processes/GPUs")

    ax1.set_ylabel("time in ms")
    # ax1.set_ylabel("parallel efficiency")

    # ax1.set_title("Angular momentum")
    # angMom = np.array(angular_momentum)
    # ax1.plot(time, angMom[:, 0], label="L_x")
    # ax1.plot(time, angMom[:, 1], label="L_y")
    # ax1.plot(time, angMom[:, 2], label="L_z")
    # plt.legend(loc="best")

    # milupHPC sedov N = 126 on de.NBI FRNN 0
    #ax1.scatter(1, 800.0, c="k", label=r"miluphcuda on de.NBI: n=$126^3$")
    #ax1.text(1 + 0.15, 800.0,
    #         "{} ms".format(800.0), verticalalignment='center',
    #         fontsize=10)

    # miluphcuda sedov N = 126 on de.NBI
    ax1.scatter(1, 677.98, c="darkred", label=r"miluphcuda on Tesla V100: n=$126^3$")
    #ax1.text(1 + 0.15, 677.98,
    #         "{} ms".format(677.98), verticalalignment='center',
    #         fontsize=10)

    #ideal = np.linspace(1, 5, 200)
    #ax1.plot(ideal, ideal, c="red", label="ideal")

    # ax1.set_xlim([0.5, 4.5])
    # ax1.set_xlim([0.5, 16.5])
    ax1.set_xlim([0.5, 32.5])
    # ax1.set_ylim([0.2, 1.1])
    # ax1.set_xticks([1, 2, 4])
    # ax1.set_xticks([1, 2, 4, 8, 16])
    ax1.set_xticks([1, 2, 4, 8, 16, 32])
    y_max = ax1.get_ylim()
    #ax1.set_ylim([0, y_max[1]])

    ax1.grid(alpha=0.4, color='grey')

    ax1.legend(loc="best")
    #plt.show()

    fig.tight_layout()
    plt.savefig("cluster_comparison/figures/sedov_strong_scaling_32.png")
