import csv
import matplotlib.pyplot as plt
import re

if __name__ == '__main__':

    #file = "cluster_comparison/binac/pl_N4096_sfc0F_np1_mass_quantiles.csv"
    #file = "cluster_comparison/binac/pl_N4096_sfc0D_np2_mass_quantiles.csv"
    #file = "cluster_comparison/binac/pl_N4096_sfc0D_np4_mass_quantiles.csv"
    #file = "cluster_comparison/binac/pl_N4096_sfc1D_np2_mass_quantiles.csv"
    file = "cluster_comparison/binac/pl_N4096_sfc1D_np4_mass_quantiles.csv"

    #file = "cluster_comparison/binac/long_pl_N10000000_sfc0F_np1_mass_quantiles.csv"
    #file = "cluster_comparison/binac/long_pl_N10000000_sfc1D_np4_mass_quantiles.csv"

    data_dic = {}

    per_file_data_dic = {}
    with open("{}".format(file), "r") as csvFile:
        file_content = csvFile.read()
        file_content = file_content.replace("'", "")

    rows = file_content.splitlines()
    for i_row, row in enumerate(rows):
        if i_row == 0:
            header = row.split(";")
        else:
            per_file_data_dic[header[i_row - 1]] = [float(elem) for elem in row.split(";")]

    # print(per_file_data_dic)
    # for key in per_file_data_dic:
        # print(key)

    data_dic[file] = per_file_data_dic

    fig, ax1 = plt.subplots(figsize=(6, 4), dpi=250)
    # fig.patch.set_facecolor("black")
    ax1.set_xlabel(r"time $t$", fontsize=14)

    color = "k"  # "darkgrey"

    # (pl_N)([0-9]+)(_sfc)([0-9]+)(([A-Z]))(_np)(([0-9]+))
    regex = re.search(r"(pl_N)([0-9]+)(_sfc)([0-9]+)(([A-Z]))(_np)(([0-9]+))", file).groups()
    label = "np: {}, sfc: {}, lB: {}".format(regex[7], regex[3], regex[4])

    ax1.plot(data_dic[file]["time"], data_dic[file]["quantiles_0"], label="10% mass quantile", linestyle="dotted", linewidth=2.0, color="k")
    ax1.plot(data_dic[file]["time"], data_dic[file]["quantiles_1"], label="50% mass quantile", linestyle="dashed", linewidth=2.0, color="k")
    ax1.plot(data_dic[file]["time"], data_dic[file]["quantiles_2"], label="90% mass quantile", linestyle="dashdot", linewidth=2.0, color="k")

    # Shrink current axis's height by 10% on the bottom
    #box = ax1.get_position()
    #ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                     #box.width, box.height * 0.9])

    # Put a legend below current axis
    #ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=4)
    ax1.legend(loc="best")
    ax1.set_ylabel(r"radius $r$", fontsize=14) # containing a quantile of the total mass $M$")
    ax1.set_ylim([0.01, 0.7])

    #plt.show()
    fig.tight_layout()
    plt.savefig("{}".format("mass_quantiles.png"))
