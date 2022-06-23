import csv
import matplotlib.pyplot as plt
import re

if __name__ == '__main__':

    #files = ["pl_N4096_sfc0D_np2",
    #         "pl_N4096_sfc0F_np2",
    #         "pl_N4096_sfc0D_np4",
    #         "pl_N4096_sfc1D_np4",
    #         "pl_N4096_sfc1F_np2",
    #         "pl_N4096_sfc1D_np2",
    #         "pl_N4096_sfc0F_np1"]

    files = ["pl_N4096_sfc0F_np1"]

    data_dic = {}

    for file in files:
        per_file_data_dic = {}
        with open("cluster_comparison/binac/{}_mass_quantiles.csv".format(file), "r") as csvFile:
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

    fig, ax1 = plt.subplots(figsize=(12, 9), dpi=200)
    # fig.patch.set_facecolor("black")
    ax1.set_xlabel(r"time $t$")

    color = "k"  # "darkgrey"
    for file in files:
        # (pl_N)([0-9]+)(_sfc)([0-9]+)(([A-Z]))(_np)(([0-9]+))
        regex = re.search(r"(pl_N)([0-9]+)(_sfc)([0-9]+)(([A-Z]))(_np)(([0-9]+))", file).groups()
        label = "np: {}, sfc: {}, lB: {}".format(regex[7], regex[3], regex[4])

        ax1.plot(data_dic[file]["time"], data_dic[file]["quantiles_0"], label=label, linestyle="dotted", linewidth=2.0)
        ax1.plot(data_dic[file]["time"], data_dic[file]["quantiles_1"], linestyle="dashed", linewidth=2.0)
        ax1.plot(data_dic[file]["time"], data_dic[file]["quantiles_2"], linestyle="dashdot", linewidth=2.0)

    # Shrink current axis's height by 10% on the bottom
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=4)
    # ax1.legend(loc="best")
    ax1.set_ylabel(r"radius $r$ containing a quantile of the total mass $M$")
    ax1.set_ylim([0.01, 0.7])

    plt.show()
