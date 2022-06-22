import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    num_processes = [2, 4, 8] #, 16, 32]

    rhs_sfc0F_np1 = 725.487
    sf_sfc0F_np1 = 0.0
    gf_sfc0F_np1 = 517.44
    sending_sfc0F_np1 = 0.0

    rhs_master = [1123.39, 1968.42]
    sf_master = [473.21, 1409.15]
    gf_master = [417.78, 337.51]
    sending_master = [10.53, 17.65]

    rhs_brute_force = [767.28, 1202.19]
    sf_brute_force = [137.32, 684.43]
    gf_brute_force = [416.70, 337.92]
    sending_brute_force = [5.0, 16.10]

    rhs_brute_force_termination = [647.27, 573.74, 922.33]
    sf_brute_force_termination = [15.55, 58.38, 258.91]
    gf_brute_force_termination = [419.32, 338.18, 307.31]
    sending_brute_force_termination = [4.54, 14.9, 133.09]

    rhs_brute_force_termination_skip_level_12 = [718.14, 734.61, 2151.48]
    sf_brute_force_termination_skip_level_12 = [10.4, 18.29, 34.669]
    gf_brute_force_termination_skip_level_12 = [432.35, 365.63, 331.64]
    sending_brute_force_termination_skip_level_12 = [15.19, 104.5, 1487.05]

    rhs_brute_force_termination_skip_level_6 = [644.25, 551.83, 750.19]
    sf_brute_force_termination_skip_level_6 = [13.82, 32.83, 65.44]
    gf_brute_force_termination_skip_level_6 = [417.67, 341.33, 306.84]
    sending_brute_force_termination_skip_level_6 = [4.72, 15.28, 181.61]


    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(7, 5), dpi=200)

    ax1.scatter(1, rhs_sfc0F_np1, marker="o", color="black", label="Single-GPU")
    ax1.hlines(y=rhs_sfc0F_np1, xmin=0.75, xmax=8.25, linewidth=1, color="black", alpha=0.3, linestyle="--")

    ax1.text(0.75, rhs_sfc0F_np1 + 100, str(rhs_sfc0F_np1) + " ms", fontsize=8) #verticalalignment='top')#, bbox=props)

    ax2.scatter(1, sf_sfc0F_np1, marker="o", color="black", label="Single-GPU")

    ax3.scatter(1, gf_sfc0F_np1, marker="o", color="black", label="Single-GPU")
    # ax2.hlines(y=sf_sfc0F_np1, xmin=0.75, xmax=8.25, linewidth=1, color="black", alpha=0.3, linestyle="--")

    color = "salmon"
    for i, elapsed in enumerate(rhs_master):
        if i == 0:
            ax1.scatter(num_processes[i], elapsed, marker="x", color=color, label="Master version")
            ax2.scatter(num_processes[i], sf_master[i], marker="x", color=color, label="Master version")
            ax3.scatter(num_processes[i], gf_master[i], marker="x", color=color, label="Master version")
            ax4.scatter(num_processes[i], sending_master[i], marker="x", color=color, label="Master version")
        else:
            ax1.scatter(num_processes[i], elapsed, marker="x", color=color)
            ax2.scatter(num_processes[i], sf_master[i], marker="x", color=color)
            ax3.scatter(num_processes[i], gf_master[i], marker="x", color=color)
            ax4.scatter(num_processes[i], sending_master[i], marker="x", color=color)

    color = "cornflowerblue"
    for i, elapsed in enumerate(rhs_brute_force):
        if i == 0:
            ax1.scatter(num_processes[i], elapsed, marker="x", color=color, label="brute force")
            ax2.scatter(num_processes[i], sf_brute_force[i], marker="x", color=color, label="brute force")
            ax3.scatter(num_processes[i], gf_brute_force[i], marker="x", color=color, label="brute force")
            ax4.scatter(num_processes[i], sending_brute_force[i], marker="x", color=color, label="brute force")
        else:
            ax1.scatter(num_processes[i], elapsed, marker="x", color=color)
            ax2.scatter(num_processes[i], sf_brute_force[i], marker="x", color=color)
            ax3.scatter(num_processes[i], gf_brute_force[i], marker="x", color=color)
            ax4.scatter(num_processes[i], sending_brute_force[i], marker="x", color=color)

    color = "navy"
    for i, elapsed in enumerate(rhs_brute_force_termination):
        if i == 0:
            ax1.scatter(num_processes[i], elapsed, marker="x", color=color, label="brute force termination")
            ax2.scatter(num_processes[i], sf_brute_force_termination[i], marker="x", color=color, label="brute force termination")
            ax3.scatter(num_processes[i], gf_brute_force_termination[i], marker="x", color=color, label="brute force termination")
            ax4.scatter(num_processes[i], sending_brute_force_termination[i], marker="x", color=color, label="brute force termination")
        else:
            ax1.scatter(num_processes[i], elapsed, marker="x", color=color)
            ax2.scatter(num_processes[i], sf_brute_force_termination[i], marker="x", color=color)
            ax3.scatter(num_processes[i], gf_brute_force_termination[i], marker="x", color=color)
            ax4.scatter(num_processes[i], sending_brute_force_termination[i], marker="x", color=color)

    color = "limegreen"
    for i, elapsed in enumerate(rhs_brute_force_termination_skip_level_6):
        if i == 0:
            ax1.scatter(num_processes[i], elapsed, marker="+", color=color, label="brute force termination skip level 6")
            ax2.scatter(num_processes[i], sf_brute_force_termination_skip_level_6[i], marker="+", color=color, label="brute force termination skip level 6")
            ax3.scatter(num_processes[i], gf_brute_force_termination_skip_level_6[i], marker="+", color=color, label="brute force termination skip level 6")
            ax4.scatter(num_processes[i], sending_brute_force_termination_skip_level_6[i], marker="+", color=color, label="brute force termination skip level 6")
        else:
            ax1.scatter(num_processes[i], elapsed, marker="+", color=color)
            ax2.scatter(num_processes[i], sf_brute_force_termination_skip_level_6[i], marker="+", color=color)
            ax3.scatter(num_processes[i], gf_brute_force_termination_skip_level_6[i], marker="+", color=color)
            ax4.scatter(num_processes[i], sending_brute_force_termination_skip_level_6[i], marker="+", color=color)

    ax1.text(4.15, rhs_brute_force_termination_skip_level_6[1] - 50, str(rhs_brute_force_termination_skip_level_6[1]) + " ms", fontsize=8) #verticalalignment='top')#, bbox=props)

    color = "darkgreen"
    for i, elapsed in enumerate(rhs_brute_force_termination_skip_level_12):
        if i == 0:
            ax1.scatter(num_processes[i], elapsed, marker="*", color=color, label="brute force termination skip level 12")
            ax2.scatter(num_processes[i], sf_brute_force_termination_skip_level_12[i], marker="*", color=color, label="brute force termination skip level 12")
            ax3.scatter(num_processes[i], gf_brute_force_termination_skip_level_12[i], marker="*", color=color, label="brute force termination skip level 12")
            ax4.scatter(num_processes[i], sending_brute_force_termination_skip_level_12[i], marker="*", color=color, label="brute force termination skip level 12")
        else:
            ax1.scatter(num_processes[i], elapsed, marker="*", color=color)
            ax2.scatter(num_processes[i], sf_brute_force_termination_skip_level_12[i], marker="*", color=color)
            ax3.scatter(num_processes[i], gf_brute_force_termination_skip_level_12[i], marker="*", color=color)
            ax4.scatter(num_processes[i], sending_brute_force_termination_skip_level_12[i], marker="*", color=color)

    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax1.set_xticks([1, 2, 4, 8])
    ax2.set_xticks([1, 2, 4, 8])
    # ax.set_xticklabels([1,4,5], fontsize=12)
    ax1.set_ylabel(r"elapsed rhs() in ms")
    ax2.set_ylabel(r"elapsed symbolicForce() in ms")
    ax3.set_ylabel(r"elapsed grav. force in ms")
    ax4.set_ylabel(r"elapsed sending in ms")
    ax4.set_xlabel(r"number of processes")

    ax1.set_title(r"Plummer $5 \cdot 10^{6}$ particles Lebesgue dynamic load balancing")
    fig.savefig("symbolic_force.png", bbox_inches="tight")
    plt.show()





