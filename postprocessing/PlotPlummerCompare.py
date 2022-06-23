#!/usr/bin/env python3

import argparse
import glob
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
import csv

"""
Based on https://github.com/jammartin/ParaLoBstar/blob/main/tools/conservation/main.py
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot conservation of energy and angular momentum for Plummer test case.")
    parser.add_argument("--data_1", "-d", metavar="str", type=str, help="input directory",
                        nargs="?", default="../output")
    parser.add_argument("--data_2", "-f", metavar="str", type=str, help="input directory",
                        nargs="?", default="../output")
    parser.add_argument("--output", "-o", metavar="str", type=str, help="output directory",
                        nargs="?", default="../output")
    parser.add_argument("--angular_momentum", "-L", action="store_true", help="plot angular momentum (defaul: energy and mass)")
    parser.add_argument("--mass_quantiles", "-Q", action="store_true", help="plot 10, 50 and 90 percent mass quantiles (default: energy and mass)")

    args = parser.parse_args()

    time_1 = []
    time_2 = []
    energy_1 = []
    energy_2 = []
    mass_1 = []
    mass_2 = []
    angular_momentum_1 = []
    angular_momentum_2 = []
    mass_quantiles_1 = []
    mass_quantiles_2 = []

    for h5file in sorted(glob.glob(os.path.join(args.data_1, "*.h5")), key=os.path.basename):
        print("Processing ", h5file, " ...")
        data_1 = h5py.File(h5file, 'r')
        time_1.append(data_1["time"][0])
        # energy.append(data["E_tot"][()])

        if args.angular_momentum:
            print("... reading angular momentum ...")
            angular_momentum_1.append(np.array(data_1["L_tot"][:]))

        elif args.mass_quantiles:
            print("... computing mass quantiles ...")
            vecs2com = data_1["x"][:] - data_1["COM"][:]
            radii_1 = np.linalg.norm(vecs2com, axis=1)
            radii_1.sort()
            numParticles = len(data_1["m"])
            # print("NOTE: Only works for equal mass particle distributions!")
            mass_quantiles_1.append(np.array([
                radii_1[int(np.ceil(.1 * numParticles))],
                radii_1[int(np.ceil(.5 * numParticles))],
                radii_1[int(np.ceil(.9 * numParticles))]]))
        else:
            print("... computing mass and reading energy ...")
            mass_1.append(np.sum(data_1["m"][:]))
            #energy.append(data["E_tot"][()])
        
        print("... done.")

    for h5file in sorted(glob.glob(os.path.join(args.data_2, "*.h5")), key=os.path.basename):
        print("Processing ", h5file, " ...")
        data_2 = h5py.File(h5file, 'r')
        print("data_2 time: {}".format(data_2["t"][()]))
        time_2.append(data_2["t"][()])
        # energy.append(data["E_tot"][()])

        if args.angular_momentum:
            print("... reading angular momentum ...")
            angular_momentum_1.append(np.array(data_2["L_tot"][:]))

        elif args.mass_quantiles:
            print("... computing mass quantiles ...")
            vecs2com = data_2["x"][:] - data_2["COM"][:]
            radii_2 = np.linalg.norm(vecs2com, axis=1)
            radii_2.sort()
            numParticles = len(data_2["m"])
            # print("NOTE: Only works for equal mass particle distributions!")
            mass_quantiles_2.append(np.array([
                radii_2[int(np.ceil(.1 * numParticles))],
                radii_2[int(np.ceil(.5 * numParticles))],
                radii_2[int(np.ceil(.9 * numParticles))]]))
        else:
            print("... computing mass and reading energy ...")
            mass_2.append(np.sum(data_2["m"][:]))
            #energy.append(data["E_tot"][()])

        print("... done.")

    # font = {'family': 'normal', 'weight': 'bold', 'size': 18}
    # font = {'family': 'normal', 'size': 18}
    font = {'size': 12}
    matplotlib.rc('font', **font)

    # plt.style.use("dark_background")
    fig, ax1 = plt.subplots(figsize=(12, 9), dpi=200)
    # fig.patch.set_facecolor("black")
    ax1.set_xlabel("Time")
    
    if args.angular_momentum:
        ax1.set_title("Angular momentum")

        angMom = np.array(angular_momentum)
        
        ax1.plot(time, angMom[:, 0], label="L_x")
        ax1.plot(time, angMom[:, 1], label="L_y")
        ax1.plot(time, angMom[:, 2], label="L_z")

        plt.legend(loc="best")
        
        fig.tight_layout()
        plt.savefig("{}angular_momentum.png".format(args.output))

    elif args.mass_quantiles:
        ax1.set_title("Radii containing percentage of total mass")

        quantiles_1 = np.array(mass_quantiles_1)
        quantiles_2 = np.array(mass_quantiles_2)

        print(quantiles_2)

        color_1 = "red"  # "darkgrey"
        color_2 = "darkgreen"
        ax1.plot(time_1, quantiles_1[:, 0], label="milupHPC 10%", color=color_1, linestyle="-", linewidth=2.0)
        ax1.plot(time_2, quantiles_2[:, 0], label="paralobstar 10%", color=color_2, linestyle="-", linewidth=2.0)
        ax1.plot(time_1, quantiles_1[:, 1], label="milupHPC 50%", color=color_1, linestyle="--", linewidth=2.0)
        ax1.plot(time_2, quantiles_2[:, 1], label="paralobstar 50%", color=color_2, linestyle="--", linewidth=2.0)
        ax1.plot(time_1, quantiles_1[:, 2], label="milupHPC 90%", color=color_1, linestyle="-.", linewidth=2.0)
        ax1.plot(time_2, quantiles_2[:, 2], label="paralobstar 90%", color=color_2, linestyle="-.", linewidth=2.0)
        ax1.legend(loc="best")
        ax1.set_ylabel("Radius")
        ax1.set_ylim([0.01, 0.7])

        fig.tight_layout()
        plt.savefig("{}mass_quantiles.png".format(args.output))

        with open("{}mass_quantiles.csv".format(args.output), 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=";")
            header = ["time", "quantiles_0", "quantiles_1", "quantiles_2"]
            csv_writer.writerow(header)
            csv_writer.writerow(time_1)
            csv_writer.writerow(quantiles_1[:, 0])
            csv_writer.writerow(quantiles_1[:, 1])
            csv_writer.writerow(quantiles_1[:, 2])
    else:

        ax1.set_title("Total energy and mass")
        ax1.set_ylabel("Energy")
    
        # ax1.plot(time, energy, "r-", label="E_tot")

        ax2 = ax1.twinx()
        ax2.plot(time_1, mass_1, "b-", label="M")
        ax2.set_ylabel("Mass")

        fig.tight_layout()
        fig.legend()
        plt.savefig("{}energy_mass.png".format(args.output))
