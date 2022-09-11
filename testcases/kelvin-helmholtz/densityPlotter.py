#!/usr/bin/env python3

import argparse
import pathlib
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

def createPlot(h5File, outDir):
    data = h5.File(h5File, 'r')
    pos = data["x"][:]
    rho = data["rho"][()]
    fig, ax = plt.subplots(figsize=(8,6), dpi=200)
    rhoPlt = ax.scatter(pos[:,0], pos[:,1], c=rho, s=1.)
    fig.colorbar(rhoPlt, ax=ax)
    plt.title(r"Color coded density $\rho$")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.tight_layout()
    print("Saving figure to", outDir + "/" + pathlib.Path(h5File).stem + ".png")
    plt.savefig(outDir + "/" + pathlib.Path(h5File).stem + ".png")
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Plot density of results from Kelvin-Helmholtz test case.")
    parser.add_argument("--simOutputDir", "-d", metavar="string", type=str, help="output directory of simulation", required=True)
    parser.add_argument("--outDir", "-o", metavar="string", type=str, help="output directory for generated plots", default="output")

    args = parser.parse_args()

    plt.rc('text', usetex=True)
    
    print("Examining files in", args.simOutputDir, "...")
    
    for h5File in pathlib.Path(args.simOutputDir).glob('*.h5'):
        print("\t", h5File)
        createPlot(h5File, args.outDir)

    print("... done.")
    
