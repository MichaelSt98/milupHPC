#!/usr/bin/env python3

import argparse
import pathlib
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

### EDIT BELOW ###
MARKER_SIZE = 7.
AX_LIM_X = (-.5, .5)
AX_LIM_Y = (-.5, .5)

# slice thickness for plotting 
DELTA_SLICE = .02

def createPlot(h5File, outDir, label, xz):
    data = h5.File(h5File, 'r')
    time = data["time"][0]
    pos = data["x"][:]
    pos2d = []
    pltData = data[label][()]
    data2d = []

    if xz:
        for i, pos in enumerate(data["x"]):
            if pos[1] < DELTA_SLICE/2. and pos[1] > -DELTA_SLICE/2.:
                pos2d.append(np.array([pos[0], pos[2]]))
                data2d.append(pltData[i])
    else:
        for i, pos in enumerate(data["x"]):
            if pos[2] < DELTA_SLICE/2. and pos[2] > -DELTA_SLICE/2.:
                pos2d.append(np.array([pos[0], pos[1]]))
                data2d.append(pltData[i])
            
    
    print(len(data2d), "particles in slice")
    pos2d = np.array(pos2d)
    data2d = np.array(data2d)
    
    fig, ax = plt.subplots(figsize=(7,6), dpi=200)
    if label == "u" or label == "e":
        scatterPlt = ax.scatter(pos2d[:,0], pos2d[:,1], c=data2d, s=MARKER_SIZE, norm=LogNorm())
    else:    
        scatterPlt = ax.scatter(pos2d[:,0], pos2d[:,1], c=data2d, s=MARKER_SIZE)

    ax.set_xlim(AX_LIM_X)
    ax.set_ylim(AX_LIM_Y)
        
    if label == "p":
        plt.title(r"Color coded pressure $P$ at $" + f"t = {time:.2f}" + r"$")
    elif label == "u" or label == "e":
        plt.title(r"Color coded internal energy $u$ at $" + f"t = {time:.2f}" + r"$")
    else:
        plt.title(r"Color coded density $\varrho$ at $" + f"t = {time:.2f}" + r"$")

    plt.xlabel("$x$")
    if xz:
        plt.ylabel("$z$")
    else:
        plt.ylabel("$y$")    

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)  

    fig.colorbar(scatterPlt, cax=cax)

    ax.set_aspect('equal')
    
    plt.tight_layout()
    print("Saving figure to", outDir + "/" + label + "_" + pathlib.Path(h5File).stem + ".png")
    plt.savefig(outDir + "/" + label + "_" + pathlib.Path(h5File).stem + ".png")
    plt.close()

    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Plot density of results from Sedov-Taylor test case.")
    parser.add_argument("--simOutputDir", "-d", metavar="string", type=str, help="output directory of simulation")
    parser.add_argument("--outDir", "-o", metavar="string", type=str, help="output directory for generated plots", default="output")
    parser.add_argument("--initFile", "-i", metavar="string", type=str, help="initial conditions file")
    parser.add_argument("--pressure", "-P", action="store_true", help="plot pressure instead of density")
    parser.add_argument("--energy", "-u", action="store_true", help="plot internal energy instead of density")
    parser.add_argument("--xzPlane", "-z", action="store_true", help="plot x-z-slice instead of x-y-slice")
    
    args = parser.parse_args()

    
    sliceXZFlag = args.xzPlane
    
    plt.rc('text', usetex=True)
    plt.rcParams.update({'font.size': 18})
    
    if args.initFile:
        if args.pressure:
            createPlot(args.initFile, args.outDir, "p", sliceXZFlag)
        elif args.energy:
            createPlot(args.initFile, args.outDir, "u", sliceXZFlag)
        else:
            # density is the default
            createPlot(args.initFile, args.outDir, "rho", sliceXZFlag)
    else:

        print("Examining files in", args.simOutputDir, "...")
        
        for h5File in pathlib.Path(args.simOutputDir).glob('*.h5'):
            print("\t", h5File)
            if args.pressure:
                createPlot(h5File, args.outDir, "p", sliceXZFlag)
            elif args.energy:
                createPlot(h5File, args.outDir, "e", sliceXZFlag)
            else:
                # density is the default
                createPlot(h5File, args.outDir, "rho", sliceXZFlag)

    print("... done.")
    
