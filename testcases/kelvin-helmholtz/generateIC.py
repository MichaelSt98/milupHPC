#!/usr/bin/env python3

import argparse
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

DIM = 2

# EDIT PARAMETERS BELOW
rho2 = 2.
rho1 = 1.
Dy = .025
P = 5./2.
gamma = 5./3.
Drho = .5
dvy0 = .01

def getVelsX(y):
    velsX = np.empty(len(y))
    
    mask1 = y<.25
    velsX[mask1] = -.5+.5*np.exp((y[mask1]-.25)/Dy)
    mask2 = (.25<=y) & (y<.5)
    velsX[mask2] = .5-.5*np.exp((.25-y[mask2])/Dy)
    mask3 = (.5<=y) & (y<.75)
    velsX[mask3] = .5-.5*np.exp((y[mask3]-.75)/Dy)
    mask4 = .75<y
    velsX[mask4] = -.5-.5*np.exp((.75-y[mask4])/Dy)

    return velsX


def getVelsY(x):
    return dvy0*np.sin(4.*np.pi*x)


def getDensities(y):

    densities = np.empty(len(y))
    
    mask1 = y<.25
    densities[mask1] = rho2-Drho*np.exp((y[mask1]-.25)/Dy)
    mask2 = (.25<=y) & (y<.5)
    densities[mask2] = rho1+Drho*np.exp((.25-y[mask2])/Dy)
    mask3 = (.5<=y) & (y<.75)
    densities[mask3] = rho1+Drho*np.exp((y[mask3]-.75)/Dy)
    mask4 = .75<=y
    densities[mask4] = rho2-Drho*np.exp((.75-y[mask4])/Dy)

    return densities
    

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Create an initial condition HDF5 file for a 2D Kelvin-Helmholtz test case.")
    parser.add_argument("--numParticles", "-N", metavar="int", type=int, help="number of particles", required=True)
    
    args = parser.parse_args()

    N = args.numParticles

    # initialize default random generator
    rng = np.random.default_rng(6102003)
    
    outH5 = h5.File("khN{}.h5".format(N), "w")
    print("Generating Kelvin-Helmholtz initial conditions with", N, "particles ...")

    # randomly generate N points in DIM dimensions
    pos = rng.random(size=(N, DIM))
    # set velocities
    vel = np.empty(pos.shape)
    vel[:,0] = getVelsX(pos[:,1])
    vel[:,1] = getVelsY(pos[:,0])
    # set densities
    rho = getDensities(pos[:,1])
    # create material ID
    matId = np.zeros(len(rho), dtype=np.int8)
    # volume is 1
    m = rho/N
    # create specific internal energy
    u = P/((gamma-1.)*rho)

    pos = pos - [.5, .5] # transform for symmetry around origin
    
    outH5.create_dataset("x", data=pos) 
    outH5.create_dataset("v", data=vel)
    outH5.create_dataset("m", data=m)
    outH5.create_dataset("u", data=u)
    outH5.create_dataset("materialId", data=matId)
    
    print("... done. Output written to", outH5.filename)
    outH5.close()

    print("Plotting initial configuration ...")
    # plot densities for verification
    plt.rc('text', usetex=True)
    #plt.rc('text.latex', preamble=r'\usepackage{siunitx}\usepackage{nicefrac}')

    fig, ax = plt.subplots(figsize=(8,6), dpi=200)
    rhoPlt = ax.scatter(pos[:,0], pos[:,1], c=rho, s=1.)
    fig.colorbar(rhoPlt, ax=ax)
    plt.title(r"Color coded density $\rho$")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.tight_layout()
    plt.savefig("khN{}.png".format(N))
    print("... saved to khN{}.png".format(N))

    
