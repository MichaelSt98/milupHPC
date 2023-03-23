#!/usr/bin/env python3

import argparse
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

DIM = 3

# EDIT PARAMETERS BELOW
rho = 1.
explosion_energy = 1.
u_floor = 1e-6

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Create an initial condition HDF5 file for a 3D Sedov blast wave test case with particles initialized on a grid and the energy seeded into a single particle. This testcase is used to test the meshless methods MFV and MFM (Hopkins, 2015).")
    parser.add_argument("--numParticles", "-N", metavar="int", type=int, help="cube root of total number of particles", required=True)
    
    args = parser.parse_args()

    Ncbrt = args.numParticles
    N = Ncbrt**DIM
    
    outH5 = h5.File("sedov_N{}.h5".format(Ncbrt), "w")
    print("Generating Sedov blast wave initial conditions with", N, "particles ...")

    pos = np.empty((N, DIM))
    xv = np.linspace(-.5, .5, int(Ncbrt), endpoint=True)
    yv = np.linspace(-.5, .5, int(Ncbrt), endpoint=True)
    zv = np.linspace(-.5, .5, int(Ncbrt), endpoint=True)

    i = 0
    for x in xv:
        for y in yv:
            for z in zv:
                pos[i,0] = x
                pos[i,1] = y
                pos[i,2] = z
                i += 1
    # set velocities
    vel = np.zeros(pos.shape)

    # equal mass particles
    volume = 1.
    m = volume/float(N) * np.ones(N)
    
    # create material ID
    matId = np.zeros(N, dtype=np.int8)

    # create specific internal energy
    u = np.ones(N) * u_floor
    u[int(N/2)] = explosion_energy/m[int(N/2)]
    
    outH5.create_dataset("x", data=pos) 
    outH5.create_dataset("v", data=vel)
    outH5.create_dataset("m", data=m)
    outH5.create_dataset("u", data=u)
    outH5.create_dataset("materialId", data=matId)
    
    print("... done. Output written to", outH5.filename)
    outH5.close()

    
