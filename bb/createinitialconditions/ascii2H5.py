
import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

if __name__ == '__main__':

    # x, y, z, vx, vy, vz, mass, materialId
    inputFile = "gas.output"

    outputFile = inputFile + ".h5"

    data = np.loadtxt(inputFile)
    x = np.array([[elem[0], elem[1], elem[2]] for elem in data])
    v = np.array([[elem[3], elem[4], elem[5]] for elem in data])
    mass = np.array([elem[6] for elem in data])
    materialId = np.array([int(elem[7]) for elem in data])

    print("x: {}".format(x.shape))
    print("v: {}".format(v.shape))
    print("mass: {}".format(mass.shape))
    print("materialId: {}".format(materialId.shape))

    print(materialId)

    hf = h5py.File(outputFile, "w")

    hf.create_dataset("x", data=x)
    hf.create_dataset("v", data=v)
    hf.create_dataset("m", data=mass)
    hf.create_dataset("materialId", data=materialId)

    hf.close()




