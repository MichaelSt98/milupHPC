from analytical import SedovSolution
from analytical import Sedov

import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py

if __name__ == '__main__':

    filename_1 = "simulation_data/sedov_seagen_470k_1e-2.h5"
    # filename_1 = "simulation_data/sedov_seagen_100k_1e-2.h5"
    # filename = "simulation_data/ts000100.h5"
    filename_2 = "simulation_data/ts000058.h5"

    # data from filename_1
    h5f_1 = h5py.File(filename_1, 'r')
    time_1 = float(h5f_1['time'][0])
    coordinates_1 = h5f_1['x']
    rho_1 = h5f_1['rho']
    pressure_1 = h5f_1['p']
    energy_1 = h5f_1['e']
    x_1 = coordinates_1[:, 0]
    y_1 = coordinates_1[:, 1]
    z_1 = coordinates_1[:, 2]
    r_1 = np.sqrt(x_1**2 + y_1**2 + z_1**2)
    #h5f.close()

    # data from filename_2

    h5f_2 = h5py.File(filename_2, 'r')
    time_2 = float(h5f_2['time'][0])
    coordinates_2 = h5f_2['x']
    rho_2 = h5f_2['rho']
    pressure_2 = h5f_2['p']
    energy_2 = h5f_2['e']
    x_2 = coordinates_2[:, 0]
    y_2 = coordinates_2[:, 1]
    z_2 = coordinates_2[:, 2]
    r_2 = np.sqrt(x_2**2 + y_2**2 + z_2**2)
    #h5f.close()

    # plot
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
    ax1.scatter(r_1, rho_1, c='r', s=0.1, alpha=0.3)
    ax1.scatter(r_2, rho_2, c='k', s=0.1, alpha=0.3)
    ax2.scatter(r_1, pressure_1, c='b', s=0.1, alpha=0.3)
    ax2.scatter(r_2, pressure_2, c='k', s=0.1, alpha=0.3)
    ax2.set_ylim(-1, 20)
    ax3.scatter(r_1, energy_1, c='darkgreen', s=0.1, alpha=0.3)
    ax3.scatter(r_2, energy_2, c='k', s=0.1, alpha=0.3)
    ax3.set_ylim(-10, 1e3)
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax3.legend(loc='best')
    plt.show()
