from analytical import SedovSolution
from analytical import Sedov

import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py

if __name__ == '__main__':

    # filename = "simulation_data/sedov_seagen_470k_1e-2.h5"
    filename = "simulation_data/ts000000.h5"
    # filename = "simulation_data/sedov_balsara.h5"

    h5f = h5py.File(filename, 'r')
    time = float(h5f['time'][0])
    coordinates = h5f['x']
    rho = h5f['rho']
    pressure = h5f['p']
    energy = h5f['e']
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    z = coordinates[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)

    sedov = Sedov(time=time, r_max=0.5)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
    ax1.set_title("t = {:.4f}".format(time))
    ax1.plot(*sedov.compute('rho'), label="rho", color='r')
    ax1.scatter(r, rho, c='r', s=0.1, alpha=0.3)
    ax2.plot(*sedov.compute('pressure'), label="pressure", color='b')
    ax2.scatter(r, pressure, c='b', s=0.1, alpha=0.3)
    ax2.set_ylim(0, 20)
    ax3.plot(*sedov.compute('internal_energy'), label="internal energy", color='darkgreen')
    ax3.scatter(r, energy, c='darkgreen', s=2.8, alpha=0.3, marker='x')

    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax3.legend(loc='best')
    plt.show()