#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py
from analytical import Sedov

if len(sys.argv) != 2:
    print("Usage: %s <hdf5 output file>" % sys.argv[0])
    sys.exit(1)
            
filename = sys.argv[1]
print("current file: {}".format(filename))


h5f = h5py.File(filename, 'r')
coordinates = h5f['x']
rho = h5f['rho']
pressure = h5f['p']
x = coordinates[:,0]
y = coordinates[:,1]
z = coordinates[:,2]
r = np.sqrt(x**2 + y**2 + z**2)
energy = h5f['e']
time = float(h5f['time'][0])

sedov = Sedov(time=time, r_max=0.5)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
plt.subplots_adjust(hspace=0.1)
ax1.plot(*sedov.compute('rho'), color='r')
ax1.scatter(r, rho, c='r', s=0.1, alpha=0.3)
ax1.set_title("Time t = %.2e" % time)
ax1.set_ylabel(r'$\varrho$')
ax1.set_xlim(0,0.5)
ax1.set_ylim(0,4.0)
ax2.plot(*sedov.compute('pressure'), color='b')
ax2.scatter(r, pressure, c='b', s=0.1, alpha=0.3)
ax2.set_ylabel(r'$p$')
ax2.set_xlim(0,0.5)
ax2.set_ylim(0,25)
ax3.plot(*sedov.compute('internal_energy'), color='darkgreen')
ax3.scatter(r, energy, c='darkgreen', s=0.1, alpha=0.3)
ax3.set_xlabel(r'$r$')
ax3.set_ylabel(r'$e$')
ax3.set_xlim(0,0.5)
fig.savefig(filename + ".png")
h5f.close()

