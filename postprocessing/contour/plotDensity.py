#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np

#plt.xkcd()

t2M, rhomax2M = np.loadtxt('den_time128.dat', usecols=(0,2), unpack=True)
t8k, rhomax8k = np.loadtxt('../bb-8000p/den_time.dat', usecols=(0,2), unpack=True)
t8kn, rhomax8kn = np.loadtxt('../bb-8000p-normal/den_time.dat', usecols=(0,2), unpack=True)
tbb, rhomaxbb = np.loadtxt('bb_density1', usecols=(0,2), unpack=True)
t200k, rhomax200k = np.loadtxt('../den_time.dat', usecols=(0,2), unpack=True)


tff = 5.519e11

#plt.xlabel(r'$t\, [t_{ff}]$')
#plt.ylabel(r'$\rho_{max}\, [kg / m^3]$')
plt.xlabel('time in units of free-fall time')
plt.ylabel('maximal density in kg/m$^3$')
plt.yscale('log')

plt.plot(t2M/tff, rhomax2M, label='2M particles')
plt.plot(t200k/tff, rhomax200k, label='200k particles')
plt.plot(t8kn/tff, rhomax8kn, label='8k particles')
plt.plot(t8k/tff, rhomax8k, label='8k particles, gravitational smoothing')
plt.plot(tbb/tff, rhomaxbb, 'rx', label='Boss & Bodenheimer (1979)')
plt.legend(frameon=False, loc=2, numpoints=1)
plt.show()
