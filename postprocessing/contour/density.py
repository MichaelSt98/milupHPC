#!/usr/bin/python3

import os
import numpy as np


def main():
    N = 284
    dt = 2207640000

    t = []
    rho_central = []
    rho_max = []
    d = []

    for i in range(1,N+1):
        fname = 'gas.'+str(i).zfill(4)+'.grid'
        os.system('./find_density.py '+fname+' `wc -l '+fname+'`')

    for i in range(1,N+1):
        with open('gas.'+str(i).zfill(4)+'.density', 'r') as f:
            line = f.read()
            data = line.split()
            t.append(i*dt)
            rho_central.append(eval(data[0]))
            rho_max.append(eval(data[1]))
            d.append(eval(data[2]))

    with open('den_time.dat', 'w') as of:
        for i in range(N):
            print(t[i],rho_central[i],rho_max[i],d[i],file=of)

    #find the specified data points to compare them to the paper
    a = 31557600  #1a [s]
    T = [1.96e4 * a, 2.09e4 * a]
    idx = []
    for time in T:
        delta_t = (np.array(t) - time)**2
        idx.append(np.argmin(delta_t))

    with open('compare.dat', 'w') as f:
        for i in idx:
            print("%3.2e %3.2e %3.2e %3.2e" % (t[i]/a, rho_central[i], rho_max[i], d[i]), file=f)

        



if __name__ == '__main__':
    main()

