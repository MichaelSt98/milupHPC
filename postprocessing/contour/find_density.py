open #!/usr/bin/python3

from sys import argv
from math import sqrt
import numpy as np


def main():
    #someone should test whether argv[1] exists
    ifname = argv[1]
    ofname = ifname[:8] + '.density'
    print('Reading file',ifname)

    N = eval(argv[2])
    g = round(N**(1/3)) #number of grid cells, should be even
    print('Number of lines:',N)
    print('Number of grid cells', g)

    x = np.empty(N)
    y = np.empty(N)
    z = np.empty(N)
    rho = np.empty(N)

    with open(ifname,'r') as f:
        i = 0
        for line in f:
            data = line.split('\t')
            x[i] = eval(data[0])
            y[i] = eval(data[1])
            z[i] = eval(data[2])
            rho[i] = eval(data[3])
            i+=1

    print('Read data.')

    #get maximum density and its index
    imax = 0
    rho_max = 0
    for i in range(N):
        if rho[i] > rho_max:
            rho_max = rho[i]
            imax = i

    #get distance of maximum density from center
    d2 = x*x + y*y + z*z #minimum index is at center
    d = sqrt(d2[imax])

    #get central density
    #therefore take the mean density of a central cube
    idx = []
    idx.append(g/2 + g/2 * g + g/2 * g * g)
    idx.append((g/2-1) + g/2 * g + g/2 * g * g)
    idx.append(g/2 + (g/2-1) * g + g/2 * g * g)
    idx.append(g/2 + g/2 * g + (g/2-1) * g * g)
    idx.append((g/2-1) + (g/2-1) * g + g/2 * g * g)
    idx.append((g/2-1) + g/2 * g + (g/2-1) * g * g)
    idx.append(g/2 + (g/2-1) * g + (g/2-1) * g * g)
    idx.append((g/2-1) + (g/2-1) * g + (g/2-1) * g * g)
    rho_central = np.mean(np.take(rho, idx))

    #print results to file
    with open(ofname,'w') as f:
        print('Writing file',ofname)
        print(rho_central,rho_max,d,file=f)



if __name__ == '__main__':
    main()
