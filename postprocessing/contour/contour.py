#!/usr/bin/python3

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import sys

def main():
    x = []
    y = []
    z = []
    rho = []

    with open('bb.47.grid', 'r') as f:
        for line in f:
            data = line.split()
            x.append(float(data[0]))
            y.append(float(data[1]))
            z.append(float(data[2]))
            rho.append(float(data[3]))

    idx = []
    print(len(x))
    for i in range(len(x)):
        if z[i] < 8e12 and z[i] > -8e-12:
            idx.append(i)
    
    print(len(idx))
    if len(idx) == 0:
        sys.exit(1)

    with open('contour.dat', 'w') as f:

        x_arr = np.array(x)[idx]
        y_arr = np.array(y)[idx]
        rho_arr = np.array(rho)[idx]

        print('Plotting...')
        #create contour plot
        xi, yi = np.linspace(x_arr.min(), x_arr.max(), 100), np.linspace(y_arr.min(), y_arr.max(), 100)
        xi, yi = np.meshgrid(xi,yi)

        rhoi = scipy.interpolate.griddata((x_arr,y_arr), rho_arr, (xi, yi), method='linear')
        """
        plt.imshow(rhoi, vmin=rho_arr.min(), vmax=rho_arr.max(), origin='lower', cmap='gist_heat',
                extent=[x_arr.min(), x_arr.max(), y_arr.min(), y_arr.max()])
        plt.colorbar()
        plt.show()
        """
        CS = plt.contour(xi, yi, rhoi)
        plt.clabel(CS, inline=1, fmt='%3.2e')
    plt.show()



if __name__ == '__main__':
    main()
