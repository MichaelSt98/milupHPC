#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py
from analytical import Sedov
import numpy as np
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
import argparse
import csv
import os


class SedovSolution(object):
    """
    see: [The Sedov self-similar point blast solutions in nonuniform media](https://link.springer.com/content/pdf/10.1007/BF01414626.pdf)

    rho0 = A*r**(-w)
    R_s = ((e * t**2)/(alpha * A))**(1/(nu + 2 - w))
    """
    def __init__(self, e, rho, gamma=4/3., nu=3, w=0., epsilon=1e-50):

        # w = 0 --> uniform background

        if not any(nu == np.array([1, 2, 3])):
            raise ValueError("nu (dimension of problem) need to be 1, 2 or 3!")

        self._epsilon = epsilon
        self._e = e
        self._gamma = gamma

        self._rho0 = rho
        self._rho1 = ((gamma + 1.)/(gamma - 1.)) * rho

        self._nDim = nu
        self._w = w

        # Constants for the parametric equations:
        self.w1 = (3*nu - 2 + gamma*(2-nu))/(gamma + 1.)
        self.w2 = (2.*(gamma-1) + nu)/gamma
        self.w3 = nu*(2.-gamma)

        self.b0 = 1./(nu*gamma - nu + 2)
        self.b2 = (gamma-1.)/(gamma*(self.w2-w))
        self.b3 = (nu-w)/(float(gamma)*(self.w2-w))
        self.b5 = (2.*nu-w*(gamma+1))/(self.w3-w)
        self.b6 = 2./(nu+2-w)
        self.b1 = self.b2 + (gamma+1.)*self.b0 - self.b6
        self.b4 = self.b1*(nu-w)*(nu+2.-w)/(self.w3-w)
        self.b7 = w*self.b6
        self.b8 = nu*self.b6

        self.c0 = 2*(nu-1)*np.pi + (nu-2)*(nu-3)  # simple interpolation of correct function (only for nu=1,2,3)
        self.c5 = 2./(gamma - 1)
        self.c6 = (gamma + 1)/2.
        self.c1 = self.c5*gamma
        self.c2 = self.c6/gamma
        self.c3 = (nu*gamma - nu + 2.)/((self.w1-w)*self.c6)
        self.c4 = (nu + 2. - w)*self.b0*self.c6

        # Characterize the solution
        f_min = self.c2 if self.w1 > w else self.c6

        f = np.logspace(np.log10(f_min), 0, 1e5)

        # Sort the etas for our interpolation function
        eta = self.parametrized_eta(f)
        f = f[eta.argsort()]
        eta.sort()

        d = self.parametrized_d(f)
        p = self.parametrized_p(f)
        v = self.parametrized_v(f)

        # If min(eta) != 0 then all values for eta < min(eta) = 0
        if eta[0] > 0:
            e01 = [0., eta[0]*(1-1e-10)]
            d01 = [0., 0]
            p01 = [0., 0]
            v01 = [0., 0]

            eta = np.concatenate([np.array(e01), eta])
            d = np.concatenate([np.array(d01), d])
            p = np.concatenate([np.array(p01), p])
            v = np.concatenate([np.array(v01), v])

        # Set up our interpolation functions
        self._d = interp1d(eta, d, bounds_error=False, fill_value=1./self._rho1)
        self._p = interp1d(eta, p, bounds_error=False, fill_value=0.)
        self._v = interp1d(eta, v, bounds_error=False, fill_value=0.)

        # Finally Calculate the normalization of R_s:
        integral = eta**(nu-1)*(d*v**2 + p)
        integral = 0.5 * (integral[1:] + integral[:-1])
        d_eta = (eta[1:] - eta[:-1])

        # calculate integral and multiply by factor
        alpha = (integral*d_eta).sum() * (8*self.c0)/((gamma**2-1.)*(nu+2.-w)**2)
        self._c = (1./alpha)**(1./(nu+2-w))

    def parametrized_eta(self, var):
        return (var**-self.b6)*((self.c1*(var-self.c2))**self.b2)*((self.c3*(self.c4-var))**(-self.b1))

    def parametrized_d(self, var):
        return (var**-self.b7)*((self.c1*(var-self.c2))**(self.b3-self._w*self.b2)) * \
               ((self.c3*(self.c4-var))**(self.b4+self._w*self.b1))*((self.c5*(self.c6-var))**-self.b5)

    def parametrized_p(self, var):
        return (var**self.b8)*((self.c3*(self.c4-var))**(self.b4+(self._w-2)*self.b1)) * \
               ((self.c5*(self.c6-var))**(1-self.b5))

    def parametrized_v(self, var):
        return self.parametrized_eta(var) * var

    # Shock properties
    def shock_radius(self, t):
        # outer radius at time t
        t = np.maximum(t, self._epsilon)
        return self._c * (self.e*t**2/self.rho0)**(1./(self._nDim + 2-self._w))

    def shock_velocity(self, t):
        # velocity of the shock wave
        t = np.maximum(t, self._epsilon)
        return (2./(self._nDim+2-self._w)) * self.shock_radius(t) / t

    def post_shock_pressure(self, t):
        # post shock pressure
        return (2./(self.gamma+1))*self.rho0*self.shock_velocity(t)**2

    @property
    def post_shock_density(self, t=0):
        # post shock density
        return self._rho1

    def rho(self, r, t):
        # density at radius r and time t
        eta = r/self.shock_radius(t)
        return self.post_shock_density*self._d(eta)

    def pressure(self, r, t):
        # pressure at radius r and time t
        eta = r/self.shock_radius(t)
        return self.post_shock_pressure(t)*self._p(eta)

    def velocity(self, r, t):
        # velocity at radius r, and time t
        eta = r/self.shock_radius(t)
        return self._v(eta)*(2/(self.gamma+1))*self.shock_velocity(t)

    def internal_energy(self, r, t):
        # internal energy at radius r and time t
        return self.pressure(r, t)/(self.rho(r, t)*(self.gamma-1))

    def entropy(self, r, t):
        # entropy at radius, r, and time, t
        return self.pressure(r, t)/self.rho(r, t)**self.gamma

    # Other properties
    @property
    def e(self):
        # total energy
        return self._e

    @property
    def gamma(self):
        # ratio of specific heats
        return self._gamma

    @property
    def rho0(self):
        # background density
        return self._rho0


class Sedov(object):
    """
    Analytical solution for the sedov blast wave problem
    """
    def __init__(self, time, r_max):

        rho0 = 1.0  #1
        e0 = 1.0 #1e5
        gamma = 5/3. #1.666667 #1.333
        w = 0  # Power law index
        n_dim = 3

        self.sol = SedovSolution(e0, rho0, gamma=gamma, w=w, nu=n_dim)
        self.r = np.linspace(0, r_max, 1001)[1:]
        self.t = time

        print("Shock radius: {}".format(self.sol.shock_radius(self.t)))

    def compute(self, y):
        return map(self.determine, ['r', y])

    def determine(self, x):
        if x == 'r':
            return self.r
        elif x == 'velocity':
            return self.sol.velocity(self.r, self.t)
        elif x == 'rho':
            return self.sol.rho(self.r, self.t)
        elif x == 'pressure':
            return self.sol.pressure(self.r, self.t)
        elif x == 'internal_energy':
            return self.sol.internal_energy(self.r, self.t)
        else:
            raise AttributeError("Sedov solution for variable %s not known"%x)


if __name__ == '__main__':

    # if len(sys.argv) != 2:
    #     print("Usage: %s <hdf5 output file>" % sys.argv[0])
    #     sys.exit(1)
    # filename = sys.argv[1]
    # print("current file: {}".format(filename))

    colors = \
        {
            "rho": "r",
            "pressure": "b",
            "energy": "darkgreen",
            "noi": "darkgrey"
        }

    plot_analytical_solution = False
    write_to_csv = False

    parser = argparse.ArgumentParser(description="Plotting Sedov simulation data and analytical solution.")
    parser.add_argument("--input", "-i",  metavar="str", type=str, help="input file", required=True)
    parser.add_argument("--output", "-o", metavar="str", type=str, help="output directory", required=True)
    parser.add_argument('--analytical', "-a", action='store_true')
    parser.add_argument('--csv', "-c", action='store_true')
    parser.add_argument("--plot_type", "-p", metavar="int", type=int,
                        help="plot type ([0]: rho; [1]: rho, p, e;[2]: rho, p, e, noi)", required=True)
    parser.add_argument("--radius", "-r", type=float, help="max(radius)", default=0.5)
    args = parser.parse_args()

    r_max = args.radius

    if args.analytical:
        plot_analytical_solution = True
    if args.csv:
        write_to_csv = True

    filename = args.input
    if ".h5" in filename and ".csv" not in filename:
        write_to_csv = True
        h5f = h5py.File(filename, 'r')
        coordinates = h5f['x']
        rho = h5f['rho']
        pressure = h5f['p']
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        z = coordinates[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        energy = h5f['e']
        noi = h5f['noi']
        time = float(h5f['time'][0])

        if plot_analytical_solution:
            sedov = Sedov(time=time, r_max=r_max)
            r_analytical, rho_analytical = sedov.compute('rho')
            _, pressure_analytical = sedov.compute('pressure')
            _, energy_analytical = sedov.compute('internal_energy')

    elif ".csv" in filename:
        with open(filename, newline='\n') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar='|')
            for i_row, row in enumerate(reader):
                print("i_row: {}, row[0]: {}".format(i_row, row[0]))
                if i_row == 0:
                    header = row
                if i_row == 1:
                    time = float(row[0])
                if i_row == 2:
                    r = row
                    r = [float(i) for i in r]
                    print("r: {}".format(r[0:10]))
                if i_row == 3:
                    rho = row
                    rho = [float(i) for i in rho]
                    print("rho: {}".format(rho[0:10]))
                if i_row == 4:
                    pressure = row
                    pressure = [float(i) for i in pressure]
                if i_row == 5:
                    energy = row
                    energy = [float(i) for i in energy]
                if i_row == 6:
                    noi = row
                    noi = [float(i) for i in noi]
                if i_row == 7:
                    r_analytical = row
                    r_analytical = [float(i) for i in r_analytical]
                    print("r_analytical: {}".format(r_analytical[0:10]))
                if i_row == 8:
                    rho_analytical = row
                    rho_analytical = [float(i) for i in rho_analytical]
                    print("rho_analytical: {}".format(rho_analytical[0:10]))
                if i_row == 9:
                    pressure_analytical = row
                    pressure_analytical = [float(i) for i in pressure_analytical]
                if i_row == 10:
                    energy_analytical = row
                    energy_analytical = [float(i) for i in energy_analytical]
    else:
        sys.exit(1)

    # font = {'family': 'normal', 'weight': 'bold', 'size': 18}
    # font = {'family': 'normal', 'size': 18}
    font = {'size': 15}
    matplotlib.rc('font', **font)

    if args.plot_type == 0:
        fig, (ax1) = plt.subplots(nrows=1, sharex=True)
        plt.subplots_adjust(hspace=0.1)
        if plot_analytical_solution:
            ax1.plot(r_analytical, rho_analytical, color=colors["rho"])
        ax1.scatter(r, rho, c=colors["rho"], s=0.1, alpha=0.3)
        ax1.set_title("Time t = %.2e" % float(time))
        ax1.set_xlabel(r'$r$')
        ax1.set_ylabel(r'$\rho$')
        ax1.set_xlim(0, r_max)
        ax1.set_ylim(0, 4.0)
    elif args.plot_type == 1:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
        plt.subplots_adjust(hspace=0.1)
        if plot_analytical_solution:
            ax1.plot(r_analytical, rho_analytical, color=colors["rho"])
        ax1.scatter(r, rho, c=colors["rho"], s=0.1, alpha=0.3)
        ax1.set_title("Time t = %.2e" % float(time))
        ax1.set_ylabel(r'$\rho$')
        ax1.set_xlim(0, r_max)
        ax1.set_ylim(0, 4.0)
        if plot_analytical_solution:
            ax2.plot(r_analytical, pressure_analytical, color=colors["pressure"])
        ax2.scatter(r, pressure, c=colors["pressure"], s=0.1, alpha=0.3)
        ax2.set_ylabel(r'$p$')
        ax2.set_xlim(0, r_max)
        ax2.set_ylim(0, 25)
        if plot_analytical_solution:
            ax3.plot(r_analytical, energy_analytical, color=colors["energy"])
        ax3.scatter(r, energy, c=colors["energy"], s=0.1, alpha=0.3)
        ax3.set_xlabel(r'$r$')
        ax3.set_ylabel(r'$e$')
        ax3.set_xlim(0, r_max)
    elif args.plot_type == 2:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True)
        plt.subplots_adjust(hspace=0.1)
        if plot_analytical_solution:
            ax1.plot(r_analytical, rho_analytical, color=colors["rho"])
        ax1.scatter(r, rho, c=colors["rho"], s=0.1, alpha=0.3)
        ax1.set_title("Time t = %.2e" % float(time))
        ax1.set_ylabel(r'$\rho$')
        ax1.set_xlim(0, r_max)
        ax1.set_ylim(0, 4.0)
        if plot_analytical_solution:
            ax2.plot(r_analytical, pressure_analytical, color=colors["pressure"])
        ax2.scatter(r, pressure, c=colors["pressure"], s=0.1, alpha=0.3)
        ax2.set_ylabel(r'$p$')
        ax2.set_xlim(0, r_max)
        ax2.set_ylim(0, 25)
        if plot_analytical_solution:
            ax3.plot(r_analytical, energy_analytical, color=colors["energy"])
        ax3.scatter(r, energy, c=colors["energy"], s=0.1, alpha=0.3)
        ax3.set_ylabel(r'$e$')
        ax3.set_xlim(0, r_max)
        ax4.scatter(r, noi, c=colors["noi"], s=0.1, alpha=0.3)
        ax4.set_xlabel(r'$r$')
        ax4.set_ylabel(r'#interactions')
        ax4.set_xlim(0, r_max)
    else:
        sys.exit(1)

    if write_to_csv and ".h5" not in filename and ".csv" in filename:
        with open("{}{}.png.csv".format(args.output, os.path.basename(filename).replace(".png.csv", "")), 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=";")
            header = ["t", "r", "rho", "pressure", "energy", "noi", "r_analytical", "analytical_rho", "analytical_pressure",
                      "analytical_energy"]
            csv_writer.writerow(header)
            csv_writer.writerow([time])
            csv_writer.writerow(r)
            csv_writer.writerow(rho)
            csv_writer.writerow(pressure)
            csv_writer.writerow(energy)
            csv_writer.writerow(noi)
            csv_writer.writerow(r_analytical)
            csv_writer.writerow(rho_analytical)
            csv_writer.writerow(pressure_analytical)
            csv_writer.writerow(energy_analytical)

    fig.savefig("{}{}.png".format(args.output, os.path.basename(filename)))

    if ".h5" in filename and ".csv" not in filename:
        h5f.close()
