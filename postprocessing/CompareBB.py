import csv
import matplotlib.pyplot as plt

if __name__ == '__main__':

    files = ["../master/testcases/bb/np1_leb_f/density_evolution.csv",
             "../master/testcases/bb/np2_hil_f/density_evolution.csv",
             "../master/testcases/bb/np2_hil_d/density_evolution.csv"]
    num_processes = [1, 2, 2]
    linestyles = ["dashed", "dashdot", "dotted"]

    t_f = 5.529e11  # free-fall time

    fig, ax = plt.subplots(1, 1)

    for i_file, file in enumerate(files):
        with open(file, newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=';')
            for i_row, row in enumerate(csv_reader):
                print("row: {}: {}".format(i_row, row))
                if i_row == 1:
                    time = [float(elem)/t_f for elem in row]
                if i_row == 2:
                    max_density = [float(elem) for elem in row]

        ax.plot(time, max_density, label="np: {}".format(num_processes[i_file]), linestyle=linestyles[i_file])

    ax.set_ylim([1e-14, 1e-10])
    ax.set_xlim([0, 1.3])
    ax.set_yscale('log')
    ax.legend(loc="best")
    ax.set_xlabel(r"time $t$ $[t_f]$")
    ax.set_ylabel(r"max($\rho$) $[\frac{kg}{m^3}]$")
    plt.show()

