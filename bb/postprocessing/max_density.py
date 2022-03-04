import numpy as np
import h5py
import glob
import matplotlib.pyplot as plt

def get_dataset_keys(f):
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys

if __name__ == '__main__':

    directory = "../../../output/"
    files = glob.glob(directory + "ts*.h5")
    print("files: {}".format(sorted(files)))
    sorted_files = sorted(files)

    max_density = []
    time = []
    for file in sorted_files:
        f = h5py.File(file, 'r')
        max_density.append(np.array(f["rho"][:]).max())
        time.append(f["time"])

    print("max densities: {}".format(max_density))
    time = np.array(time)
    t_f = 5.529e11
    time_tf = time/t_f

    """
    plt.figure()
    plt.plot(time_tf, max_density)
    plt.ylim([1e-14, 1e-10])
    plt.yscale('log')
    plt.savefig("density_evolution.png")
    """

    fig, (ax1) = plt.subplots(nrows=1, sharex=True)
    ax1.plot(time_tf, max_density)
    ax1.set_title("max(density) evolution")
    ax1.set_ylim([1e-14, 1e-10])
    ax1.set_xlim([0, 1.2])
    ax1.set_yscale('log')
    ax1.set_xlabel(r"time $t$ $[t_f]$")
    ax1.set_ylabel(r"$\rho$ $[\frac{kg}{m^3}]$")
    fig.savefig("density_evolution.png")


