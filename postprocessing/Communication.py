from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import h5py


def get_dataset_keys(f):
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys


class Communication:

    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.numProcesses = len(data[0])

    def get_data(self, proc=0):
        if 0 <= proc < self.numProcesses:
            return [elem[proc] for elem in self.data]
        else:
            print("Only {} processes available!".format(self.numProcesses))

    def get_sum(self):
        sum = []
        for proc in self.data:
            proc_sum = 0
            for elem in proc:
                proc_sum += elem
            sum.append(proc_sum)
        return sum

    def get_data_average(self):
        return [mean(elem) for elem in self.data]

    def get_data_max(self):
        return [max(elem) for elem in self.data]

    def average(self):
        averages = []
        for proc in range(self.numProcesses):
            averages.append(mean(self.get_data(proc)))
        return averages


if __name__ == '__main__':

    # f = h5py.File('../log/performance.h5', 'r')
    # f = h5py.File("/Users/Michi/Desktop/milupHPC/master/binac_data/plummer/milupHPC2/milupHPC/plummer/pl_N1000000_sfc1D_np4/log/performance.h5", "r")
    # f = h5py.File("/Users/Michi/Desktop/milupHPC/master/binac_data/plummer/VerificationRuns/pl_N4096_sfc1D_np4/log/performance.h5", "r")
    f = h5py.File("master/binac_data/sedov/milupHPC/sedov/sedov_N81_sfc1D_np4/log/performance.h5", "r")
    keys = get_dataset_keys(f)
    print(keys)

    communication = {}

    keyType = "sending"
    # keyType = "receiving"

    proc_colors = ["darkgreen", "darkblue", "red", "grey"]
    num_particles = 1000000
    num_procs = 4

    for key in keys:
        if keyType in key and "time" not in key:
            name = key.replace("sending/", "")
            print("name: {}".format(name))
            communication[name] = Communication(name, f[key][:])
            #print("key: {}, sum: {}".format(key, communication[key].get_sum()))

    gravity_particles_send = []
    # gravity_pseudo_particles_send = []
    for proc in range(num_procs):
        gravity_particles_send.append([[] for proc in range(num_procs)])
        # gravity_pseudo_particles_send.append([[] for proc in range(num_procs)])
        # for i_elem, elem in enumerate(communication["gravityParticles"].data):
        for i_elem, elem in enumerate(communication["sph"].data):
            for _proc in range(num_procs):
                if elem[proc][_proc] < 100 and proc != _proc:
                    print("zero for {}".format(i_elem))
                if i_elem != 925:
                    gravity_particles_send[proc][_proc].append(elem[proc][_proc])
                    # gravity_pseudo_particles_send[proc][_proc].append(communication["gravityPseudoParticles"].data[proc][_proc])

    # print(communication["gravityParticles"].data)
    # print(gravity_particles_send)

    fig, axes = plt.subplots(num_procs, num_procs, figsize=(8, 6), dpi=200) #, sharex=True, sharey=True)
    #fig.tight_layout()
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.01, hspace=0.01)
    for proc in range(num_procs):
        for _proc in range(num_procs):
            if proc != _proc:
                # axes[proc][_proc].set_ylim([0, 100])
                axes[proc][_proc].set_ylim([0, 30000])
                # axes[proc][_proc].plot(np.array(gravity_particles_send[proc][_proc])/num_particles * 100., color=proc_colors[_proc])
                axes[proc][_proc].plot(np.array(gravity_particles_send[proc][_proc]), color=proc_colors[_proc])
                # axes[proc][_proc].plot(np.array(gravity_pseudo_particles_send[proc][_proc])/2., color=proc_colors[_proc])
            else:
                #fig.delaxes(axes[proc][_proc])
                # axes[proc][_proc].set_visible(False)
                axes[proc][_proc].set_xticks([])
                axes[proc][_proc].set_yticks([])
                # axes[proc][_proc].axis("off")
            if proc == (num_procs - 1) and _proc == 0:
                axes[proc][_proc].set_xlabel("integration step")
                axes[proc][_proc].set_ylabel("# sent")
            else:
                axes[proc][_proc].set_xticklabels([])
                axes[proc][_proc].set_yticklabels([])
    # axes[3][0].set_yticks([])

    pad = 5 # in points

    cols = ['sending to {}'.format(col) for col in range(0, num_procs)]
    rows = ['proc {}'.format(row) for row in range(0, num_procs)]
    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    ha='center', va='baseline') # size='large',

    for ax, row in zip(axes[:, 0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-pad, 0), #(-axes[num_procs-1][0].yaxis.labelpad - 10 * pad, 0), # -ax.yaxis.labelpad
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    ha='right', va='center') # size='large',
    #fig.tight_layout()

    # plt.show()
    fig.savefig("test.png", bbox_inches="tight")

    f.close()
