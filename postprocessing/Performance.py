from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import h5py

def get_dataset_keys(f):
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys


class Performance:

    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.numProcesses = len(data[0])

    def get_data(self, proc=0):
        if 0 <= proc < self.numProcesses:
            return [elem[proc] for elem in self.data]
        else:
            print("Only {} processes available!".format(self.numProcesses))

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

    f = h5py.File('../log/performance.h5', 'r')

    keys = get_dataset_keys(f)
    print(keys)

    performance = {}

    for key in keys:
        if "time" in key:
            print(key)
            name = key.replace("time/", "")
            performance[key] = Performance(name, f[key][:])
            if key == "time/rhsElapsed":
                print(performance[key].get_data_average())

    f.close()

    rhsElapsedPerStep = performance["time/rhsElapsed"].get_data_max()
    print("rhs elapsed per step: {}".format(rhsElapsedPerStep))
    fig, ax1 = plt.subplots(nrows=1, sharex=True)
    ax1.plot(np.arange(0, len(rhsElapsedPerStep)), rhsElapsedPerStep)
    plt.show()

    for key, value in performance.items():
        print("{}: {}".format(key, value.average()))

    stack = []
    bar = []
    labels = []

    #relevantKeys = ["time/loadBalancing", "time/reset", "time/boundingBox", "time/assignParticles", "time/tree",
    #                "time/pseudoParticle ", "time/gravity"]  # , "time/sph"]

    # relevantKeys = ["time/tree", "time/gravity", "time/reset", "time/boundingBox", "time/assignParticles",
    #                "time/pseudoParticle"] #, "time/loadBalancing"]


    relevantKeys = [#"time/loadBalancing",
                    "time/reset",
                    "time/removeParticles",
                    "time/boundingBox",
                    "time/assignParticles",
                    "time/tree",
                    "time/pseudoParticle",
                    "time/gravity", 
                    "time/sph"]
                    #"/time/IO"]

    offset = 0

    for i_key, key in enumerate(relevantKeys):
        # if performance[key].get_data_average() != 0.0:
        stack.append(performance[key].get_data_average())
        bar.append(mean(performance[key].average()))
        #offset += bar[i_key]
        labels.append(performance[key].name)


    #np_stack = np.vstack(stack)

    print("stack: {}".format(stack))

    fig, ax = plt.subplots()
    ax.stackplot([i for i in range(len(stack[0]))], stack, labels=labels)
    ax.scatter([i for i in range(len(rhsElapsedPerStep))], rhsElapsedPerStep)
    ax.legend(loc='best')
    plt.show()

    """
    fig, ax = plt.subplots()
    for i_key, key in enumerate(relevantKeys):
        print("key: {}".format(key))
        # if bar[i_key] != 0.0:
        ax.bar([0], bar[i_key], 0.5, bottom=offset, label=labels[i_key]) #, color='b')
        offset += bar[i_key]
    ax.legend(loc='best')
    plt.show()
    """

    rhsElapsed = performance["time/rhsElapsed"].get_data_average()
    print("average rhs elapsed: {}".format(mean(rhsElapsed)))
