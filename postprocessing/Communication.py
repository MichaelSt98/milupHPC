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

    f = h5py.File('../log/performance.h5', 'r')

    keys = get_dataset_keys(f)
    print(keys)

    communication = {}

    keyType = "sending"
    # keyType = "receiving"

    for key in keys:
        if keyType in key and "time" not in key:
            name = key.replace("sending/", "")
            communication[key] = Communication(name, f[key][:])
            print("key: {}, sum: {}".format(key, communication[key].get_sum()))

    f.close()
