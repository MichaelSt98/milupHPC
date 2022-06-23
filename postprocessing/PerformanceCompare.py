#!/usr/bin/env python3

import argparse
import h5py
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Rhs performance evaluation")
    parser.add_argument("--data1", "-a", metavar="str", type=str, help="input directory",
                        nargs="?", default="../output")
    parser.add_argument("--data2", "-b", metavar="str", type=str, help="input directory",
                        nargs="?", default="../output")
    #parser.add_argument("--output", "-o", metavar="str", type=str, help="output directory",
    #                    nargs="?", default="../output")

    args = parser.parse_args()


    f1 = h5py.File(args.data1, 'r')
    f2 = h5py.File(args.data2, 'r')

    rhs_elapsed1 = np.array(f1["time/rhsElapsed"][:])
    rhs_elapsed2 = np.array(f2["time/rhsElapsed"][:])

    #gravity_symbolicForce = np.array(f["time/gravity_symbolicForce"][:])

    rhs_elapsed_max1 = [np.array(elem).max() for elem in rhs_elapsed1]
    rhs_elapsed_max2 = [np.array(elem).max() for elem in rhs_elapsed2]

    #gravity_symbolicForce_max = [np.array(elem).max() for elem in gravity_symbolicForce]

    # for i_elem, elem in enumerate(rhs_elapsed_max):
    #     print("{}: {} ms".format(i_elem, elem))

    mean_rhs_elapsed1 = np.array(rhs_elapsed_max1).mean()
    mean_rhs_elapsed2 = np.array(rhs_elapsed_max2).mean()

    #mean_gravity_symbolicForce = np.array(gravity_symbolicForce_max).mean()

    print("rhs elapsed average: {} | {}".format(round(mean_rhs_elapsed1, 2), round(mean_rhs_elapsed2, 2)))
    #print("gravity symbolic force average: {}".format(mean_gravity_symbolicForce))

    #groups = list(f.keys())
    keys = f1['time'].keys()

    max_key_length = max([len(key) for key in keys])
    print("max key length: {}".format(max_key_length))

    for key in keys:
        elapsed1 = np.array(f1["time/{}".format(key)][:])
        elapsed2 = np.array(f2["time/{}".format(key)][:])


        elapsed_max1 = [np.array(elem).max() for elem in elapsed1]
        elapsed_max2 = [np.array(elem).max() for elem in elapsed2]

        mean_elapsed1 = np.array(elapsed_max1).mean()
        mean_elapsed2 = np.array(elapsed_max2).mean()

        print("{}{}: {} ms ({} %)| {} ms ({} %)".format(key, " " * (max_key_length - len(key)), round(mean_elapsed1, 2), round(mean_elapsed1/mean_rhs_elapsed1 * 100, 2), round(mean_elapsed2, 2), round(mean_elapsed2/mean_rhs_elapsed2 * 100, 2)))
