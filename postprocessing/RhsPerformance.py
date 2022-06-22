#!/usr/bin/env python3

import argparse
import h5py
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Rhs performance evaluation")
    parser.add_argument("--data", "-d", metavar="str", type=str, help="input directory",
                        nargs="?", default="../output")
    #parser.add_argument("--output", "-o", metavar="str", type=str, help="output directory",
    #                    nargs="?", default="../output")

    args = parser.parse_args()


    f = h5py.File(args.data, 'r')
    rhs_elapsed = np.array(f["time/rhsElapsed"][:])
    #gravity_symbolicForce = np.array(f["time/gravity_symbolicForce"][:])

    rhs_elapsed_max = [np.array(elem).max() for elem in rhs_elapsed]
    #gravity_symbolicForce_max = [np.array(elem).max() for elem in gravity_symbolicForce]

    # for i_elem, elem in enumerate(rhs_elapsed_max):
    #     print("{}: {} ms".format(i_elem, elem))

    mean_rhs_elapsed = np.array(rhs_elapsed_max).mean()
    #mean_gravity_symbolicForce = np.array(gravity_symbolicForce_max).mean()

    print("rhs elapsed average: {}".format(mean_rhs_elapsed))
    #print("gravity symbolic force average: {}".format(mean_gravity_symbolicForce))

    #groups = list(f.keys())
    keys = f['time'].keys()

    for key in keys:
        elapsed = np.array(f["time/{}".format(key)][:])
        elapsed_max = [np.array(elem).max() for elem in elapsed]
        mean_elapsed = np.array(elapsed_max).mean()
        print("{}: {} ms".format(key, mean_elapsed))
