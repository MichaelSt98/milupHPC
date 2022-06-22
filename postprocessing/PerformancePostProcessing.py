#!/usr/bin/env python3

from statistics import mean
import numpy as np
import h5py
import argparse
import sys
import csv


class CSVReader(object):

    def __init__(self, filename):
        self.filename = filename
        self.input_file = None
        self.sim_type = None
        self.num_processes = None
        self.data = {}
        self.read()

    def read(self):
        # data = {}
        header = []
        with open(self.filename, newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=';')
            for i_row, row in enumerate(csv_reader):
                if i_row == 0:
                    self.input_file = row[0]
                    self.sim_type = row[1]
                    self.num_processes = row[2]
                elif i_row == 1:
                    for entry in row:
                        header.append(entry)
                        self.data[entry] = []
                else:
                    for i_entry, entry in enumerate(row):
                        self.data[header[i_entry]].append(entry)
        print("input file: {}".format(self.input_file))
        print("sim type: {}".format(self.sim_type))
        print("num processes: {}".format(self.num_processes))
        # print("data: {}".format(self.data))


if __name__ == '__main__':

    # plummer
    """
    file = "time.csv" # time_gravity
    directories = ["/Users/Michi/Desktop/milupHPC/master/binac_data/plummer/milupHPC2/milupHPC/plummer/pl_N1000000_sfc0F_np1/",
                   "/Users/Michi/Desktop/milupHPC/master/binac_data/plummer/milupHPC2/milupHPC/plummer/pl_N1000000_sfc1D_np2/",
                   "/Users/Michi/Desktop/milupHPC/master/binac_data/plummer/milupHPC2/milupHPC/plummer/pl_N1000000_sfc1D_np4/",
                   "/Users/Michi/Desktop/milupHPC/master/binac_data/plummer/milupHPC2/milupHPC/plummer/pl_N1000000_sfc1D_np8/",
                   "/Users/Michi/Desktop/milupHPC/master/binac_data/plummer/milupHPC2/milupHPC/plummer/pl_N1000000_sfc1D_np16/",
                   "/Users/Michi/Desktop/milupHPC/master/binac_data/plummer/milupHPC2/milupHPC/plummer/pl_N1000000_sfc1D_np32/"]
    """

    """
    # sedov
    file = "time_tree.csv" # time_sph
    directories = ["/Users/Michi/Desktop/milupHPC/master/binac_data/sedov/milupHPC/sedov/sedov_N81_sfc0F_np1/",
                   "/Users/Michi/Desktop/milupHPC/master/binac_data/sedov/milupHPC/sedov/sedov_N81_sfc1D_np2/",
                   "/Users/Michi/Desktop/milupHPC/master/binac_data/sedov/milupHPC/sedov/sedov_N81_sfc1D_np4/",
                   "/Users/Michi/Desktop/milupHPC/master/binac_data/sedov/milupHPC/sedov/sedov_N81_sfc1D_np8/",
                   "/Users/Michi/Desktop/milupHPC/master/binac_data/sedov/milupHPC/sedov/sedov_N81_sfc1D_np16/",
                   "/Users/Michi/Desktop/milupHPC/master/binac_data/sedov/milupHPC/sedov/sedov_N81_sfc1D_np32/"]
    """

    # bb
    file = "time_sph.csv" # time_sph, time_gravity
    directories = ["/Users/Michi/Desktop/milupHPC/master/binac_data/bb/milupHPC2_short/bb/bb_N500000_sfc0F_np1/",
                   "/Users/Michi/Desktop/milupHPC/master/binac_data/bb/milupHPC2_short/bb/bb_N500000_sfc1D_np2/",
                   "/Users/Michi/Desktop/milupHPC/master/binac_data/bb/milupHPC2_short/bb/bb_N500000_sfc1D_np4/",
                   "/Users/Michi/Desktop/milupHPC/master/binac_data/bb/milupHPC2_short/bb/bb_N500000_sfc1D_np8/",
                   "/Users/Michi/Desktop/milupHPC/master/binac_data/bb/milupHPC2_short/bb/bb_N500000_sfc1D_np16/",
                   "/Users/Michi/Desktop/milupHPC/master/binac_data/bb/milupHPC2_short/bb/bb_N500000_sfc1D_np32/"]

    file_names = ["{}{}".format(directory, file) for directory in directories]

    data = {}
    available_np = []
    for i_file_name, file_name in enumerate(file_names):
        csv_reader = CSVReader(file_name)
        available_np.append(str(csv_reader.num_processes))
        data["{}".format(csv_reader.num_processes)] = csv_reader.data

    header = ["number of processes"]
    for _np in available_np:
        header.append(_np)
    header.append("\\\\")
    print(header)
    with open('eggs.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='&', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)

        for i_key, key in enumerate(data["1"]["key"]):
            row = [data["1"]["name"][i_key]]
            for _np in available_np:
                row.append(round(float(data[_np]["total_average"][i_key]), 3))
                # row.append("{} ({})".format(round(float(data[_np]["total_average"][i_key]), 3), round(float(data[_np]["real_average"][i_key]), 3)))
            row.append("\\\\")
            writer.writerow(row)


