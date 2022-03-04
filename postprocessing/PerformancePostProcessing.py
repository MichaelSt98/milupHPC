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

    file = ""
    directories = ["./"]
    file_names = ["{}{}".format(directory, file) for directory in directories]

    for i_file_name, file_name in enumerate(file_names):
        csv_reader = CSVReader(file_name)
