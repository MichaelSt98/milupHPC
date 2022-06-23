#!/usr/bin/env python3

from statistics import mean
import numpy as np
import h5py
import argparse
import sys
import csv


def get_dataset_keys(f):
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys


class H5entry(object):

    def __init__(self, name, data, extend=False, extend_length=0):
        self.name = name
        self.data = data
        self.numProcesses = len(data[0])
        if extend:
            self.extend_data(extend_length)

    def extend_data(self, extend_length):
        to_be_extended = self.data  # np.array(self.data)
        length = len(to_be_extended)
        append_length = int(extend_length/length)
        # print("shape: {}".format(to_be_extended.shape))
        # print("to_be_extended[0] = {}".format(to_be_extended[0]))
        # print("to_be_extended[1] = {}".format(to_be_extended[1]))
        self.data = []
        for i_step in range(length):
            value = []
            for i_proc in range(self.numProcesses):
                value.append(to_be_extended[i_step][i_proc])
            # print("append: {}, shape: {}".format(value, np.array(value).shape))
            self.data.append(value)
            for i in range(append_length-1):
                self.data.append(np.zeros(np.array(value).shape))
                #print("append: {}".format(np.zeros(np.array(value).shape)))
        self.data = np.array(self.data)
        # print("new shape: {}".format(self.data.shape))

    @staticmethod
    def get_percentage(data, reference):
        return data/reference

    def get_proc_data(self, proc=0):
        if 0 <= proc < self.numProcesses:
            return [elem[proc] for elem in self.data]
        else:
            print("Only {} processes available!".format(self.numProcesses))

    def get_averages(self):
        return [mean(elem) for elem in self.data]

    def get_average_mean(self):
        return mean(self.get_averages())

    def get_maxima(self):
        return [max(elem) for elem in self.data]

    def get_maxima_mean(self):
        maxima = self.get_maxima()
        zeros = maxima.count(0)
        return mean(maxima), sum(maxima)/(len(maxima) - zeros), zeros

    def get_minima(self):
        return [min(elem) for elem in self.data]

    def get_minima_mean(self):
        return mean(self.get_minima())

    def get_average_per_proc(self):
        averages = []
        for proc in range(self.numProcesses):
            averages.append(mean(self.get_data(proc)))
        return averages

    def get_sum_per_step(self):
        sums = []
        for i_step in range(len(self.data)):
            sum_per_proc = []
            for i_proc in range(self.numProcesses):
                sum_per_proc.append(sum(self.data[i_step][i_proc]))
            sums.append(sum_per_proc)
        return sums


class TimeEntry:

    def __init__(self, key, name, color="grey"):
        self.key = key
        self.name = name
        self.color = color
        self.mean = 0
        self.real_mean = 0
        self.unit = "ms"


class TimeEvaluator(object):

    reference_entry = TimeEntry("rhsElapsed", "rhs")

    preprocessing_entries = \
        [
            TimeEntry("removingParticles", "remove particles"),
            TimeEntry("loadBalancing", "load balancing")
        ]

    postprocessing_entries = \
        [
            TimeEntry("IO", "IO"),
            TimeEntry("integrate", "integration")
        ]

    gravity_sim_entries = \
        [
            TimeEntry("reset", "reset"),
            TimeEntry("assignParticles", "assign particles"),
            TimeEntry("boundingBox", "bounding box"),
            TimeEntry("tree", "build tree"),
            TimeEntry("pseudoParticle", "calculate pseudo-particle"),
            TimeEntry("gravity", "gravitational force")
        ]

    sph_sim_entries = \
        [
            TimeEntry("reset", "reset"),
            TimeEntry("assignParticles", "assign particles"),
            TimeEntry("boundingBox", "bounding box"),
            TimeEntry("tree", "build tree"),
            TimeEntry("pseudoParticle", "calculate pseudo-particle"),
            TimeEntry("sph", "SPH")
        ]

    gravity_sph_sim_entries = \
        [
            TimeEntry("reset", "reset"),
            TimeEntry("assignParticles", "assign particles"),
            TimeEntry("boundingBox", "bounding box"),
            TimeEntry("tree", "build tree"),
            TimeEntry("pseudoParticle", "calculate pseudo-particle"),
            TimeEntry("gravity", "gravitational force"),
            TimeEntry("sph", "SPH")
        ]

    details_tree_entries = \
        [
            TimeEntry("tree_createDomainList", "create domain list"),
            TimeEntry("tree_buildTree", "build tree from particles"),
            TimeEntry("tree_buildDomainTree", "assign and add domain list nodes")
        ]

    details_gravity_entries = \
        [
            TimeEntry("gravity_compTheta", "determine relevant domain list nodes"),
            TimeEntry("gravity_symbolicForce", "determine (pseudo-) particles to be sent"),
            TimeEntry("gravity_gravitySendingParticles", "send (pseudo-) particles"),
            TimeEntry("gravity_insertReceivedPseudoParticles", "insert received pseudo-particles"),
            TimeEntry("gravity_insertReceivedParticles", "insert received particles"),
            TimeEntry("gravity_force", "gravitational force calculation"),
            TimeEntry("gravity_repairTree", "repair tree (delete received particles)")
        ]

    details_sph_entries = \
        [
            TimeEntry("sph_compTheta", "determine relevant domain list nodes"),
            TimeEntry("sph_determineSearchRadii", "calculate search radius"),
            TimeEntry("sph_symbolicForce", "determine particles to be sent"),
            TimeEntry("sph_sendingParticles", "send particles"),
            TimeEntry("sph_insertReceivedParticles", "insert received particles"),
            TimeEntry("sph_fixedRadiusNN", "FRNN search"),
            TimeEntry("sph_density", "calculate density"),
            TimeEntry("sph_pressure", "calculate pressure"),
            TimeEntry("sph_soundSpeed", "calculate speed of sound"),
            TimeEntry("sph_resendingParticles", "resend relevant entries"),
            TimeEntry("sph_internalForces", "calculate internal forces"),
            TimeEntry("sph_repairTree", "repair tree (delete received particles)")
        ]

    def __init__(self, input_file, data_dic, sim_type, unit="ms"):
        print("Time evaluation ...")
        self.input_file = input_file
        self.data_dic = data_dic
        # 0: gravity, 1: sph, 2: gravity + sph
        self.sim_type = sim_type
        # print("sim_type: {}".format(self.sim_type))
        self.unit = unit
        self.relevant_entries = None
        self.numProcesses = self.data_dic["tree"].numProcesses
        self.get_relevant_entries()

    def get_relevant_entries(self):
        if self.sim_type == 0:
            self.relevant_entries = self.gravity_sim_entries
        elif self.sim_type == 1:
            self.relevant_entries = self.sph_sim_entries
        elif self.sim_type == 2:
            self.relevant_entries = self.gravity_sph_sim_entries
        else:
            print("sim_type {} not available! exiting...".format(self.sim_type))
            sys.exit(1)

    def summarize(self, filename, preprocessing=False, postprocessing=False):

        summarize_entries = []
        if preprocessing:
            summarize_entries.extend(self.preprocessing_entries)
        summarize_entries.extend(self.relevant_entries)
        if postprocessing:
            summarize_entries.extend(self.postprocessing_entries)

        with open(filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=";")
            csv_writer.writerow([self.input_file, self.sim_type, self.numProcesses])
            header = ["key", "name", "total_average", "real_average"]
            csv_writer.writerow(header)
            for relevant_entry in summarize_entries:
                relevant_entry.mean, self.real_mean, zeros = self.data_dic[relevant_entry.key].get_maxima_mean()
                # print("{}: {} {} | {} ({})".format(relevant_entry.name, relevant_entry.mean, relevant_entry.unit,
                #                                self.real_mean, zeros))
                csv_writer.writerow([relevant_entry.key, relevant_entry.name, relevant_entry.mean, self.real_mean])

    def summarize_tree(self, filename):
        with open(filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=";")
            csv_writer.writerow([self.input_file, self.sim_type, self.numProcesses])
            header = ["key", "name", "total_average", "real_average"]
            csv_writer.writerow(header)
            for relevant_entry in self.details_tree_entries:
                relevant_entry.mean, self.real_mean, zeros = self.data_dic[relevant_entry.key].get_maxima_mean()
                # print("{}: {} {} | {} ({})".format(relevant_entry.name, relevant_entry.mean, relevant_entry.unit,
                #                                    self.real_mean, zeros))
                csv_writer.writerow([relevant_entry.key, relevant_entry.name, relevant_entry.mean, self.real_mean])

    def summarize_gravity(self, filename):
        with open(filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=";")
            csv_writer.writerow([self.input_file, self.sim_type, self.numProcesses])
            header = ["key", "name", "total_average", "real_average"]
            csv_writer.writerow(header)
            for relevant_entry in self.details_gravity_entries:
                relevant_entry.mean, self.real_mean, zeros = self.data_dic[relevant_entry.key].get_maxima_mean()
                # print("{}: {} {} | {} ({})".format(relevant_entry.name, relevant_entry.mean, relevant_entry.unit,
                #                                    self.real_mean, zeros))
                csv_writer.writerow([relevant_entry.key, relevant_entry.name, relevant_entry.mean, self.real_mean])

    def summarize_sph(self, filename):
        with open(filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=";")
            csv_writer.writerow([self.input_file, self.sim_type, self.numProcesses])
            header = ["key", "name", "total_average", "real_average"]
            csv_writer.writerow(header)
            for relevant_entry in self.details_sph_entries:
                relevant_entry.mean, self.real_mean, zeros = self.data_dic[relevant_entry.key].get_maxima_mean()
                # print("{}: {} {} | {} ({})".format(relevant_entry.name, relevant_entry.mean, relevant_entry.unit,
                #                                    self.real_mean, zeros))
                csv_writer.writerow([relevant_entry.key, relevant_entry.name, relevant_entry.mean, self.real_mean])


class ParticleEvaluator(object):

    # entries = ["numParticles", "numParticlesLocal", "ranges"]

    def __init__(self, input_file, data_dic, sim_type):
        print("Particle evaluation ...")
        self.input_file = input_file
        self.data_dic = data_dic
        self.sim_type = sim_type
        self.numProcesses = self.data_dic["numParticles"].numProcesses

    def summarize(self, filename):
        start_number_particles = self.data_dic["numParticles"].data[0][0]
        with open(filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=";")
            csv_writer.writerow([self.input_file, self.sim_type, self.numProcesses])
            header = ["numParticles", "numParticlesLocal", "numParticlesPercentage", "loss", "lossPercentage"]
            csv_writer.writerow(header)
            for i_step in range(len(self.data_dic["numParticles"].data)):
                num_particles = self.data_dic["numParticles"].data[i_step][0]
                num_particles_local = self.data_dic["numParticlesLocal"].data[i_step]
                num_particles_percentage = 100 * (self.data_dic["numParticlesLocal"].data[i_step] /
                                                  self.data_dic["numParticles"].data[i_step][0])
                # print("numParticles: {} numParticlesLocal: {} percentage: {} loss: {} = {}".format(
                #     num_particles,
                #     num_particles_local,
                #     num_particles_percentage,
                #     start_number_particles - num_particles,
                #     (start_number_particles - num_particles)/num_particles
                # ))
                row = [num_particles]
                temp = [num_particles_local[proc] for proc in range(self.numProcesses)]
                row.extend(temp)
                temp = [num_particles_percentage[proc] for proc in range(self.numProcesses)]
                row.extend(temp)
                row.append(start_number_particles - num_particles)
                row.append((start_number_particles - num_particles)/num_particles)
                csv_writer.writerow(row)


class CommunicationEvaluator(object):

    # entries = ["receiving/gravityParticles", "receiving/gravityPseudoParticles", "receiving/sph"]
    # entries = ["sending/gravityParticles", "sending/gravityPseudoParticles", "sending/sph"]

    def __init__(self, input_file, data_dic, sim_type):
        print("Communication evaluation ...")
        self.input_file = input_file
        self.data_dic = data_dic
        # 0: gravity, 1: sph, 2: gravity + sph
        self.sim_type = sim_type
        # print("sim_type: {}".format(self.sim_type))
        self.numProcesses = 0
        try:
            self.numProcesses = self.data_dic["gravityParticles"].numProcesses
        except:
            self.numProcesses = self.data_dic["sph"].numProcesses
        self.gravity_particle_sums = None
        self.gravity_pseudo_particle_sums = None
        self.gravity_sums = None
        self.sph_sums = None
        self.total_communication_per_step_per_proc()

        self.gravity_total_sums = None
        self.gravity_total_particle_sums = None
        self.gravity_total_pseudo_particle_sums = None
        self.sph_total_sums = None
        self.total_sums = None
        self.total_communication_per_step()

    def total_communication_per_step_per_proc(self):
        if self.sim_type == 0:
            self.gravity_particle_sums = self.data_dic["gravityParticles"].get_sum_per_step()
            self.gravity_pseudo_particle_sums = self.data_dic["gravityPseudoParticles"].get_sum_per_step()
            self.gravity_sums = [self.gravity_particle_sums[i] + self.gravity_pseudo_particle_sums[i] for i in range(len(self.gravity_particle_sums))]
            self.sph_sums = [0 for i in range(len(self.gravity_particle_sums))]
        elif self.sim_type == 1:
            self.sph_sums = self.data_dic["sph"].get_sum_per_step()
            self.gravity_particle_sums = [0 for i in range(len(self.sph_sums))]
            self.gravity_pseudo_particle_sums = [0 for i in range(len(self.sph_sums))]
            self.gravity_sums = [0 for i in range(len(self.sph_sums))]
        elif self.sim_type == 2:
            self.gravity_particle_sums = self.data_dic["gravityParticles"].get_sum_per_step()
            self.gravity_pseudo_particle_sums = self.data_dic["gravityPseudoParticles"].get_sum_per_step()
            self.gravity_sums = [self.gravity_particle_sums[i] + self.gravity_pseudo_particle_sums[i] for i in range(len(self.gravity_particle_sums))]
            self.sph_sums = self.data_dic["sph"].get_sum_per_step()
        else:
            print("sim_type {} not available! exiting...".format(self.sim_type))
            sys.exit(1)

    def total_communication_per_step(self):
        if self.sim_type == 0:
            self.gravity_total_sums = [sum(elem) for elem in self.gravity_sums]
            self.gravity_total_particle_sums = [sum(elem) for elem in self.gravity_particle_sums]
            self.gravity_total_pseudo_particle_sums = [sum(elem) for elem in self.gravity_pseudo_particle_sums]
            self.sph_total_sums = [0 for i in range(len(self.sph_sums))]
            self.total_sums = self.total_sums
        elif self.sim_type == 1:
            self.gravity_total_sums = [0 for i in range(len(self.sph_sums))]
            self.gravity_total_particle_sums = [0 for i in range(len(self.sph_sums))]
            self.gravity_total_pseudo_particle_sums = [0 for i in range(len(self.sph_sums))]
            self.sph_total_sums = [sum(elem) for elem in self.sph_sums]
            self.total_sums = self.sph_total_sums
        elif self.sim_type == 2:
            self.gravity_total_sums = [sum(elem) for elem in self.gravity_sums]
            self.gravity_total_particle_sums = [sum(elem) for elem in self.gravity_particle_sums]
            self.gravity_total_pseudo_particle_sums = [sum(elem) for elem in self.gravity_pseudo_particle_sums]
            self.sph_total_sums = [sum(elem) for elem in self.sph_sums]
            self.total_sums = [self.gravity_total_sums[i] + self.sph_total_sums[i] for i in range(len(self.gravity_total_sums))]
        else:
            print("sim_type {} not available! exiting...".format(self.sim_type))
            sys.exit(1)

    def summarize(self, filename):

        # self.gravity_particle_sums = None
        # self.gravity_pseudo_particle_sums = None
        # self.gravity_sums = None
        # self.sph_sums = None
        # self.total_communication_per_step_per_proc()
        # self.gravity_total_sums = None
        # self.sph_total_sums = None
        # self.total_sums = None

        # gravity all, gravity particle all, gravity pseudo-particle all, sph all, gravity per proc,
        # gravity particle per proc, gravity pseudo-particle per proc, sph per proc

        with open(filename, 'w', newline='') as csv_file:
            if self.sim_type == 0:
                csv_writer = csv.writer(csv_file, delimiter=";")
                csv_writer.writerow([self.input_file, self.sim_type, self.numProcesses])
                header = ["total_gravity", "total_gravity_particle", "total_gravity_pseudo_particle"]
                [header.append("gravity_proc{}".format(proc)) for proc in range(self.numProcesses)]
                [header.append("gravity_particle_proc{}".format(proc)) for proc in range(self.numProcesses)]
                [header.append("gravity_pseudo_particle_proc{}".format(proc)) for proc in range(self.numProcesses)]
                csv_writer.writerow(header)

                for i in range(len(self.gravity_sums)):
                    row = [self.gravity_total_sums[i],
                           self.gravity_total_particle_sums[i],
                           self.gravity_total_pseudo_particle_sums[i]]
                    temp = [self.gravity_sums[i][proc] for proc in range(self.numProcesses)]
                    row.extend(temp)
                    temp = [self.gravity_particle_sums[i][proc] for proc in range(self.numProcesses)]
                    row.extend(temp)
                    temp = [self.gravity_pseudo_particle_sums[i][proc] for proc in range(self.numProcesses)]
                    row.extend(temp)

                    csv_writer.writerow(row)
            elif self.sim_type == 1:
                csv_writer = csv.writer(csv_file, delimiter=";")
                csv_writer.writerow([self.input_file, self.sim_type, self.numProcesses])
                header = ["total_sph"]
                [header.append("sph_proc{}".format(proc)) for proc in range(self.numProcesses)]
                csv_writer.writerow(header)

                for i in range(len(self.gravity_sums)):
                    row = [self.sph_total_sums[i]]
                    temp = [self.sph_sums[i][proc] for proc in range(self.numProcesses)]
                    row.extend(temp)

                    csv_writer.writerow(row)
            elif self.sim_type == 2:
                csv_writer = csv.writer(csv_file, delimiter=";")
                csv_writer.writerow([self.input_file, self.sim_type, self.numProcesses])
                header = ["total_gravity", "total_gravity_particle", "total_gravity_pseudo_particle", "total_sph"]
                [header.append("gravity_proc{}".format(proc)) for proc in range(self.numProcesses)]
                [header.append("gravity_particle_proc{}".format(proc)) for proc in range(self.numProcesses)]
                [header.append("gravity_pseudo_particle_proc{}".format(proc)) for proc in range(self.numProcesses)]
                [header.append("sph_proc{}".format(proc)) for proc in range(self.numProcesses)]
                csv_writer.writerow(header)

                for i in range(len(self.gravity_sums)):
                    row = [self.gravity_total_sums[i],
                           self.gravity_total_particle_sums[i],
                           self.gravity_total_pseudo_particle_sums[i],
                           self.sph_total_sums[i]]
                    temp = [self.gravity_sums[i][proc] for proc in range(self.numProcesses)]
                    row.extend(temp)
                    temp = [self.gravity_particle_sums[i][proc] for proc in range(self.numProcesses)]
                    row.extend(temp)
                    temp = [self.gravity_pseudo_particle_sums[i][proc] for proc in range(self.numProcesses)]
                    row.extend(temp)
                    temp = [self.sph_sums[i][proc] for proc in range(self.numProcesses)]
                    row.extend(temp)

                    csv_writer.writerow(row)
            else:
                print("sim type {} not available!".format(self.sim_type))
                sys.exit(1)


class Dic2H5(object):

    def __init__(self, data_dic):
        self.data_dic = data_dic

    def write_to_h5(self, filename="summary.h5"):

        hf = h5py.File(filename, "w")
        for data_key in self.data_dic:
            hf.create_dataset(data_key, data=self.data_dic[data_key])

        hf.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Performance postprocessing, generating csv (summary) files.")
    parser.add_argument("--file", "-f",  metavar="str", type=str, help="path to the h5 profiling data file", required=True)
    parser.add_argument("--directory", "-d", metavar="str", type=str, help="output path", required=True)
    parser.add_argument("--sim_type", "-s", metavar="int", type=int, help="simulation type", required=True)
    args = parser.parse_args()

    particle_evaluation = True
    particle_evaluation_file = "{}particle".format(args.directory)

    time_evaluation = True
    time_evaluation_file = "{}time".format(args.directory)
    time_detailed_tree = True
    if args.sim_type == 0 or args.sim_type == 2:
        time_detailed_gravity = True
    else:
        time_detailed_gravity = False
    if args.sim_type == 1 or args.sim_type == 2:
        time_detailed_sph = True
    else:
        time_detailed_sph = False

    communication_evaluation = True
    communication_evaluation_file = "{}communication".format(args.directory)

    f = h5py.File(args.file, 'r')
    keys = get_dataset_keys(f)
    # print("keys: {}".format(keys))

    max_length = 0
    for key in keys:
        if len(f[key][:]) > max_length:
            max_length = len(f[key][:])

    ##########################################################################
    if particle_evaluation:
        particle_dic = {}
        for key in keys:
            if "general/" in key:
                # print(key)
                name = key.replace("general/", "")
                particle_dic[name] = H5entry(name, f[key][:])
        particle_evaluator = ParticleEvaluator(args.file, particle_dic, args.sim_type)
        particle_evaluator.summarize("{}.csv".format(particle_evaluation_file))

    ##########################################################################
    if time_evaluation:
        time_dic = {}
        for key in keys:
            if "time" in key:
                # print(key)
                name = key.replace("time/", "")
                if len(f[key][:]) < max_length:
                    time_dic[name] = H5entry(name, f[key][:], True, max_length)
                else:
                    time_dic[name] = H5entry(name, f[key][:])

        time_evaluator = TimeEvaluator(args.file, time_dic, args.sim_type)
        time_evaluator.summarize("{}.csv".format(time_evaluation_file), False, True)
        if time_detailed_tree:
            time_evaluator.summarize_tree("{}_tree.csv".format(time_evaluation_file))
        if time_detailed_gravity:
            time_evaluator.summarize_gravity("{}_gravity.csv".format(time_evaluation_file))
        if time_detailed_sph:
            time_evaluator.summarize_sph("{}_sph.csv".format(time_evaluation_file))

    ##########################################################################
    if communication_evaluation:
        communication_dic = {}
        for key in keys:
            if "sending" in key and "time" not in key:
                name = key.replace("sending/", "")
                # print("key: {}, name: {}".format(key, name))
                if len(f[key][:]) < max_length:
                    communication_dic[name] = H5entry(name, f[key][:], True, max_length)
                else:
                    communication_dic[name] = H5entry(name, f[key][:])

        communication_evaluator = CommunicationEvaluator(args.file, communication_dic, args.sim_type)
        communication_evaluator.summarize("{}.csv".format(communication_evaluation_file))

    f.close()

