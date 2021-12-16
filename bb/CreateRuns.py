#!/usr/bin/env python3
import csv
import os
import re

# (bb_N)([0-9]+)(_sfc)([0-9]+)(([A-Z]))(_np)(([0-9]+))


class CSVReader(object):

    def __init__(self, filename):
        self.filename = filename
        self.data = {}
        self.read()

    def read(self):
        # data = {}
        header = []
        with open(self.filename, newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=';')
            for i_row, row in enumerate(csv_reader):
                if i_row == 0:
                    for entry in row:
                        header.append(entry)
                        self.data[entry] = []
                else:
                    for i_entry, entry in enumerate(row):
                        self.data[header[i_entry]].append(entry)
        # print("data: {}".format(self.data))


# directory script being run
#  os.path.dirname(os.path.abspath(__file__))
# current working directory
#  os.path.abspath(os.getcwd())
class Run(object):

    def __init__(self, settings, base_directory, submit_template_binac, submit_template_naboo, config_template,
                 material_config_template):
        self.settings = settings
        self.base_directory = base_directory
        self.submit_template_binac = submit_template_binac
        self.submit_template_naboo = submit_template_naboo
        self.config_template = config_template
        self.material_config_template = material_config_template
        self.simulation_type = "bb"
        self.simulation_type_postprocessing = 1
        self.pmem = "10gb"
        self.wall_time = "02:00:00"
        self.initial_distribution = self.get_initial_distribution_files("initial_bb/")
        # print("directories: {}".format(self.initial_distribution))
        self.created_jobs = []

    def get_simulation_directory(self, index):
        if int(self.settings["loadBalancing"][index]) == 0:
            load_balancing = "F"
        else:
            load_balancing = "D"
        directory = "{}_N{}_sfc{}{}_np{}".format(self.simulation_type, int(self.settings["numParticles"][index]),
                                                int(self.settings["sfc"][index]), load_balancing,
                                                int(self.settings["numProcs"][index]))
        # print(directory)
        return directory

    @staticmethod
    def get_initial_distribution_files(directory="./"):
        h5_files = [x for x in os.listdir(directory) if ".h5" in x]  # if "plN" in x[0]]
        # print(h5_files)
        initial_distribution_files = {}
        for h5_file in h5_files:
            regex = re.search("(bbN)([0-9]+)", h5_file).groups()
            # print("dir: {}, N: {}".format(dir, int(regex[1])))
            initial_distribution_files[str(int(regex[1]))] = {"file": h5_file,
                                                              "path": directory + h5_file,
                                                              "fullpath": os.path.abspath(os.getcwd())
                                                              + "/" + directory + h5_file}
        return initial_distribution_files

    def create_run_list(self, file_name):
        file_content = ""
        for created_job in self.created_jobs:
            file_content += "{}\n".format(created_job)
        with open(file_name, 'w') as file:
            file.write(file_content)

    def create_run(self, index=0):
        # additional
        # print("simulation type: {}".format(self.simulation_type))
        # print("run: {}".format(self.settings["run"][index]))
        # print("numParticles: {}".format(int(self.settings["numParticles"][index])))
        # print("note: {}".format(self.settings["note"][index]))
        # config file
        # print("sfc: {}".format(int(self.settings["sfc"][index])))
        # print("loadBalancing: {}".format(int(self.settings["loadBalancing"][index])))
        # print("FRNN version: {}".format(int(self.settings["frnnVersion"][index])))
        # print("integrator: {}".format(int(self.settings["integrator"][index])))
        # print("timeEnd: {}".format(float(self.settings["timeEnd"][index])))
        # print("timeStep: {}".format(float(self.settings["timeStep"][index])))
        # command line
        # print("numProcs: {}".format(int(self.settings["numProcs"][index])))
        # print("numOutput: {}".format(int(self.settings["numOutput"][index])))

        os.system("mkdir -p {}".format(self.get_simulation_directory(index)))
        self._create_config(index)
        self._create_material_config(index)
        self._create_submit_binac(index)
        self._create_submit_naboo(index)
        self._create_post_run(index)
        self._create_summary(index)

        print("created run/job: {}".format(self.get_simulation_directory(index)))
        self.created_jobs.append("{}{}".format(self.base_directory, self.get_simulation_directory(index)))
        # print(self.created_jobs)

    def _create_summary(self, index=0):

        file_content = "simulation type:  {}\n".format(self.simulation_type)
        file_content += "numParticles:     {}\n".format(int(self.settings["numParticles"][index]))
        file_content += "numProcs:         {}\n".format(int(self.settings["numProcs"][index]))
        file_content += "integrator:       {}\n".format(int(self.settings["integrator"][index]))
        file_content += "timeStep:         {}\n".format(float(self.settings["timeStep"][index]))
        file_content += "timeEnd:          {}\n".format(float(self.settings["timeEnd"][index]))
        file_content += "sfc:              {}\n".format(int(self.settings["sfc"][index]))
        file_content += "loadBalancing:    {}\n".format(int(self.settings["loadBalancing"][index]))
        file_content += "theta:            {}\n".format(float(self.settings["theta"][index]))
        file_content += "force version:    {}\n".format(int(self.settings["computeForceVersion"][index]))
        file_content += "smoothing:        {}\n".format(float(self.settings["smoothing"][index]))
        file_content += "frnn version:     {}\n".format(int(self.settings["frnnVersion"][index]))
        file_content += "numOutput:        {}\n".format(int(self.settings["numOutput"][index]))
        file_content += "note:             {}\n".format(self.settings["note"][index])

        with open("{}/generation_summary.txt".format(self.get_simulation_directory(index)), 'w') as file:
            file.write(file_content)

    def _create_config(self, index):
        # directory _DIRECTORY_
        # integrator _INTEGRATOR_
        # timeStep _TIMESTEP_
        # timeEnd _TIMEEND_
        # sfc _SFC_
        # loadBalancing _LOADBALANCING_
        # gravityForceVersion _GRAVITYFORCEVERSION_

        with open(self.config_template, 'r') as file:
            file_content = file.read()

        # Replace the target string
        file_content = file_content.replace('_DIRECTORY_', self.base_directory + self.get_simulation_directory(index) + "/")
        file_content = file_content.replace('_INTEGRATOR_', str(int(self.settings["integrator"][index])))
        file_content = file_content.replace('_TIMESTEP_', str(float(self.settings["timeStep"][index])))
        file_content = file_content.replace('_TIMEEND_', str(float(self.settings["timeEnd"][index])))
        file_content = file_content.replace('_SFC_', str(int(self.settings["sfc"][index])))
        file_content = file_content.replace('_THETA_', str(float(self.settings["theta"][index])))
        file_content = file_content.replace('_SMOOTHING_', str(float(self.settings["smoothing"][index])))
        file_content = file_content.replace('_LOADBALANCING_', "false" if int(self.settings["loadBalancing"][index]) == 0 else "true")
        file_content = file_content.replace('_FRNNVERSION_', str(int(self.settings["frnnVersion"][index])))
        file_content = file_content.replace('_GRAVITYFORCEVERSION_', str(int(self.settings["computeForceVersion"][index])))

        # print(file_content)
        # Write the file out again
        with open("{}/config.info".format(self.get_simulation_directory(index)), 'w') as file:
            file.write(file_content)

    def _create_material_config(self, index):
        with open(self.material_config_template, 'r') as file:
            file_content = file.read()

        file_content = file_content.replace('_SML_', str(float(self.settings["sml"][index])))

        # print(file_content)
        # Write the file out again
        with open("{}/material.cfg".format(self.get_simulation_directory(index)), 'w') as file:
            file.write(file_content)

    def _create_run_cmd_binac(self, index):
        ...  # mpirun ...
        # -C config_file.info
        if int(self.settings["numProcs"][index]) == 1:
            cmd = "mpirun -np 1 bin/runner"
        else:
            cmd = "mpirun --report-bindings --map-by socket --bind-to core bin/runner"
        cmd += " -n {}".format(int(self.settings["numOutput"][index]))
        cmd += " -f {}".format("{}{}".format(self.base_directory, self.initial_distribution[str(int(self.settings["numParticles"][index]))]["path"]))
        cmd += " -C {}".format("{}{}/config.info".format(self.base_directory, self.get_simulation_directory(index)))
        cmd += " -m {}".format("{}{}/material.cfg".format(self.base_directory, self.get_simulation_directory(index)))
        # print(cmd)
        return cmd

    def _create_run_cmd_naboo(self, index):
        ...  # mpirun ...
        # -C config_file.info
        cmd = "mpirun -np {} bin/runner".format(int(self.settings["numProcs"][index]))
        cmd += " -n {}".format(int(self.settings["numOutput"][index]))
        cmd += " -f {}".format("{}{}".format(self.base_directory, self.initial_distribution[str(int(self.settings["numParticles"][index]))]["path"]))
        cmd += " -C {}".format("{}{}/config.info".format(self.base_directory, self.get_simulation_directory(index)))
        cmd += " -m {}".format("{}{}/material.cfg".format(self.base_directory, self.get_simulation_directory(index)))
        # print(cmd)
        return cmd

    def _create_post_run(self, index):
        # TODO: apply to bb

        file_content = ""
        renderer_directory = "{}{}/".format(self.base_directory, self.get_simulation_directory(index))
        file_content += "./H5Renderer/bin/h5renderer -c H5Renderer/h5renderer.info -i {} -o {} > /dev/null".format(renderer_directory, renderer_directory)
        file_content += "\n"
        file_content += "yes | ./H5Renderer/createMP4From {} &> /dev/null".format(renderer_directory)
        file_content += "\n"
        file_content += "\n"

        # max density evolution
        file_content += "./postprocessing/PlotBB.py -d {}{}/ -o {}{}/".format(self.base_directory, self.get_simulation_directory(index),
                                                                             self.base_directory, self.get_simulation_directory(index))
        file_content += "\n"
        file_content += "\n"

        # min max mean ...
        file_content += "./postprocessing/GetMinMaxMean.py -i {}{}/ -o {}{}/".format(self.base_directory, self.get_simulation_directory(index),
                                                                                     self.base_directory, self.get_simulation_directory(index))
        file_content += "\n"
        file_content += "./postprocessing/PlotMinMaxMean.py -i {}{}/min_max_mean.csv -o {}{}/ -a".format(self.base_directory,
                                                                                                         self.get_simulation_directory(index),
                                                                                                         self.base_directory,
                                                                                                         self.get_simulation_directory(index))

        file_content += "\n"
        file_content += "\n"
        file_content += "./postprocessing/Performance.py -f {}{}/log/performance.h5 -d {}{}/ -s {}".format(self.base_directory, self.get_simulation_directory(index),
                                                                                                           self.base_directory, self.get_simulation_directory(index),
                                                                                           self.simulation_type_postprocessing)
        file_content += "\n"

        with open("{}/postprocess.sh".format(self.get_simulation_directory(index)), 'w') as file:
            file.write(file_content)

        os.system("chmod 755 {}/postprocess.sh".format(self.get_simulation_directory(index)))

    def _create_submit_binac(self, index):
        # _JOBNAME_
        # _NODES_ e.g.: 2
        # _PPN_ e.g.: 4
        # _GPUS_ e.g.: 4
        # _WALLTIME_ e.g.: 00:12:00
        # _PMEM_ e.g.: 10gb
        # _CUDAVISIBLEDEVICES_ e.g.: 0,1,2,3,4,5,6,7
        # _MPIRUN_

        # Read in the file
        with open(self.submit_template_binac, 'r') as file:
            file_content = file.read()

        # Replace the target string
        file_content = file_content.replace('_JOBNAME_', self.get_simulation_directory(index))
        nodes = int(int(self.settings["numProcs"][index])/4)
        if nodes == 0:
            nodes = 1
        file_content = file_content.replace('_NODES_', str(nodes))
        ppn_gpus = int(self.settings["numProcs"][index]) % 4
        if ppn_gpus == 0:
            ppn_gpus = 4
        file_content = file_content.replace('_PPN_', str(ppn_gpus))
        file_content = file_content.replace('_GPUS_', str(ppn_gpus))
        file_content = file_content.replace('_WALLTIME_', self.wall_time)
        file_content = file_content.replace('_PMEM_', self.pmem)
        cuda_visible_devices = ""
        for i in range(int(self.settings["numProcs"][index])):
            cuda_visible_devices += str(i)+","
        cuda_visible_devices = cuda_visible_devices[0:-1]
        file_content = file_content.replace('_CUDAVISIBLEDEVICES_', cuda_visible_devices)
        cmd = self._create_run_cmd_binac(index)
        file_content = file_content.replace('_MPIRUN_', cmd)
        file_content = file_content.replace('_POSTPROCESS_', "./{}{}/postprocess.sh".format(self.base_directory, self.get_simulation_directory(index)))

        # print(file_content)
        # Write the file out again
        with open("{}/submit.sh".format(self.get_simulation_directory(index)), 'w') as file:
            file.write(file_content)

        os.system("chmod 755 {}/submit.sh".format(self.get_simulation_directory(index)))

    def _create_submit_naboo(self, index):
        # _MPIRUN_
        # Read in the file
        with open(self.submit_template_naboo, 'r') as file:
            file_content = file.read()

        cmd = self._create_run_cmd_naboo(index)
        file_content = file_content.replace('_MPIRUN_', cmd)
        file_content = file_content.replace('_POSTPROCESS_', "./{}{}/postprocess.sh".format(self.base_directory, self.get_simulation_directory(index)))

        # print(file_content)
        # Write the file out again
        with open("{}/submit_naboo.sh".format(self.get_simulation_directory(index)), 'w') as file:
            file.write(file_content)

        os.system("chmod 755 {}/submit_naboo.sh".format(self.get_simulation_directory(index)))


if __name__ == '__main__':

    filename = "BBRuns.csv"
    csv_reader = CSVReader(filename=filename)

    run = Run(csv_reader.data, "bb/", "initial_bb/submit.sh",
                             "initial_bb/submit_naboo.sh", "initial_bb/config.info",
                             "initial_bb/material.cfg")

    for i in range(len(csv_reader.data["run"])):
        run.create_run(i)

    run.create_run_list("List" + filename.replace(".csv", ".txt"))
