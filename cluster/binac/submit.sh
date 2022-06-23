#!/bin/bash
#PBS -N N1e6SSG
#PBS -l nodes=1:ppn=4:gpus=4:exclusive_process
#PBS -l walltime=00:12:00
#PBS -l pmem=10gb
#PBS -q tiny
#PBS -m aeb -M michael.staneker@student.uni-tuebingen.de
source ~/.bashrc

# Loading modules
module load compiler/gnu/8.3
module load devel/cuda/10.1
module load mpi/openmpi/4.1-gnu-8.3
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-8.3

# Going to working directory
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/local/lib
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PSM3_MULTI_EP=1

nvidia-smi

# Starting program
#mpirun --bind-to socket --map-by core --report-bindings bin/runner -i 10 -c 0 -f examples/kepler.h5
mpirun --bind-to core --map-by core --report-bindings bin/runner -i 100 -c 1 -f examples/plummer.h5 -l -L 10
