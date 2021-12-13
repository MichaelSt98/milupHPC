#!/bin/bash
#PBS -N _JOBNAME_
#PBS -l nodes=_NODES_:ppn=_PPN_:gpus=_GPUS_:exclusive_process
#PBS -l walltime=_WALLTIME_
#PBS -l pmem=_PMEM_
#PBS -q gpu
#PBS -m aeb -M michael.staneker@student.uni-tuebingen.de
source ~/.bashrc

# Loading modules
module load compiler/gnu/8.3
module load devel/cuda/10.1
module load mpi/openmpi/4.1-gnu-8.3
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-8.3
module load devel/python/3.7.1

# Going to working directory
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/local/lib
export CUDA_VISIBLE_DEVICES=_CUDAVISIBLEDEVICES_
export PSM3_MULTI_EP=1

nvidia-smi
nvidia-smi -L

_MPIRUN_

_POSTPROCESS_
