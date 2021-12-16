#!/bin/bash
#PBS -N pl_N4096_sfc0D_np4
#PBS -l nodes=1:ppn=4:gpus=4:exclusive_process
#PBS -l walltime=02:00:00
#PBS -l pmem=10gb
#PBS -q gpu
//#PBS -m aeb -M michael.staneker@student.uni-tuebingen.de
source ~/.bashrc

# Loading modules
module load compiler/gnu/8.3
module load devel/cuda/10.1
module load mpi/openmpi/4.1-gnu-8.3
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-8.3
module load devel/python/3.7.1
source ~/milupHPCpython/bin/activate

# Going to working directory
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/local/lib
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PSM3_MULTI_EP=1

nvidia-smi
nvidia-smi -L

mpirun --report-bindings --map-by socket --bind-to core bin/runner -n 200 -f plummer/initial_plummer/plN4096seed2723515521.h5 -C plummer/pl_N4096_sfc0D_np4/config.info

./plummer/pl_N4096_sfc0D_np4/postprocess.sh
