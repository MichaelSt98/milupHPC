#!/bin/bash
#SBATCH --job-name=sedov_N126_sfc1D_np4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --mem=10G
#SBATCH --gres=gpu:4
source ~/.bashrc

# Loading modules
# module load compiler/gnu/12.1
module load compiler/gnu/10.2
module load devel/cuda/11.4
module load mpi/openmpi/4.1
# module load lib/hdf5/1.12.2-gnu-12.1-openmpi-4.1
module load lib/hdf5/1.12.2-gnu-10.2-openmpi-4.1

# Going to working directory
cd $SLURM_SUBMIT_DIR
echo $SLURM_SUBMIT_DIR

which mpicc

#export OMPI_MCA_pml=ucx

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/local/lib

nvidia-smi
nvidia-smi -L

mpirun -np 4 bin/runner -n 50 -f sedov/initial_sedov/sedovN126.h5 -C sedov/sedov_N126_sfc1D_np4/config.info -m sedov/sedov_N126_sfc1D_np4/material.cfg

#mpirun -np 1 bin/runner -n 10 -v 3 -f plummer/initial_plummer/plN4096seed2723515521.h5 -C plummer/pl_N4096_sfc0F_np1/config.info

#mpirun --mca opal_common_ucx_opal_mem_hooks 1 -np 1 bin/runner -n 200 -v 3 -f plummer/initial_plummer/plN4096seed2723515521.h5 -C plummer/pl_N4096_sfc0F_np1/config.info

#mpirun --report-bindings --map-by socket --bind-to core bin/runner -n 200 -v 3 -f plummer/initial_plummer/plN4096seed2723515521.h5 -C plummer/pl_N4096_sfc0D_np2/config.info

