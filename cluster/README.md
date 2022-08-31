# Cluster & Queing systems

## PBS

* a **submit script** could look like this

```
#!/bin/bash
#PBS -N N1e6SSG
#PBS -l nodes=1:ppn=4:gpus=4:exclusive_process
#PBS -l walltime=00:12:00
#PBS -l pmem=10gb
#PBS -q tiny
#PBS -m aeb -M mail@gmail.com
source ~/.bashrc

# Loading modules
module load compiler/gnu/8.3
module load devel/cuda/10.1
...

# Going to working directory
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/local/lib
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export PSM3_MULTI_EP=1

nvidia-smi

# Starting program
mpirun --bind-to core --map-by core --report-bindings bin/runner -n 50 -f <input particle file> -C <config file>
```

* **executed** via `qsub <submit script>`
* information via `qstat`
* delete via `qdel`

## Slurm


* [Slurm documentation](https://slurm.schedmd.com/documentation.html)
* a **submit script** could look like this

```
#!/bin/bash
#SBATCH --job-name=<job name>
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --mem=10G
#SBATCH --gres=gpu:4
source ~/.bashrc

# Loading modules
module load compiler/gnu/10.2
module load devel/cuda/11.4
...

# Going to working directory
cd $SLURM_SUBMIT_DIR
echo $SLURM_SUBMIT_DIR

# export OMPI_MCA_pml=ucx
# LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/local/lib

nvidia-smi

mpirun -np 4 bin/runner -n 50 -f <input particle file> -C <input particle file> -m <material config file>
```

* **executed** via `sbatch -p <queue> <submit script>`
* information via `squeue`
* delete via `scancel`
