# Binac 

## Links

* [bwhpc](https://wiki.bwhpc.de/e/Main_Page)

## Accessing

* Login: 
	* ssh 
		* `ssh <UserID>@login01.binac.uni-tuebingen.de`
		* `ssh <UserID>@login01.binac.uni-tuebingen.de`
		* `ssh <UserID>@login01.binac.uni-tuebingen.de`
	* generate and enter Code using App (FreeOTP)
	* enter password
	* navigate to directory (e.g.: `cd /beegfs/work/tu_zxmjo49`)


## Compiling

### Modules

* `$ man module` for more information
	* loaded modules: `module list`
	* available modules: `module avail` 

* modules to load:
	* `module load devel/cuda/10.1`
	* `module load mpi/openmpi/3.1-gnu-8.3` 

### Linking

```
CXXFLAGS    := -std=c++11 -I/opt/bwhpc/common/lib/hdf5/1.10.7-openmpi-3.1-gnu-9.2/include -I/opt/bwhp
c/common/lib/boost/1.69.0/include
```

```
LFLAGS      := -std=c++11 -L/opt/bwhpc/common/lib/hdf5/1.10.7-openmpi-3.1-gnu-9.2/lib -L/opt/bwhpc/co
mmon/lib/boost/1.69.0/lib -lboost_filesystem -lboost_system -lhdf5  #-L/usr/lib/x86_64-linux-gnu/hdf5
/openmpi -lhdf5
```

### Batch job

Used Queing system: [OpenPBS](https://www.openpbs.org/)

* `#PBS -l nodes=1:ppn=4:gpus=4:default` for four GPUs on one node
	* `export CUDA_VISIBLE_DEVICES=0,1,2,3`

```
#!/bin/bash
#PBS -N N1e6SSG
#PBS -l nodes=1:ppn=2
#PBS -l walltime=00:02:00
#PBS -l pmem=1gb
#PBS -q tiny
#PBS -m aeb -M johannes-stefan.martin@student.uni-tuebingen.de
source ~/.bashrc

# Loading modules
module load mpi/openmpi/3.1-gnu-9.2
module load lib/hdf5/1.10.7-openmpi-3.1-gnu-9.2
module load lib/boost/1.69.0

# Going to working directory
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

# Starting program
mpirun --bind-to core --map-by core -report-bindings bin/runner
```


