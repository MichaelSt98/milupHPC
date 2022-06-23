# Getting started

## Quickstart

1. check for the [prerequisites](Prerequisites.md)
2. compile using the [Makefile](../Makefile) (for help/instructions see [Compilation.md](Compilation.md)
	* with the corresponding/wanted preprocessor directives in `include/parameter.h`
3. provide/create/set
	* initial particle distribution
	* config file
	* material config file (for a simulation including SPH)
3. run via `mpirun -np <np> <binary> -n <#output files> -f <input hdf5 file> -C <config file> -m <material-config>`
	* or by using an adequate batch script for the used queing system
4. post-process results


