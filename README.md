# milupHPC

**High Performance Computing Smooth(ed) Particle Hydrodynamics**

The successor of [miluphcuda](https://github.com/christophmschaefer/miluphcuda) targeting GPU cluster via CUDA aware MPI.

## Current status

**Principally working, but not (fully) implemented yet!**

![](documents/sample.gif)

* See [MichaelSt98/SPH](https://github.com/MichaelSt98/SPH) for a proof of concept

## Usage 

* **compile** using the *Makefile* via: `make`
	* for debug: `make debug`
		* using *cuda-gdb*: `./debug/cuda_debug.sh`
* **run** via: `mpirun -np <number of processes> ./bin/runner`
* clean via: `make clean`, `make cleaner`
* rebuild via: `make remake` 	 