# MilupHPC

This is the **API documentation for milupHPC** - a parallel smoothed particle hydrodynamics (**SPH**) targeting GPU cluster via CUDA-aware MPI.

Relevant links:

* [GitHub: milupHPC](https://github.com/michaelst98/miluphpc)
* [GitHub: miluphcuda](https://github.com/christophmschaefer/miluphcuda)


## About
________

The computational method Smoothed Particle Hydrodynamics is an expedient method for solving hydrodynamic equations and is in combination with a method for self-gravity applicable for astrophysical simulations. Since computationally expensive, appropriate methods and technologies need to be utilized. Especially parallelization is a promising approach. 

Thus, this work aims to describe and implement a multi-GPU Smoothed Particle Hydrodynamics code with self-gravity via the Barnes-Hut method using CUDA-aware MPI targeting GPU cluster. The Barnes-Hut method is an approximative N-body tree method subdividing the simulation domain hierarchically and approximating the potential of a distant group using a multipole expansion of the potential which reduces the overall complexity of the gravitational force computation. While the parallelization for shared memory systems is relatively straightforward, many challenges and problems arise for distributed memory parallelization. 

The approach presented here is based on the parallelization of the tree as constructed for the Barnes-Hut method. Every process' local tree contains the particles assigned to this process and in addition nodes identical for all processes forming a common coarse tree. This shared part of the tree enables processes to gain knowledge about the local trees on the other processes. Moreover, the tree constructed in this way permit the introduction of space-filling-curves which themselves are basis for an efficient domain decomposition and load-balancing. In addition, the parallel tree facilitates the determination of communication patterns which are necessary for calculating the forces arising from the Barnes-Hut method and SPH. Particle exchange via message passing is necessary since not all particles are accessible on all processes as the target architecture are distributed memory systems. Besides, this thesis investigates efficient GPU implementations of SPH and the Barnes-Hut algorithm including the gravitational force computation and fixed-radius near neighbor search. 

________


## Usage 

> you need to provide an appropriate H5 file as initial (particle) distribution
> 
> * see [GitHub: ParticleDistributor](https://github.com/MichaelSt98/ParticleDistributor)

* **compile** using the *Makefile* via: `make`
	* for debug: `make debug`
		* using *cuda-gdb*: `./debug/cuda_debug.sh`
	* for single-precision: `make single-precision` (default: double-precision)
* **run** via: `mpirun -np <number of processes> ./bin/runner -f <filename>`
	* e.g.: `mpirun -np 2 ./bin/runner -f examples/kepler.h5` 
* clean via: `make clean`, `make cleaner`
* rebuild via: `make remake` 	 

### Relevant preprocessor directives 

* adjust in *include/parameters.h* and rebuild

```c
#define DEBUGGING 0

/**
 * * `SAFETY_LEVEL 0`: almost no safety measures
 * * `SAFETY_LEVEL 1`: most relevant/important safety measures
 * * `SAFETY_LEVEL 2`: more safety measures, including assertions
 * * `SAFETY_LEVEL 3`: many security measures, including all assertions
 */
#define SAFETY_LEVEL 2

/// Dimension of the problem
#define DIM 3

/// [0]: natural units, [1]: SI units
#define SI_UNITS 1

/// [0]: rectangular (and not necessarily cubic domains), [1]: cubic domains
#define CUBIC_DOMAINS 1

/// Simulation with gravitational forces
#define GRAVITY_SIM 1

/// SPH simulation
#define SPH_SIM 1

/// integrate energy equation
#define INTEGRATE_ENERGY 0

/// integrate density equation
#define INTEGRATE_DENSITY 1

/// integrate smoothing length
#define INTEGRATE_SML 0

/// decouple smoothing length for pc integrator(s)
#define DECOUPLE_SML 0

/// variable smoothing length
#define VARIABLE_SML 1

/// correct smoothing length
#define SML_CORRECTION 0

/**
 * Choose the SPH representation to solve the momentum and energy equation:
 * * **SPH_EQU_VERSION 1:** original version with
 *     * HYDRO $dv_a/dt ~ - (p_a/rho_a**2 + p_b/rho_b**2)  \nabla_a W_ab$
 *     * SOLID $dv_a/dt ~ (sigma_a/rho_a**2 + sigma_b/rho_b**2) \nabla_a W_ab$
 * * **SPH_EQU_VERSION 2:** slighty different version with
 *     * HYDRO $dv_a/dt ~ - (p_a+p_b)/(rho_a*rho_b)  \nabla_a W_ab$
 *     * SOLID $dv_a/dt ~ (sigma_a+sigma_b)/(rho_a*rho_b)  \nabla_a W_ab$
 */
#define SPH_EQU_VERSION 1
```

### Config file settings

```
; IO RELATED
; ------------------------------------------------------
; output directory (will be created if it does not exist)
directory bb/

; outputRank (-1 corresponds to all)
outputRank -1
; omit logType::TIME for standard output
omitTime true
; create log file (including warnings, errors, ...)
log true
; create performance log
performanceLog true
; write particles to be sent to h5 file
particlesSent2H5 true


; INTEGRATOR RELATED
; ------------------------------------------------------
; integrator selection
; explicit euler [0], predictor-corrector euler [1]
integrator 1
; initial time step
timeStep 1e-4
; max time step allowed
maxTimeStep 1e9
; end time for simulation
;timeEnd 1.0
;timeEnd 5.0e7
timeEnd 6.9e11
;timeEnd 0.5e10;
;timeEnd 1e12
;timeEnd 6e-2

; SIMULATION RELATED
; ------------------------------------------------------
; space-filling curve selection
; lebesgue [0], hilbert [1]
sfc 1

; theta-criterion for Barnes-Hut (approximative gravity)
theta 0.5
; smoothing parameter for gravitational forces
;smoothing 0.025
;smoothing 3.0e12
;smoothing 1.6e13
;smoothing 3.2e13
smoothing 2.56e+20

; SPH smoothing kernel selection
; spiky [0], cubic spline [1], wendlandc2 [3], wendlandc4 [4], wendlandc6 [5]
smoothingKernel 1

; remove particles (corresponding to some criterion)
removeParticles true
; spherically [0], cubic [1]
removeParticlesCriterion 0
; allowed distance to center (0, 0, 0)
;removeParticlesDimension 1.0
removeParticlesDimension 3.6e14

; execute load balancing
loadBalancing false
; interval for executing load balancing (every Nth step)
loadBalancingInterval 1
; amount of bins for load balancing
loadBalancingBins 2000

; how much memory to allocate (1.0 -> all particles can in principle be on one process)
particleMemoryContingent 1.0

; calculate angular momentum (and save to output file)
calculateAngularMomentum true
; calculate (total) energy (and save to output file)
calculateEnergy true
; calculate center of mass (and save to output file)
calculateCenterOfMass false

```


### Command line arguments

* `./bin/runner -h` gives help:

```
Multi-GPU CUDA Barnes-Hut NBody/SPH code
Usage:
  HPC NBody [OPTION...]

  -n, --number-output-files arg
                                number of output files (default: 100)
  -t, --max-time-step arg       time step (default: -1.)
  -l, --load-balancing          load balancing
  -L, --load-balancing-interval arg
                                load balancing interval (default: -1)
  -C, --config arg              config file (default: config/config.info)
  -m, --material-config arg     material config file (default: 
                                config/material.cfg)
  -c, --curve-type arg          curve type (Lebesgue: 0/Hilbert: 1) 
                                (default: -1)
  -f, --input-file arg          File name (default: -)
  -v, --verbosity arg           Verbosity level (default: 0)
  -h, --help                    Show this help
```



