# Getting started

## Quickstart

1. check for the **[prerequisites](Prerequisites.md)**
2. **compile** using the [Makefile](../Makefile) (for help/instructions see [Compilation.md](Compilation.md)
	* with the corresponding/wanted preprocessor directives in `include/parameter.h`
3. provide/create/set
	* **initial particle distribution**
	* **config file**: especially output `directory` for the output files
	* **material config file** (for a simulation including SPH): especially `sml` in dependence of generated number of particles
3. **execute/run** via `mpirun -np <np> <binary> -n <#output files> -f <input hdf5 file> -C <config file> -m <material-config>`
	* or by using an adequate batch script for the used queing system
4. postprocess results


## Generally

<details>
 <summary>
   Preprocessor directives: parameter.h
 </summary>
 
* see `include/parameter.h`

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
</details>


<details>
 <summary>
   Input HDF5 file
 </summary>
 
* for **gravity only**
	* provide mass "m", position "x" and velocity "v" 

```
GROUP "/" {
   DATASET "m" {
      DATATYPE  H5T_IEEE_F64LE
      DATASPACE  SIMPLE { ( <num particles> ) / ( <num particles>  ) }
   }
   DATASET "v" {
      DATATYPE  H5T_IEEE_F64LE
      DATASPACE  SIMPLE { ( <num particles> , <dim> ) / ( <num particles> , <dim> ) }
   }
   DATASET "x" {
      DATATYPE  H5T_IEEE_F64LE
      DATASPACE  SIMPLE { ( <num particles> , <dim>  ) / (<num particles> , <dim>  ) }
   }
}
}
```

* **with SPH** provide (at least)
	* provide mass "m", material identifier "materialId", internal energy "u", position "x" and velocity "v" 

```
GROUP "/" {
   DATASET "m" {
      DATATYPE  H5T_IEEE_F64LE
      DATASPACE  SIMPLE { ( <num particles> ) / ( <num particles>  ) }
   }
   DATASET "materialId" {
      DATATYPE  H5T_STD_I8LE
      DATASPACE  SIMPLE { ( <num particles>  ) / ( <num particles>  ) }
   }
   DATASET "u" {
      DATATYPE  H5T_IEEE_F64LE
      DATASPACE  SIMPLE { ( <num particles>  ) / ( <num particles>  ) }
   }
   DATASET "v" {
      DATATYPE  H5T_IEEE_F64LE
      DATASPACE  SIMPLE { ( <num particles> , <dim> ) / ( <num particles> , <dim> ) }
   }
   DATASET "x" {
      DATATYPE  H5T_IEEE_F64LE
      DATASPACE  SIMPLE { ( <num particles> , <dim>  ) / (<num particles> , <dim>  ) }
   }
}
}
```
 
</details>


<details>
 <summary>
   Config file
 </summary>
 
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
; explicit euler [0], predictor-corrector euler [1], leapfrog [2]
integrator 1
; initial time step
timeStep 1e-4
; max time step allowed
maxTimeStep 1e-4
; end time for simulation
;timeEnd 6e-2

; SIMULATION RELATED
; ------------------------------------------------------
; space-filling curve selection
; lebesgue [0], hilbert [1]
sfc 1

; theta-criterion for Barnes-Hut (approximative gravity)
theta 0.5
; smoothing parameter for gravitational forces
smoothing 2.56e+20

; SPH smoothing kernel selection
; spiky [0], cubic spline [1], wendlandc2 [3], wendlandc4 [4], wendlandc6 [5]
smoothingKernel 1

; remove particles (corresponding to some criterion)
removeParticles true
; spherically [0], cubic [1]
removeParticlesCriterion 0
; allowed distance to center (0, 0, 0)
removeParticlesDimension 3.6e14

; execute load balancing
loadBalancing false
; interval for executing load balancing (every Nth step)
loadBalancingInterval 1

; how much memory to allocate (1.0 -> all particles can in principle be on one process)
particleMemoryContingent 1.0

; calculate angular momentum (and save to output file)
calculateAngularMomentum true
; calculate (total) energy (and save to output file)
calculateEnergy true
; calculate center of mass (and save to output file)
calculateCenterOfMass false

; IMPLEMENTATION SELECTION
; ------------------------------------------------------
; force version for gravity (use [2])
; burtscher [0], burtscher without presorting [1], miluphcuda with presorting [2],
; miluphcuda without presorting [3], miluphcuda shared memory (experimental) [4]
gravityForceVersion 0
; fixed radius NN version for SPH (use [0])
; normal [0], brute-force [1], shared-memory [2], within-box [3]
sphFixedRadiusNNVersion 3
```

</details>


<details>
 <summary>
   Material config file
 </summary>
 
```
materials = (
{
    ID = 0
    name = "IsothermalGas"
    #sml = 1e12
    sml = 5.2e11
    interactions = 50
    artificial_viscosity = { alpha = 1.0; beta = 2.0; };
    eos = {
        type = 3
    };
}
);

...
```
 
</details>


<details>
 <summary>
   Command line arguments
 </summary>
 
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

</details>


## Test cases

**Available test cases including instructions can be found within [testcases/](../testcases/).**

Currently available:

* the [Plummer/](../testcases/plummer/) test case ([README](../testcases/plummer/README.md))
* the [Taylor-von Neumann-Sedov blast wave/](../testcases/sedov/) test case ([README](../testcases/sedov/README.md))

