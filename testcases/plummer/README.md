# Plummer test case

The Plummer model or Plummer sphere is a distribution that remains stable over time as shown in *Plummer (1911)*. The initial distributions used for the simulation are generated as described in *Aarseth et al. (1979)*.

<img src="../../documents/4proc_plummer_dynamic.gif" alt="Plummer sample" width="50%"/>

____

## Overview

0. check the [prerequisites](../../documents/Prerequisites.md)
1. **compile** according to the test case
	* either copy [parameter.h](parameter.h) to `../../include/`
	* or set
		* `#define SI_UNITS 0`
		* `#define GRAVITY_SIM 1`
		* `#define SPH_SIM 0`
		* `#define INTEGRATE_ENERGY 0`
		* `#define INTEGRATE_DENSITY 0`
		* `#define INTEGRATE_SML 0`
		* `#define DECOUPLE_SML 0`
		* `#define VARIABLE_SML 0`
		* `#define SML_CORRECTION 0`
		* `#define ARTIFICIAL_VISCOSITY 0`
		* `#define BALSARA_SWITCH 0`
2. **generate initial** HDF5 **particle distribution** (choose amount of particles to be simulated)
	* e.g. using [GitHub: ParticleDistributor](https://github.com/MichaelSt98/ParticleDistributor)
3. adjust [config.info](config.info): especially output `directory` for the output files
4. **execute simulation** via `mpirun -np <num processes> bin/runner -n <num output files> -f <generated initial particle distribution> -C testcases/plummer/config.info` 
	* assuming execution from the `milupHPC` root directory
5. possibly [postprocess](../../postprocessing/README.md)

_____

### Compilation

* See [Compilation.md](../../documents/Compilation.md) for help

A proper version of `parameter.h` is shown in the following. It is important to have/copy this version to the `include/` directory before the compilation.

<details>
<summary>
parameter.h
</summary>

```cpp
#ifndef MILUPHPC_PARAMETER_H
#define MILUPHPC_PARAMETER_H

#include <limits>

/**
 * Type definitions
 * * `real` corresponds to floating point precision for whole program
 * * `keyType` influences the maximal tree depth
 *     * maximal tree depth: (sizeof(keyType) - (sizeof(keyType) % DIM))/DIM
 */
#ifdef SINGLE_PRECISION
    typedef float real;
#else
    typedef double real;
#endif
typedef int integer;
typedef unsigned long keyType;
typedef int idInteger;

#include <iostream>

//#define theta 0.5

#define MAX_LEVEL 21

#define DEBUGGING 0

/**
 * * `SAFETY_LEVEL 0`: almost no safety measures
 * * `SAFETY_LEVEL 1`: most relevant/important safety measures
 * * `SAFETY_LEVEL 2`: more safety measures, including assertions
 * * `SAFETY_LEVEL 3`: many security measures, including all assertions
 */
#define SAFETY_LEVEL 1

/// Dimension of the problem
#define DIM 3
#define power_two(x) (1 << (x))
#define POW_DIM power_two(DIM)

/// [0]: natural units, [1]: SI units
#define SI_UNITS 0

/// [0]: rectangular (and not necessarily cubic domains), [1]: cubic domains
#define CUBIC_DOMAINS 1

/// Simulation with gravitational forces
#define GRAVITY_SIM 1

/// SPH simulation
#define SPH_SIM 0

/// integrate energy equation
#define INTEGRATE_ENERGY 0

/// integrate density equation
#define INTEGRATE_DENSITY 0

/// integrate smoothing length
#define INTEGRATE_SML 0

/// decouple smoothing length for pc integrator(s)
#define DECOUPLE_SML 0

/// variable smoothing length
#define VARIABLE_SML 0

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

// deprecated flag
#define ARTIFICIAL_VISCOSITY 1
#define BALSARA_SWITCH 0

// to be (fully) implemented flags
#define AVERAGE_KERNELS 0
#define DEAL_WITH_TOO_MANY_INTERACTIONS 0
#define SHEPARD_CORRECTION 0
#define SOLID 0
#define NAVIER_STOKES 0
#define ARTIFICIAL_STRESS 0
#define POROSITY 0
#define ZERO_CONSISTENCY 0
#define LINEAR_CONSISTENCY 0
#define FRAGMENTATION 0
#define PALPHA_POROSITY 0
#define PLASTICITY 0
#define KLEY_VISCOSITY 0

#define KEY_MAX ULONG_MAX
//TODO: make domain list size to run time constant
#define DOMAIN_LIST_SIZE 512
#define MAX_DEPTH 128
#define MAX_NUM_INTERACTIONS 180
#define NUM_THREADS_LIMIT_TIME_STEP 256
#define NUM_THREADS_CALC_CENTER_OF_MASS 256

// Courant (CFL) number (note that our sml is defined up to the zero of the kernel, not half of it)
#define COURANT_FACT 0.4

#define FORCES_FACT 0.2

constexpr real dbl_max = std::numeric_limits<real>::max();
#define DBL_MAX dbl_max;

namespace Constants {
    constexpr real G = 6.67430e-11;
}

typedef struct SimulationParameters {
    std::string directory;
    std::string logDirectory;
    int verbosity;
    bool timeKernels;
    int numOutputFiles;
    real timeStep;
    real maxTimeStep;
    real timeEnd;
    bool loadBalancing;
    int loadBalancingInterval;
    int loadBalancingBins;
    std::string inputFile;
    std::string materialConfigFile;
    int outputRank;
    bool performanceLog;
    bool particlesSent2H5;
    int sfcSelection;
    int integratorSelection;
//#if GRAVITY_SIM
    real theta;
    real smoothing;
    int gravityForceVersion;
//#endif
//#if SPH_SIM
    int smoothingKernelSelection;
    int sphFixedRadiusNNVersion;
//#endif
    bool removeParticles;
    int removeParticlesCriterion;
    real removeParticlesDimension;
    int bins;
    bool calculateAngularMomentum;
    bool calculateEnergy;
    bool calculateCenterOfMass;
    real particleMemoryContingent;
    int domainListSize;
} SimulationParameters;

struct To
{
    enum Target
    {
        host, device
    };
    Target t_;
    To(Target t) : t_(t) {}
    operator Target () const {return t_;}
private:
    template<typename T>
    operator T () const;
};

struct Smoothing
{
    enum Kernel
    {
        spiky, cubic_spline, wendlandc2, wendlandc4, wendlandc6
    };
    Kernel t_;
    Smoothing(Kernel t) : t_(t) {}
    operator Smoothing () const {return t_;}
private:
    template<typename T>
    operator T () const;
};

struct Execution
{
    enum Location
    {
        host, device
    };
    Location t_;
    Execution(Location t) : t_(t) {}
    operator Location () const {return t_;}
private:
    template<typename T>
    operator T () const;
};

struct Curve
{
    enum Type
    {
        lebesgue, hilbert
    };
    Type t_;
    Curve(Type t) : t_(t) {}
    operator Type () const {return t_;}
    //friend std::ostream& operator<<(std::ostream& out, const Curve::Type curveType);
private:
    template<typename T>
    operator T () const;
};

struct IntegratorSelection
{
    enum Type
    {
        explicit_euler, predictor_corrector_euler, leapfrog
    };
    Type t_;
    IntegratorSelection(Type t) : t_(t) {}
    operator Type () const {return t_;}
private:
    template<typename T>
    operator T () const;
};

/// implemented equation of states
enum EquationOfStates {
    //EOS_TYPE_ACCRETED = -2, /// special flag for particles that got accreted by a gravitating point mass
    //EOS_TYPE_IGNORE = -1, /// particle is ignored
    EOS_TYPE_POLYTROPIC_GAS = 0, /// polytropic EOS for gas, needs polytropic_K and polytropic_gamma in material.cfg file
    //EOS_TYPE_MURNAGHAN = 1, /// Murnaghan EOS for solid bodies, see Melosh "Impact Cratering", needs in material.cfg: rho_0, bulk_modulus, n
    //EOS_TYPE_TILLOTSON = 2, /// Tillotson EOS for solid bodies, see Melosh "Impact Cratering", needs in material.cfg: till_rho_0, till_A, till_B, till_E_0, till_E_iv, till_E_cv, till_a, till_b, till_alpha, till_beta; bulk_modulus and shear_modulus are needed to calculate the sound speed and crack growth speed for FRAGMENTATION
    EOS_TYPE_ISOTHERMAL_GAS = 3, /// this is pure molecular hydrogen at 10 K
    //EOS_TYPE_REGOLITH = 4, /// The Bui et al. 2008 soil model
    //EOS_TYPE_JUTZI = 5, /// Tillotson EOS with p-alpha model by Jutzi et al.
    //EOS_TYPE_JUTZI_MURNAGHAN = 6, /// Murnaghan EOS with p-alpha model by Jutzi et al.
    //EOS_TYPE_ANEOS = 7, /// ANEOS (or tabulated EOS in ANEOS format)
    //EOS_TYPE_VISCOUS_REGOLITH = 8, /// describe regolith as a viscous material -> EXPERIMENTAL DO NOT USE
    EOS_TYPE_IDEAL_GAS = 9, /// ideal gas equation, set polytropic_gamma in material.cfg
    //EOS_TYPE_SIRONO = 10, /// Sirono EOS modifed by Geretshauser in 2009/10
    //EOS_TYPE_EPSILON = 11, /// Tillotson EOS with epsilon-alpha model by Wuennemann, Collins et al.
    EOS_TYPE_LOCALLY_ISOTHERMAL_GAS = 12, /// locally isothermal gas: \f$ p = c_s^2 \times \varrho \f$
    //EOS_TYPE_JUTZI_ANEOS = 13/// ANEOS EOS with p-alpha model by Jutzi et al.
};

struct Entry
{
    enum Name
    {
        x,
#if DIM > 1
        y,
#if DIM == 3
        z,
#endif
#endif
        mass
    };
    Name t_;
    Entry(Name t) : t_(t) {}
    operator Name () const {return t_;}
private:
    template<typename T>
    operator T () const;
};

#endif //MILUPHPC_PARAMETER_H
```

</details>

### Generate initial particle distribution

See [GitHub: ParticleDistributor](https://github.com/MichaelSt98/ParticleDistributor).

* `git clone https://github.com/MichaelSt98/ParticleDistributor.git`
* compile via `make`
    * make necessary changes to the *Makefile*
* generate input Plummer file via e.g. `./bin/runner -N 10000 -f pl -d 0`
    * get help via `./bin/runner -h`


### Config file

Within [config.info](config.info) adjust `directory` and possibly more:


<details>
<summary>
config.info
</summary>

```
; IO RELATED
; ------------------------------------------------------
; output directory (will be created if it does not exist)
directory <TODO: directory>

; outputRank (-1 corresponds to all)
outputRank -1

; omit logType::TIME for standard output
omitTime true

; create log file (including warnings, errors, ...)
log false

; create performance log
performanceLog true

; write particles to be sent to h5 file
particlesSent2H5 false


; INTEGRATOR RELATED
; ------------------------------------------------------
; integrator selection
; explicit euler [0], predictor-corrector euler [1]
integrator 0
; initial time step
timeStep 0.025
; max time step allowed
maxTimeStep 0.025
; end time for simulation
timeEnd 25.0

; SIMULATION RELATED
; ------------------------------------------------------
; space-filling curve selection
; lebesgue [0], hilbert [1]
sfc 0

; theta-criterion for Barnes-Hut (approximative gravity)
theta 0.5
; smoothing parameter for gravitational forces
;smoothing 0.032
;smoothing 0.001024
smoothing 0.001024

; SPH smoothing kernel selection
; spiky [0], cubic spline [1], wendlandc2 [3], wendlandc4 [4], wendlandc6 [5]
smoothingKernel 1

; remove particles (corresponding to some criterion)
removeParticles true
; spherically [0], cubic [1]
removeParticlesCriterion 0
; allowed distance to center (0, 0, 0)
removeParticlesDimension 10.0

; execute load balancing
loadBalancing false
; interval for executing load balancing (every Nth step)
loadBalancingInterval 1
; amount of bins for load balancing
loadBalancingBins 2000

; how much memory to allocate (1.0 -> all particles can in principle be on one process)
particleMemoryContingent 1.0

; calculate angular momentum (and save to output file)
calculateAngularMomentum false
; calculate (total) energy (and save to output file)
calculateEnergy false
; calculate center of mass (and save to output file)
calculateCenterOfMass true


; THESE SHOULD PROBABLY NOT EXIST IN A PRODUCTION VERSION
; ------------------------------------------------------
; ------------------------------------------------------
; force version for gravity (use [0] or [2])
; burtscher [0], burtscher without presorting [1], miluphcuda with presorting [2],
; miluphcuda without presorting [3], miluphcuda shared memory (NOT working properly) [4]
gravityForceVersion 0
; fixed radius NN version for SPH (use [0])
; normal [0], brute-force [1], shared-memory [2], within-box [3]
sphFixedRadiusNNVersion 0
```

</details>

## Postprocessing

For more information refer to [postprocessing](../../postprocessing/README.md)

* some scripts are available within [postprocessing/](../../postprocessing/) 
	* [postprocessing/PlotPlummer.py](../../postprocessing//PlotPlummer.py): plot mass quantiles for the plummer test case
		* usage: e.g. `./PlotPlummer.py -Q -d <input data directory> -o <output directory>` 

producing a plot like this:

<img src="../../documents/figures/long_pl_N10000000_sfc1D_np4_mass_quantiles.png" alt="Plummer sample" width="100%"/>