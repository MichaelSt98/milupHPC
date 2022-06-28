# Sedov test case

The Taylor-von Neumann-Sedov blast wave describes a blast wave as result of a strong explosion and is a standard test case for SPH codes. A semi-analytical, self-similar solution exists, usable for validation. The problem is set up in a uniform periodically box [-0.5, 0.5] and a high internal energy is induced at the center and smoothed over a smoothing length. An adiabatic EOS with gamma = 5/3 is used and the internal energy equation is integrated.
_____

## Overview

0. check the [prerequisites](../../documents/Prerequisites.md)
1. **compile** according to the test case
	* either copy [parameter.h](parameter.h) to `../../include/`
	* or set
		* `#define SI_UNITS 0`
		* `#define GRAVITY_SIM 0`
		* `#define SPH_SIM 1`
		* `#define INTEGRATE_ENERGY 1`
		* `#define INTEGRATE_DENSITY 0`
		* `#define INTEGRATE_SML 1` or `0`
		* `#define DECOUPLE_SML 1` or `0`
		* `#define VARIABLE_SML 0`
		* `#define SML_CORRECTION 0`
		* `#define ARTIFICIAL_VISCOSITY 1`
		* `#define BALSARA_SWITCH 0`
2. **generate initial** HDF5 **particle distribution** (choose amount of particles to be simulated)
	* e.g. execute `initial_sedov.py`
3. **adjust [material.cfg](material.cfg)**: especially/at least `sml` in dependence of particles to be simulated
4. adjust [config.info](config.info): especially output `directory` for the output files
5. **execute simulation** via `mpirun -np <num processes> bin/runner -n <num output files> -f <generated initial particle distribution> -C testcases/sedov/config.info -m testcases/sedov/material.cfg` 
	* assuming execution from the `milupHPC` root directory
6. possibly [postprocess](../../postprocessing/README.md)

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
#define SAFETY_LEVEL 2

/// Dimension of the problem
#define DIM 3
#define power_two(x) (1 << (x))
#define POW_DIM power_two(DIM)

/// [0]: natural units, [1]: SI units
#define SI_UNITS 0

/// [0]: rectangular (and not necessarily cubic domains), [1]: cubic domains
#define CUBIC_DOMAINS 1

/// Simulation with gravitational forces
#define GRAVITY_SIM 0

/// SPH simulation
#define SPH_SIM 1

/// integrate energy equation
#define INTEGRATE_ENERGY 1

/// integrate density equation
#define INTEGRATE_DENSITY 0

/// integrate smoothing length
#define INTEGRATE_SML 1

/// decouple smoothing length for pc integrator(s)
#define DECOUPLE_SML 1

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

* necessary Python3 packages:
	* seagen
	* numpy
	* h5py 

Within [initial_sedov.py](initial_sedov.py) adjust:

```python
N = 61  # num particles = N**3

sml = 0.041833  # for n = 61**3  = 226981  = ca. 2e5
# sml = 0.031375  # for n = 81**3  = 531441  = ca. 5e5
# sml = 0.0251    # for n = 101**3 = 1030301 = ca. 1e6
# sml = 0.02      # for n = 126**3 = 2000376 = ca 2e6
# sml = 0.01476   # for n = 171**3 = 5000211 = ca 5e6
```

* usage: `python3 initial_sedov.py`
* if wanted rename generated HDF5 file

### Material config file

* adjust the `sml` in dependence of the number of particles 

Within [material.cfg](material.cfg) adjust:

```
materials = (
{
    ID = 0
    name = "IdealGas"
    sml = 0.041833  # for n = 61**3  = 226981  = ca. 2e5
    # sml = 0.031375  # for n = 81**3  = 531441  = ca. 5e5
    # sml = 0.0251    # for n = 101**3 = 1030301 = ca. 1e6
    # sml = 0.02      # for n = 126**3 = 2000376 = ca 2e6
    # sml = 0.01476   # for n = 171**3 = 5000211 = ca 5e6
    interactions = 50
    artificial_viscosity = { alpha = 1.0; beta = 2.0; };
    eos = {
	    polytropic_K = 0.0
        polytropic_gamma = 1.6666666
	type = 9
    };
}
);
```

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
integrator 1
; initial time step
timeStep 0.0005
; max time step allowed
maxTimeStep 1e-3
; end time for simulation
timeEnd 0.06

; SIMULATION RELATED
; ------------------------------------------------------
; space-filling curve selection
; lebesgue [0], hilbert [1]
sfc 0

; theta-criterion for Barnes-Hut (approximative gravity)
theta 1.0
; smoothing parameter for gravitational forces
;smoothing 0.032
smoothing 0.001024

; SPH smoothing kernel selection
; spiky [0], cubic spline [1], wendlandc2 [3], wendlandc4 [4], wendlandc6 [5]
smoothingKernel 1

; remove particles (corresponding to some criterion)
removeParticles false
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
sphFixedRadiusNNVersion 3
```

</details>

## Postprocessing

For more information refer to [postprocessing](../../postprocessing/README.md)

* some scripts are available within [postprocessing/](../../postprocessing/) 
	* [postprocessing/PlotSedov.py](../../postprocessing//PlotSedov.py): plot density, pressure and internal energy in dependence of the radius for the sedov test case, including the semi-analytical solution
		* usage: e.g.  `./PlotSedov.py -i <input HDF5 particle file> -o <output directory> -a -p 1`
		* for applying script for all output files: e.g. `find <directory with input files> -type f -name 'ts*.h5' -print0 | parallel -0 -j 1 ./PlotSedov.py -i {} -o <output directory> -a -p 1 \;`
			* combining those to a video: `ffmpeg -r 24 -i <input directory>/ts%06d.h5.png -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -pix_fmt yuv420p -vcodec libx264 -y -an <output directory>/evolution.mp4`




