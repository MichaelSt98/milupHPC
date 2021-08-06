# Miluphcuda

Miluphcuda is the CUDA port of the original miluph code, therefore a SPH hydro and solid code, including self-gravity (via Barnes-Hut tree) and porosity models.

* [GitHub: christophmschaefer/miluphcuda](https://github.com/christophmschaefer/miluphcuda)
* [Documentation (Doxygen)](https://christophmschaefer.github.io/miluphcuda/index.html)
* [Documentation (PDF)](https://christophmschaefer.github.io/miluphcuda/miluphcuda_documentation.pdf) - mostly outdated

## Project structure

| File/Directory                   | Content                       | Comment                                    |
| -------------------------------- | ----------------------------- | -------------- |
| *doc/* | documentation | - |
| *examples/* | examples | - | 
| *material-config/* | material config files | - |
| *utils/* | utitlities | - |
| `aneos.cu/h` | ANEOS EOS | - |
| `artificial_stress.cu/h` | artificial stress | - |
| `boundary.cu/h` | boundary conditions | - | 
| `checks.h` | checks (combinations of) settings | settings via preprocessor directives | 
| `config_parameter.cu/h` | allocate/intialize config/material parameters | - | 
| `coupled_heun_rk4_sph_nbody.cu/h` | **Integrator** heun for SPH particles and RK4 for N-Bodys | - | 
| `cuda_utils.h` | CUDA macros | - | 
| `damage.cu/h` | damage handling for **Fragmentation** | - | 
| `density.cu/h` | density | **Basic SPH** | 
| `device_tools.cu/h` | GPU device information | - | 
| `euler.cu/h` | **Integrator** euler | - | 
| `extrema.cu/h` | determine extrema | extrema for pressure, density, energy, soundspeed | 
| `gravity.cu/h` | handling gravitational forces | direct self-gravity, and self-gravity via Barnes-Hut | 
| `internal_forces.cu/h` | internal forces | ? |
| `io.cu/h` | input/output (operations) | - | 
| `kernel.cu/h` | SPH kernel(s) | - | 
| `linalg.cu/h` | linear algebra | - |  
| `little_helpers.cu/h` | helper functions/(CUDA) kernels | checking for NaNs, sigterm handler, printing tensorial correction matrix |
| `memory_handling.cu/h` | (device) memory (de)allocation | device memory and pinned (host) memory via `cudaMallocHost()` |  
| `miluph.cu/h` | *main* function | - | 
| `parameter.h` | settings via preprocessor directives | - | 
| `plasticity.cu/h` | plasticity (models) | - | 
| `porosity.cu/h` | porosity (models) | - | 
| `predictor_corrector.cu/h` | **Integrator** predictor-corrector | - | 
| `predictor_corrector_euler.cu/h` | **Integrator** predictor-corrector | - | 
| `pressure.cu/h` | pressure | **Basic SPH** extended using all kind of other information/models | 
| `rhs.cu/h` | right hand side needed (several times) for time integration | - |
| `rk2adaptive.cu/h` | **Integrator** RK2 adaptive | - |  
| `sinking.cu/h` | particle sinking/accretion | - |
| `soundspeed.cu/h` | speed of sound | - |
| `stress.cu/h` | (determining) stress tensor | - |
| `timeintegration.cu/h` | time integration calling (corresponding) integrator | - |
| `tree.cu/h` | tree construction, traversing, NNS, ... | - |
| `velocity.cu/h` | setting location changes | ? |
| `viscosity.cu/h` | viscosity | shear stress and kinematic viscosity |
| `xsph.cu/h` | (alternative) artificial viscosity model | second derivative not needed |        

## Settings

### Preprocessor directives

```cpp
// Dimension of the problem
#define DIM 3

#define SOLID 1
#define HYDRO 0
#define REAL_HYDRO 0 // set additionally p to 0 if p < 0

// add additional point masses to the simulation, read from file <filename>.mass
#define GRAVITATING_POINT_MASSES 0

// sink particles (set point masses to be sink particles)
#define PARTICLE_ACCRETION 0 
#define UPDATE_SINK_VALUES 0 // add to sink the quantities of the accreted particle: mass, velocity and COM

// integrate the energy equation
#define INTEGRATE_ENERGY 0 // needed for SOLID with Tillotson and ANEOS

// integrate the continuity equation
#define INTEGRATE_DENSITY 1 // otherwise: the density will be calculated using the standard SPH sum \sum_i m_j W_ij

// adds viscosity to the Euler equation
#define NAVIER_STOKES 0
// choose between two different viscosity models
#define SHAKURA_SUNYAEV_ALPHA 0
#define CONSTANT_KINEMATIC_VISCOSITY 0
// artificial bulk viscosity according to Schaefer et al. (2004)
#define KLEY_VISCOSITY 0

// This is the damage model following Benz & Asphaug (1995). Set FRAGMENTATION to activate it.
// The damage acts always on pressure, but only on deviator stresses if 
#define FRAGMENTATION 0
#define DAMAGE_ACTS_ON_S 0

// Choose the SPH representation to solve the momentum and energy equation
#define SPH_EQU_VERSION 1

#define ARTIFICIAL_STRESS 0

// standard SPH alpha/beta viscosity
#define ARTIFICIAL_VISCOSITY 1
// Balsara switch: lowers the artificial viscosity in regions without shocks
#define BALSARA_SWITCH 0

// INVISCID SPH (see Cullen & Dehnen paper)
#define INVISCID_SPH 0

// consistency switches
#define SHEPARD_CORRECTION 0 // for zeroth order consistency
#define TENSORIAL_CORRECTION 1 // for linear consistency

// Available plastic flow conditions
#define VON_MISES_PLASTICITY 0
#define DRUCKER_PRAGER_PLASTICITY 0
#define MOHR_COULOMB_PLASTICITY 0
#define COLLINS_PLASTICITY 0
#define COLLINS_PLASTICITY_INCLUDE_MELT_ENERGY 0
#define COLLINS_PLASTICITY_SIMPLE 0
#define VISCOUS_REGOLITH 0
#define PURE_REGOLITH 0
#define JC_PLASTICITY 0

// Porosity models:
#define PALPHA_POROSITY 0         // pressure depends on distention
#define STRESS_PALPHA_POROSITY 0  // deviatoric stress is also affected by distention
// Sirono model modified by Geretshauser (2009/10)
#define SIRONO_POROSITY 0
// eps-alpha model implemented following Wuennemann
#define EPSALPHA_POROSITY 0
#define MAX_NUM_FLAWS 1 // only required for FRAGMENTATION
// maximum number of interactions per particle -> fixed array size
#define MAX_NUM_INTERACTIONS 128

#define VARIABLE_SML 0
#define FIXED_NOI 0
#define INTEGRATE_SML 0
#define READ_INITIAL_SML_FROM_PARTICLE_FILE 0
#define SML_CORRECTION 0

// if set to 0, h = (h_i + h_j)/2  is used to calculate W_ij
// if set to 1, W_ij = ( W(h_i) + W(h_j) ) / 2
#define AVERAGE_KERNELS 1


// important switch: if the simulations yields at some point too many interactions for
// one particle (given by MAX_NUM_INTERACTIONS), then its smoothing length will be set to 0
// and the simulation continues. It will be announced on *stdout* when this happens
// if set to 0, the simulation stops in such a case unless DEAL_WITH_TOO_MANY_INTERACTIONS is used
#define TOO_MANY_INTERACTIONS_KILL_PARTICLE 0
// important switch: if the simulations yields at some point too many interactions for
// one particle (given by MAX_NUM_INTERACTIONS), then its smoothing length will be lowered until
// the interactions are lower than MAX_NUM_INTERACTIONS
#define DEAL_WITH_TOO_MANY_INTERACTIONS 0

// additional smoothing of the velocity field, hinders particle penetration (see Morris and Monaghan 1984)
#define XSPH 0

// EXPERIMENTAL
#define BOUNDARY_PARTICLE_ID -1
#define GHOST_BOUNDARIES 0

// IO options
#define HDF5IO 1    // use HDF5 (needs libhdf5-dev and libhdf5)
#define MORE_OUTPUT 0   //produce additional output to HDF5 files (p_max, p_min, rho_max, rho_min); only useful when HDF5IO is set
#define MORE_ANEOS_OUTPUT 0 // produce additional output to HDF5 files (T, cs, entropy, phase-flag); only useful when HDF5IO is set; set only if you use the ANEOS eos, but currently not supported for porosity+ANEOS
#define OUTPUT_GRAV_ENERGY 0    // compute and output gravitational energy (at times when output files are written); of all SPH particles (and also w.r.t. gravitating point masses and between them); direct particle-particle summation, not tree; option exists to control costly computation for high particle numbers
#define BINARY_INFO 0   // generates additional output file (binary_system.log) with info regarding binary system: semi-major axis, eccentricity if GRAVITATING_POINT_MASSES == 1
```

## Kernels

* wendlandc2
* wendlandc4
* wendlandc6
* cubic_spline
* spiky
* quartic_spline

## Integrators

* rk2_adaptive
* euler
* monaghan_pc
* heun_rk4
* euler_pc

## Physics

sorted by priorization:

| Prio | Keyword                          | Description                                                                                             | Notes                                                                     | Links                                               |
|------|----------------------------------|---------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|-----------------------------------------------------|
| 0    | **Hydro**                        | standard inviscid Euler equations                                                                       | no shear forces                                                           |                                                     |
| 0    | **Solid**                        | simulate solid behavior                                                                                 | stress tensor                                                             |                                                     |
| 0    | **self-gravity**                 | gravitational forces (long range force)                                                                 | approximative implementation via Barnes-Hut method                        | [NNS](https://github.com/MichaelSt98/NNS/tree/main) |
| 0    | **Integrate energy**             | integration of energy equation in order to conserve internal energy                                     |                                                                           |                                                     |
| 0    | **Integrate density**            | integration of density                                                                                  |                                                                           |                                                     |
| 0    | **Fragmentation**                | Fragmentation as damage model                                                                           | depends on Solid                                                          | Benz & Asphaug 1994                                 |
| 0    | **artificial viscosity**         | prevent particle interpenetration                                                                       |                                                                           | Monaghan 1992                                       |
| 0    | **artificial stress**            | overcome tensile instability                                                                            | required for fully elastic solids                                         |                                                     |
| 0    | **Shepard correction**           | provide zeroth order and linear consistency                                                             |                                                                           |                                                     |
| 0    | **Tensorial correction**         | provide zeroth order and linear consistency                                                             |                                                                           |                                                     |
| 0    | **Plasiticity**                  | Plasticity models: von Mises plasticity                                                                 | depends on Solid                                                          |                                                     |
| 0    | **Porosity**                     | Porosity models: p-alpha porosity                                                                       | depends on Solid                                                          |                                                     |
| 0    | **variable SML**                 | variable and possibly different smoothing lengths for different particles                               | including FIXED\_NOI, INTEGRATE\_SML, READ\_INITIAL\_FROM\_PARTICLE\_FILE |                                                     |
| 0    | **average kernels**              | average kernels instead of smoothing lengths                                                            | depends on variable SML                                                   |                                                     |
| 0    | **damage acts on S**             | damage acts on stress tensor                                                                            | depends on Fragmentation                                                  |                                                     |
| 0    | **SPH version**                  | (choosing between) different versions of SPH equations                                                  |                                                                           |                                                     |
|      |                                  |                                                                                                         |                                                                           |                                                     |
| 1    | **Navier Stokes**                | solving navier stokes equation for  viscous flows                                                       | set of PDEs describing the motion of viscous fluids                       |                                                     |
| 1    | **Plasticity**                   | Plasticity models: Drucker-Prager, Mohr-Coloumb, Collins, Collins including melt energy, Collins-simple | depends on Solid                                                          |                                                     |
| 1    | **Porosity**                     | Porosity models: stress p-alpha, epsilon p-alpha                                                        | depends on Solid                                                          |                                                     |
| 1    | **Kley viscosity**               |                                                                                                         | depends on Navier Stokes and XSPH                                         |                                                     |
|      | **Constant kinematic viscosity** |                                                                                                         | depends on Navier Stokes                                                  |                                                     |
| 1    | **Balsara**                      | Balsara switch for reducing artificial viscosity where not needed                                       | depends on artificial viscosity                                           |                                                     |
| 1    | **Shakura-Sunayev-alpha**        |                                                                                                         | depends on Navier Stokes                                                  |                                                     |
| 1    | **SML correction**               | correction factors for variable smoothing length                                                        |                                                                           | Evitas master thesis                                |
| 1    | **XSPH**                         | additional smoothing of velocity field                                                                  |                                                                           |                                                     |
|      |                                  |                                                                                                         |                                                                           |                                                     |
| 2    | **gravitating point masses**     | additional point masses interacting gravitationally with themselves and the SPH particles               | no need for parallelization, since at most 4 additional point masses      |                                                     |
| 2    | **particle accretion**           | removing (and adding) particles (through accretion)                                                     | depends on gravitating point masses                                       |                                                     |
| 2    | **updating sink values**         |                                                                                                         | depends on particle accretion                                             |                                                     |
| 2    | **inviscid SPH**                 | modern version of the balsara switch with shock capture                                                 | time dependent artificial viscosity coefficients                          |                                                     |
|      |                                  |                                                                                                         |                                                                           |                                                     |
| 3    | **Boundary particles**           | boundary approach                                                                                       |                                                                           |                                                     |
| 3    | **Ghost boundaries**             | boundary approach                                                                                       |                                                                           |                                                     |
|      |                                  |                                                                                                         |                                                                           |                                                     |
| 4    | **Plasticity**                   | Plasticity models: viscous regolith, pure regolith, JC-plasticity                                       |                                                                           |                                                     |
| 4    | **Porosity**                     | Porosity models: Sirono porosity                                                                        |                                                                           |                                                     |
|      |                                  |                                                                                                         |                                                                           |                                                     |
| 5    | **real Hydro**                   |                                                                                                         |                                                                           |                                                     |


### Equation of states

| EOS                    | Description                                                                                                     | Coverage                                          | Notes                                                                                      |
|----------------------------|-----------------------------------------------------------------------------------------------------------------|---------------------------------------------------|--------------------------------------------------------------------------------------------|
| **Liquid EOS**             | pressure depends linearly on the change of density                                                              | small compressions and expansions of liquid       |                                                                                            |
| **Murnaghan EOS**          | extension of the liquid EOS, where the pressure depends nonlinearly on the density                              | limited to isothermal compression                 |                                                                                            |
| **Tillotson EOS**          | originally derived for high-velocity impact simulations, distinguishing two domains and interpolating inbetween | phase transitions are not handled (appropriately) | computationally simple while sophisticated enough for a wide regime of physical conditions |
| **ANEOS**                  | tabulated EOS generated by an ancient Fortran code                                                              | depending on the generation                       |                                                                                            |
| **Locally-isothermal gas** |                                                                                                                 | locally isothermal gas                            |                                                                                            |
| **Perfect gas equation**   |                                                                                                                 | ideal gas                                         |                                                                                            |
| **Polytropic gas**         |                                                                                                                 | ideal gas                                         |                                                                                            |

