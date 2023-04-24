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

/// SPH simulation (this must be enabled when MFV or MFM method is used)
// TODO: rename to HYDRO_SIM to avoid misunderstandings
#define SPH_SIM 1

/** SPH RELATED SWITCHES */
/// integrate energy equation
#define INTEGRATE_ENERGY 1

/// integrate density equation
#define INTEGRATE_DENSITY 0

/// integrate smoothing length
#define INTEGRATE_SML 1

/// decouple smoothing length for pc integrator(s)
#define DECOUPLE_SML 0

/** variable smoothing length
 * SPH: include approximately the interactions as given in material.cfg
 * MFV/MFM: interactions in material config define the effective neighbor number
*/
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

/** Use meshless particle methods Meshless Finite Volume (MFV) or Meshless finite Mass (MFM)
 *  **MESHLESS_FINITE_METHOD 0:** Use SPH for hydrodynamics
 *  **MESHLESS_FINITE_METHOD 1:** Use MFV for hydrodynamics
 *  **MESHLESS_FINITE_METHOD 2:** Use MFM for hydrodynamics
 */
#define MESHLESS_FINITE_METHOD 0

/// do not move particles which is only valid for MFV
#define MFV_FIX_PARTICLES 1

/// employ an additional pairwise limiter between particles
#define PAIRWISE_LIMITER 1

/** Courant (CFL) number (note that our sml is defined up to the zero of the kernel, not half of it)
 *  Note that this number is used for different criteria in SPH and MFV/MFM
 */
#define COURANT_FACT 0.4

// deprecated flag
#define ARTIFICIAL_VISCOSITY 1
#define BALSARA_SWITCH 0

// applying a pressure floor in MFV/MFM after forward prediction in time
//#define APPLY_PRESSURE_FLOOR 1

// switch for slope limiting (should not be disabled)
//#define SLOPE_LIMITING 1

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

/// parameters for variable kernel size for MFV/MFM
#define ROOT_FOUND_TOL_SML 1e-2
#define MAX_NUM_SML_ITERATIONS 100

#define FORCES_FACT 0.2

constexpr real dbl_max = std::numeric_limits<real>::max();
#define DBL_MAX dbl_max;
constexpr real dbl_min = std::numeric_limits<real>::min();
#define DBL_MIN dbl_min;

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
    int defaultRiemannSolver;
    bool removeParticles;
    int removeParticlesCriterion;
    real removeParticlesDimension;
    int bins;
    bool calculateAngularMomentum;
    bool calculateEnergy;
    bool calculateCenterOfMass;
    real particleMemoryContingent;
    int domainListSize;

    /// parameters for standard slope limiter
    real critCondNum;
    real betaMin;
    real betaMax;
#if PAIRWISE_LIMITER
    real psi1;
    real psi2;
#endif

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
        explicit_euler, predictor_corrector_euler, leapfrog, godunov
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

//#define SOLID
// ideal hydro, navier stokes
//#define HYDRO
//#define PLASTICITY // no changes for particles (class) itself, but for material config

/*
// Basic physical model, choose one of the following:
// SOLID solves continuum mechanics with material strength, and stress tensor \sigma^{\alpha \beta} = -p \delta^{\alpha \beta} + S^{\alpha \beta}
// HYDRO solves only the Euler equation, and there is only (scalar) pressure
#define SOLID 1
#define HYDRO 0
// set additionally p to 0 if p < 0
#define REAL_HYDRO 0

// add additional point masses to the simulation, read from file <filename>.mass
// format is location velocities mass r_min r_max, where location and velocities are vectors with size DIM and
// r_min/r_max are min/max distances of sph particles to the bodies before they are taken out of the simulation
#define GRAVITATING_POINT_MASSES 0

// sink particles (set point masses to be sink particles)
#define PARTICLE_ACCRETION 0 // check if particle is bound to one of the sink particles (crossed the accretion radius, rmin); if also UPDATE_SINK_VALUES 1: particle is accreted and ignored afterwards, else: continues orbiting without being accreted
#define UPDATE_SINK_VALUES 0 // add to sink the quantities of the accreted particle: mass, velocity and COM

// integrate the energy equation
// when setting up a SOLID simulation with Tillotson or ANEOS, it must be set to 1
#define INTEGRATE_ENERGY 0

// integrate the continuity equation
// if set to 0, the density will be calculated using the standard SPH sum \sum_i m_j W_ij
#define INTEGRATE_DENSITY 1

// adds viscosity to the Euler equation
#define NAVIER_STOKES 0
// choose between two different viscosity models
#define SHAKURA_SUNYAEV_ALPHA 0
#define CONSTANT_KINEMATIC_VISCOSITY 0
// artificial bulk viscosity according to Schaefer et al. (2004)
#define KLEY_VISCOSITY 0

// This is the damage model following Benz & Asphaug (1995). Set FRAGMENTATION to activate it.
// The damage acts always on pressure, but only on deviator stresses if DAMAGE_ACTS_ON_S is
// activated too, which is an important switch depending on the plasticity model (see comments there).
// Note: The damage model needs distribution of activation thresholds in the input file.
#define FRAGMENTATION 0
#define DAMAGE_ACTS_ON_S 0

// Choose the SPH representation to solve the momentum and energy equation:
// SPH_EQU_VERSION 1: original version with HYDRO dv_a/dt ~ - (p_a/rho_a**2 + p_b/rho_b**2)  \nabla_a W_ab
//                                     SOLID dv_a/dt ~ (sigma_a/rho_a**2 + sigma_b/rho_b**2) \nabla_a W_ab
// SPH_EQU_VERSION 2: slighty different version with
//                                     HYDRO dv_a/dt ~ - (p_a+p_b)/(rho_a*rho_b)  \nabla_a W_ab
//                                     SOLID dv_a/dt ~ (sigma_a+sigma_b)/(rho_a*rho_b)  \nabla_a W_ab
// If you do not know what to do, choose SPH_EQU_VERSION 1.
#define SPH_EQU_VERSION 1

// for the tensile instability fix
// you do not need this
#define ARTIFICIAL_STRESS 0

// standard SPH alpha/beta viscosity
#define ARTIFICIAL_VISCOSITY 1
// Balsara switch: lowers the artificial viscosity in regions without shocks
#define BALSARA_SWITCH 0

// INVISCID SPH (see Cullen & Dehnen paper)
#define INVISCID_SPH 0

// consistency switches
// for zeroth order consistency
#define SHEPARD_CORRECTION 0
// for linear consistency
// add tensorial correction tensor to dSdt calculation -> better conservation of angular momentum
#define TENSORIAL_CORRECTION 1

// Available plastic flow conditions:
// (if you do not know what this is, choose (1) or nothing)
//   (1) Simple von Mises plasticity with a constant yield strength:
#define VON_MISES_PLASTICITY 0
//   (2) Drucker-Prager (DP) yield criterion -> yield strength is given by the condition \sqrt(J_2) + A * I_1 + B = 0
//       with I_1: first invariant of stress tensor
//          J_2: second invariant of stress tensor
//          A, B: DP constants, which are calculated from angle of internal friction and cohesion
#define DRUCKER_PRAGER_PLASTICITY 0
//   (3) Mohr-Coulomb (MC) yield criterion
//       -> yield strength is given by yield_stress = tan(friction_angle) \times pressure + cohesion
#define MOHR_COULOMB_PLASTICITY 0
//       Note: DP and MC are intended for granular-like materials, therefore the yield strength simply decreases (linearly) to zero for p<0.
//       Note: For DP and MC you can additionally choose (1) to impose an upper limit for the yield stress.
//   (4) Pressure dependent yield strength following Collins et al. (2004) and the implementation in Jutzi (2015)
//       -> yield strength is different for damaged (Y_d) and intact material (Y_i), and averaged mean (Y) in between:
//              Y_i = cohesion + \mu P / (1 + \mu P / (yield_stress - cohesion) )
//          where *cohesion* is the yield strength for P=0 and *yield_stress* the asymptotic limit for P=\infty
//          \mu is the coefficient of internal friction (= tan(friction_angle))
//              Y_d = cohesion_damaged + \mu_d \times P
//          where \mu_d is the coefficient of friction of the *damaged* material
//              Y = (1-damage)*Y_i + damage*Y_d
//              Y is limited to <= Y_i
//       Note: If FRAGMENTATION is not activated only Y_i is used.
//             DAMAGE_ACTS_ON_S is not allowed for this model, since the limiting of S already depends on damage.
//       If you want to additionally model the influence of some (single) melt energy on the yield strength, then activate
//       COLLINS_PLASTICITY_INCLUDE_MELT_ENERGY, which adds a factor (1-e/e_melt) to the yield strength.
#define COLLINS_PLASTICITY 0
#define COLLINS_PLASTICITY_INCLUDE_MELT_ENERGY 0
//   (5) Simplified version of the Collins et al. (2004) model, which uses only the
//       strength representation for intact material (Y_i), irrespective of damage.
//       Unlike in (4), Y decreases to zero (following a linear yield strength curve) for p<0.
//       In addition, negative pressures are limited to the pressure corresponding to
//       yield strength = 0 (i.e., are set to this value when they get more negative).
#define COLLINS_PLASTICITY_SIMPLE 0
// Note: The deviator stress tensor is additionally reduced by FRAGMENTATION (i.e., damage) only if
//       DAMAGE_ACTS_ON_S is set. For most plasticity models it depends on the use case whether this
//       is desired, only for COLLINS_PLASTICITY it is not reasonable (and therefore not allowed).

// model regolith as viscous fluid -> experimental setup, only for powerusers
#define VISCOUS_REGOLITH 0
// use Bui model for regolith -> experimental setup, only for powerusers
#define PURE_REGOLITH 0
// use Johnson-Cook plasticity model -> experimental setup, only for powerusers
#define JC_PLASTICITY 0

// Porosity models:
// p-alpha model implemented following Jutzi (200x)
#define PALPHA_POROSITY 0         // pressure depends on distention
#define STRESS_PALPHA_POROSITY 0  // deviatoric stress is also affected by distention
// Sirono model modified by Geretshauser (2009/10)
#define SIRONO_POROSITY 0
// eps-alpha model implemented following Wuennemann
#define EPSALPHA_POROSITY 0

// max number of activation thresholds per particle, only required for FRAGMENTATION, otherwise set to 1
#define MAX_NUM_FLAWS 1
// maximum number of interactions per particle -> fixed array size
#define MAX_NUM_INTERACTIONS 128

// if set to 1, the smoothing length is not fixed for each material type
// choose either FIXED_NOI for a fixed number of interaction partners following
// the ansatz by Hernquist and Katz
// or choose INTEGRATE_SML if you want to additionally integrate an ODE for
// the sml following the ansatz by Benz and integrate the ODE for the smoothing length
// d sml / dt  = sml/DIM * 1/rho  \nabla velocity
// if you want to specify an individual initial smoothing length for each particle (instead of the material
// specific one in material.cfg) in the initial particle file, set READ_INITIAL_SML_FROM_PARTICLE_FILE to 1
#define VARIABLE_SML 0
#define FIXED_NOI 0
#define INTEGRATE_SML 0
#define READ_INITIAL_SML_FROM_PARTICLE_FILE 0

// correction terms for sml calculation: adds gradient of the smoothing length to continuity equation, equation of motion, internal energy equation
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

// additional smoothing of the velocity field
// hinders particle penetration
// see Morris and Monaghan 1984
#define XSPH 0

// boundaries EXPERIMENTAL, please do not use this....
#define BOUNDARY_PARTICLE_ID -1
#define GHOST_BOUNDARIES 0
// note: see additionally boundaries.cu with functions beforeRHS and afterRHS for boundary conditions

// IO options
#define HDF5IO 1    // use HDF5 (needs libhdf5-dev and libhdf5)
#define MORE_OUTPUT 0   //produce additional output to HDF5 files (p_max, p_min, rho_max, rho_min); only ueful when HDF5IO is set
#define MORE_ANEOS_OUTPUT 0 // produce additional output to HDF5 files (T, cs, entropy, phase-flag); only useful when HDF5IO is set; set only if you use the ANEOS eos, but currently not supported for porosity+ANEOS
#define OUTPUT_GRAV_ENERGY 0    // compute and output gravitational energy (at times when output files are written); of all SPH particles (and also w.r.t. gravitating point masses and between them); direct particle-particle summation, not tree; option exists to control costly computation for high particle numbers
#define BINARY_INFO 0   // generates additional output file (binary_system.log) with info regarding binary system: semi-major axis, eccentricity if GRAVITATING_POINT_MASSES == 1
*/

#endif //MILUPHPC_PARAMETER_H
