/**
 * @file parameter.h
 * @brief Settings via preprocessor directives, typedefs, constants, structs.
 *
 * More detailed description.
 * This file contains ...
 *
 * @author Michael Staneker
 * @bug no known bugs
 * @todo remove deprecated flags and avoid flags that don't match
 */
#ifndef MILUPHPC_PARAMETER_H
#define MILUPHPC_PARAMETER_H

#include <limits>
#include <iostream>

/**
 * @brief Precision of simulation.
 *
 * Simulations can be either
 *
 * * single precision via `SINGLE_PRECISION 1`
 * * or double precision via `SINGLE_PRECISION 0`
 *
 * @note
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

/// maximum tree level
#define MAX_LEVEL 21

/// enable/disable debugging calculations/outputs
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
/// function: \f$ 2^{x} \f$
#define power_two(x) (1 << (x))
/// \f$ 2^{DIM} \f$ which corresponds to the number of children for each node
#define POW_DIM power_two(DIM)

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
 *     * HYDRO \f$ \frac{dv_a}{dt} \sim - \left( \frac{p_a}{\rho_a^2} + \frac{p_b}{\rho_b^2} \right)  \nabla_a W_{ab} \f$
 *     * SOLID \f$ \frac{dv_a}{dt} \sim \left( \frac{\sigma_a}{\rho_a^2} + \frac{\sigma_b}{\rho_b2} \right) \nabla_a W_{ab} \f$
 * * **SPH_EQU_VERSION 2:** slighty different version with
 *     * HYDRO \f$ \frac{dv_a}{dt} \sim - \frac{p_a+p_b}{\rho_a \cdot \rho_b}  \nabla_a W_{ab} \f$
 *     * SOLID \f$ \frac{dv_a}{dt} \sim \frac{\sigma_a+\sigma_b}{\rho_a \cdot \rho_b}  \nabla_a W_{ab} \f$
 */
#define SPH_EQU_VERSION 1

/// @todo deprecated flag (`ARTIFICIAL_VISCOSITY` is default)
#define ARTIFICIAL_VISCOSITY 1

/// @todo not yet fully implemented (flags)
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
//#define DOMAIN_LIST_SIZE 512 // changed to be a runtime constant
#define MAX_DEPTH 128
#define MAX_NUM_INTERACTIONS 180
#define NUM_THREADS_LIMIT_TIME_STEP 256
#define NUM_THREADS_CALC_CENTER_OF_MASS 256

// (note that our sml is defined up to the zero of the kernel, not half of it)
/// Courant (CFL) number
#define COURANT_FACT 0.4

#define FORCES_FACT 0.2

constexpr real dbl_max = std::numeric_limits<real>::max();
#define DBL_MAX dbl_max;

/// (Physical) constants
namespace Constants {
    /// Gravitational constant
    constexpr real G = 6.67430e-11;
}

/**
 * Simulation parameters.
 *
 * Some settings/parameters for dispatching simulation.
 */
typedef struct SimulationParameters {
    /// output file(s) directory
    std::string directory;
    /// log file(s) directory
    std::string logDirectory;
    /// verbosity level
    int verbosity;
    /// time the CUDA kernels
    bool timeKernels;
    /// number of output files
    int numOutputFiles;
    /// time step
    real timeStep;
    /// max (allowed) time step
    real maxTimeStep;
    /// end time of simulation
    real timeEnd;
    /// apply load balancing
    bool loadBalancing;
    /// apply load balancing each x interval/simulation step
    int loadBalancingInterval;
    /// @todo deprecated
    int loadBalancingBins;
    /// input file containing initial particle distribution
    std::string inputFile;
    /// input file containing material configurations/parameters
    std::string materialConfigFile;
    /// specify a (MPI) rank for console logging (default: -1 logging all ranks)
    int outputRank;
    /// log performance to HDF5 file
    bool performanceLog;
    /// log particles sent to HDF5 file
    bool particlesSent2H5;
    /// space-filling curve selection
    int sfcSelection;
    /// integrator selection
    int integratorSelection;
//#if GRAVITY_SIM
    /// clumping parameter/ \f$ \theta $\f - parameter
    real theta;
    /// gravitational smoothing
    real smoothing;
    /// gravitational force version to be used
    int gravityForceVersion;
//#endif
//#if SPH_SIM
    /// (SPH) smoothing kernel selection
    int smoothingKernelSelection;
    /// SPH fixed-radius near-neighbor search (FRNN) version to be used
    int sphFixedRadiusNNVersion;
//#endif
    /// remove particles in dependence of some criterion
    bool removeParticles;
    /// criterion to remove particles (spherical/cubic/... domain)
    int removeParticlesCriterion;
    /// dimension to be removed (e.g. distance)
    real removeParticlesDimension;
    int bins;
    /// calculate the angular momentum
    bool calculateAngularMomentum;
    /// calculate the energy
    bool calculateEnergy;
    /// calculate the center of mass (COM)
    bool calculateCenterOfMass;
    /// memory contingent: percentage of all particles that are possibly able to be on one process
    real particleMemoryContingent;
    /// domain list size, possible number of domain list nodes
    int domainListSize;
} SimulationParameters;

/**
 * @brief Specify target: device or host.
 *
 * Struct usable to specify e.g. copy commands.
 */
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

/**
 * @brief (SPH) available smoothing kernels.
 *
 * available kernels:
 *
 * * spiky
 * * cubic spline
 * * wendlandc2, wendlandc4, wendlandc6
 */
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

/**
 * @brief Execution location (host/device).
 *
 * Execute e.g. a reduction operation either on the `host` or `device`.
 */
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

/**
 * @brief Available space-filling curves.
 *
 * Available space-filling curves are:
 *
 * * Lebesgue SFC
 * * Hilbert SFC
 *
 * whereas Hilbert SFC are derived from Lebesgue SFC.
 */
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

/**
 * @brief Available integrators.
 *
 * Available integrators are
 *
 * * explicit euler
 * * Predictor-corrector Euler/Heun
 *
 * @note The gravitational part is currently separated and not included
 * in the predictor-corrector integration scheme for several reasons.
 */
struct IntegratorSelection
{
    enum Type
    {
        explicit_euler, predictor_corrector_euler
    };
    Type t_;
    IntegratorSelection(Type t) : t_(t) {}
    operator Type () const {return t_;}
private:
    template<typename T>
    operator T () const;
};


/**
 * @brief Implemented equation of states
 *
 * Implemented equation of states are:
 *
 * * polytropic gas
 * * isothermal gas
 * * ideal gas
 * * locally isothermal gas
 */
enum EquationOfStates {
    //EOS_TYPE_ACCRETED = -2, // special flag for particles that got accreted by a gravitating point mass
    //EOS_TYPE_IGNORE = -1, // particle is ignored
    /** polytropic EOS for gas, needs polytropic_K and polytropic_gamma in material.cfg file */
    EOS_TYPE_POLYTROPIC_GAS = 0,
    //EOS_TYPE_MURNAGHAN = 1, // Murnaghan EOS for solid bodies, see Melosh "Impact Cratering", needs in material.cfg: rho_0, bulk_modulus, n
    //EOS_TYPE_TILLOTSON = 2, // Tillotson EOS for solid bodies, see Melosh "Impact Cratering", needs in material.cfg: till_rho_0, till_A, till_B, till_E_0, till_E_iv, till_E_cv, till_a, till_b, till_alpha, till_beta; bulk_modulus and shear_modulus are needed to calculate the sound speed and crack growth speed for FRAGMENTATION
    /** this is pure molecular hydrogen at 10 K */
    EOS_TYPE_ISOTHERMAL_GAS = 3,
    //EOS_TYPE_REGOLITH = 4, // The Bui et al. 2008 soil model
    //EOS_TYPE_JUTZI = 5, // Tillotson EOS with p-alpha model by Jutzi et al.
    //EOS_TYPE_JUTZI_MURNAGHAN = 6, // Murnaghan EOS with p-alpha model by Jutzi et al.
    //EOS_TYPE_ANEOS = 7, // ANEOS (or tabulated EOS in ANEOS format)
    //EOS_TYPE_VISCOUS_REGOLITH = 8, // describe regolith as a viscous material -> EXPERIMENTAL DO NOT USE
    /** ideal gas equation, set polytropic_gamma in material.cfg */
    EOS_TYPE_IDEAL_GAS = 9,
    //EOS_TYPE_SIRONO = 10, // Sirono EOS modifed by Geretshauser in 2009/10
    //EOS_TYPE_EPSILON = 11, // Tillotson EOS with epsilon-alpha model by Wuennemann, Collins et al.
    /** locally isothermal gas: \f$ p = c_s^2 \cdot \rho \f$ */
    EOS_TYPE_LOCALLY_ISOTHERMAL_GAS = 12,
    //EOS_TYPE_JUTZI_ANEOS = 13// ANEOS EOS with p-alpha model by Jutzi et al.
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
