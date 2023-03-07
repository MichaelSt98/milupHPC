/**
 * @file particles.cuh
 * @brief Particle struct as Structure of arrays both instantiable on host and device.
 *
 * Particle class to embrace the information and properties of the (SPH) particles including
 *
 * * position
 * * velocity
 * * acceleration
 * * density
 * * pressure
 * * ...
 *
 * @author Michael Staneker
 * @bug no known bugs
 * @todo remove deprecated flags and avoid flags that don't match
 */
#ifndef MILUPHPC_PARTICLES_CUH
#define MILUPHPC_PARTICLES_CUH

#include "parameter.h"
#include "cuda_utils/cuda_utilities.cuh"
#include <cmath>
#include <assert.h>


/**
 * @brief Particle(s) class based on SoA (Structur of Arrays).
 *
 * Since dynamic memory allocation and access to heap objects in GPUs are usually suboptimal,
 * an array-based data structure is used to store the (pseudo-)particle information as well as the tree
 * and allows for efficient cache alignment. Consequently, array indices are used instead of pointers
 * to constitute the tree out of the tree nodes, whereas "-1" represents a null pointer and "-2" is
 * used for locking nodes. For this purpose an array with the minimum length \f$ 2^{d} \cdot (M - N) \f$
 * with dimensionality \f$ d \f$ and number of cells \f$ (M -N) \f$ is needed to store the children.
 * The \f$ i \f$-th child of node \f$ c \f$ can therefore be accessed via index \f$ 2^d \cdot c + i \f$.
 *
 * Single-GPU memory layout:
 *
 * \image html images/Parallelization/single_gpu_memory_layout.png width=40%
 *
 * It is further necessary to exchange particle properties between processes and (temporarily) include them
 * into the local tree. To combine this requirement with the data structure described it is necessary to
 * reserve parts of the array or memory for this purpose.
 *
 * Multi-GPU memory layout:
 *
 * \image html images/Parallelization/multi_gpu_memory_layout.png width=40%
 */
class Particles {

public:

    /// number of particles
    integer *numParticles;
    /// number of nodes
    integer *numNodes;

    /// (pointer to) mass (array)
    real *mass;
    /// (pointer to) x position (array)
    real *x;
    /// (pointer to) x velocity (array)
    real *vx;
    /// (pointer to) x acceleration (array)
    real *ax;
    real *ax_old;
#if DIM > 1
    /// (pointer to) y position (array)
    real *y;
    /// (pointer to) y velocity (array)
    real *vy;
    /// (pointer to) y acceleration (array)
    real *ay;
    real *ay_old;
#if DIM == 3
    /// (pointer to) z position (array)
    real *z;
    /// (pointer to) z velocity (array)
    real *vz;
    /// (pointer to) z acceleration (array)
    real *az;
    real *az_old;
#endif
#endif

    /// (pointer to) x gravitational acceleration (array)
    real *g_ax;
    real *g_ax_old;
#if DIM > 1
    /// (pointer to) y gravitational acceleration (array)
    real *g_ay;
    real *g_ay_old;
#if DIM == 3
    /// (pointer to) z gravitational acceleration (array)
    real *g_az;
    real *g_az_old;
#endif
#endif


    /// (pointer to) node type
    integer *nodeType;

    /// (pointer to) level of the (pseudo-)particles
    integer *level;

    /// (pointer to) unique identifier (array)
    idInteger *uid; // unique identifier (unsigned int/long?)
    /// (pointer to) material identifier (array)
    integer *materialId; // material identfier (e.g.: ice, basalt, ...)
    /// (pointer to) smoothing length (array)
    real *sml; // smoothing length

    /// (pointer to) near(est) neighbor list (array)
    integer *nnl; // max(number of interactions)
    /// (pointer to) number of interactions (array)
    integer *noi; // number of interactions (alternatively initialize nnl with -1, ...)

    /// (pointer to) internal energy (array)
    real *e; // internal energy
    /// (pointer to) time derivative of internal energy (array)
    real *dedt;

    /// energy (kinetic + gravitational for now)
    real *u;

    /// (pointer to) sound of speed (array)
    real *cs; // soundspeed

    // simplest hydro
    /// (pointer to) density (array)
    real *rho; // density
    /// (pointer to) pressure (array)
    real *p; // pressure

#if MESHLESS_FINITE_METHOD
    /// (pointer) to inverse effective volume (array)
    real *omega;

    /// pointer to vector weights
    real *psix;
#if DIM > 1
    real *psiy;
#if DIM == 3
    real *psiz;
#endif
#endif

    /// (pointer) to gradients
    real *rhoGrad; // gradient of rho (dimension N*DIM)
    real *vxGrad; // gradient of vx (dimension N*DIM)
#if DIM > 1
    real *vyGrad; // gradient of vy (dimension N*DIM)
#if DIM == 3
    real *vzGrad; // gradient of vz (dimension N*DIM)
#endif
#endif
    real *pGrad; // gradient of p (dimension N*DIM)

    /// (pointer) to fluxes
    real *massFlux; // total mass flux
    real *vxFlux; // total x-velocity flux
#if DIM > 1
    real *vyFlux; // total y-velocity flux
#if DIM == 3
    real *vzFlux; // total z-velocity flux
#endif
#endif
    real *energyFlux; // total energy flux
    real *Ncond; // condition number on invertibility of matrix E_i

#endif // MESHLESS_FINITE_METHOD

    /// (pointer) to max(mu_ij) (array) needed for artificial viscosity and determining timestp
    real *muijmax;

#if NAVIER_STOKES
    real *Tshear;
    real *eta;
#endif

//#if INTEGRATE_DENSITY
    // integrated density
    /// (pointer to) time derivative of density (array)
    real *drhodt;
//#endif

#if VARIABLE_SML || INTEGRATE_SML
    // integrate/variable smoothing length
    /// (pointer to) time derivative of smoothing length (array)
    real *dsmldt;
#endif

#if SML_CORRECTION
    real *sml_omega;
#endif

#if SOLID
    /// (pointer to) deviatoric stress tensor (array)
    real *S; // deviatoric stress tensor (DIM * DIM)
    /// (pointer to) time derivative of deviatoric stress tensor (array)
    real *dSdt;
    /// (pointer to) local strain (array)
    real *localStrain; // local strain
#endif

#if BALSARA_SWITCH
    real *divv;
    real *curlv;
#endif

#if SOLID || NAVIER_STOKES
    /// (pointer to) sigma/stress tensor (array)
    real *sigma; // stress tensor (DIM * DIM)
#endif

#if ARTIFICIAL_STRESS
    /// (pointer to) tensile instability, tensor for correction (array)
    real *R; // tensile instability, tensor for correction (DIM * DIM)
#endif

#if POROSITY
    /// pressure of the sph particle after the last successful timestep
    real *pold;
    /// current distension of the sph particle
    real *alpha_jutzi;
    /// distension of the sph particle after the last successful timestep
    real *alpha_jutzi_old;
    /// time derivative of the distension
    real *dalphadt;
    /// partial derivative of the distension with respect to the pressure
    real *dalphadp;
    /// difference in pressure from the last timestep to the current one
    real *dp;
    /// partial derivative of the distension with respect to the density
    real *dalphadrho;
    /// additional factor to reduce the deviatoric stress tensor according to Jutzi
    real *f;
    /// partial derivative of the pressure with respect to the density
    real *delpdelrho;
    /// partial derivative of the pressure with respect to the specific internal energy
    real *delpdele;
    /// sound speed after the last successful timestep
	real *cs_old;

	/// distention in the strain-\alpha model
    real *alpha_epspor;
    /// time derivative of the distension
    real *dalpha_epspordt;
    /// volume change (trace of strain rate tensor)
    real *epsilon_v;
    /// time derivative of volume change
    real *depsilon_vdt;
#endif

#if ZERO_CONSISTENCY
    /// correction (value) for zeroth order consistency
    real *shepardCorrection;
#endif

#if LINEAR_CONSISTENCY
    /// correction matrix for linear order consistency
    real *tensorialCorrectionMatrix;
#endif

#if FRAGMENTATION
    /// DIM-root of tensile damage
    real *d;
    /// tensile damage + porous damage
    real *damage_total; // tensile damage + porous damage (directly, not DIM-root)
    /// time derivative of DIM-root of (tensile) damage
    real *dddt; // the time derivative of DIM-root of (tensile) damage
    /// total number of flaws
    integer *numFlaws; // the total number of flaws
    /// maximum number of flaws allowed per particle
    integer *maxNumFlaws; // the maximum number of flaws allowed per particle
    /// current number of activated flaws
    integer *numActiveFlaws; // the current number of activated flaws
    /// values for the strain for each flaw
    real *flaws; // the values for the strain for each flaw (array of size maxNumFlaws)
#if PALPHA_POROSITY
    /// DIM-root of porous damage
    real *damage_porjutzi; // DIM-root of porous damage
    /// time derivative of DIM-root of porous damage
    real *ddamage_porjutzidt; // time derivative of DIM-root of porous damage
#endif
#endif

    /**
     * Default constructor
     */
    CUDA_CALLABLE_MEMBER Particles();

    //TODO: constructors and setter only for specific dimension (#if , #else if, #else)
#if DIM == 1

    CUDA_CALLABLE_MEMBER Particles(integer *numParticles, integer *numNodes, real *mass, real *x, real *vx, real *ax,
                                   integer *level, idInteger *uid, integer *materialId, real *sml,
                                   integer *nnl, integer *noi, real *e, real *dedt, real *cs, real *rho, real *p);

    CUDA_CALLABLE_MEMBER void set(integer *numParticles, integer *numNodes, real *mass, real *x, real *vx, real *ax,
                                  integer *level, idInteger *uid, integer *materialId, real *sml,
                                  integer *nnl, integer *noi, real *e, real *dedt, real *cs, real *rho, real *p);
#elif DIM == 2

    CUDA_CALLABLE_MEMBER Particles(integer *numParticles, integer *numNodes, real *mass, real *x, real *y, real *vx,
                                   real *vy, real *ax, real *ay, idInteger *uid,
                                   integer *materialId, real *sml, integer *nnl, integer *noi, real *e, real *dedt,
                                   real *cs, real *rho, real *p);


    CUDA_CALLABLE_MEMBER void set(integer *numParticles, integer *numNodes, real *mass, real *x, real *y, real *vx,
                                  real *vy, real *ax, real *ay, integer *level, idInteger *uid,
                                  integer *materialId, real *sml, integer *nnl, integer *noi, real *e, real *dedt,
                                  real *cs, real *rho, real *p);
#else

    /**
     * @brief Constructor (`DIM = 3`) assigning variables to pointer members.
     *
     * @param numParticles number of particles
     * @param numNodes number of nodes
     * @param mass mass entry
     * @param x position \f$ x \f$ entry
     * @param y position \f$ y \f$ entry
     * @param z position \f$ x \f$ entry
     * @param vx velocity \f$ v_x \f$ entry
     * @param vy velocity \f$ v_y \f$ entry
     * @param vz velocity \f$ v_z \f$ entry
     * @param ax acceleration \f$ a_x \f$ entry
     * @param ay acceleration \f$ a_y \f$ entry
     * @param az acceleration \f$ a_z \f$ entry
     * @param uid unique identifier
     * @param materialId material identifier
     * @param sml smoothing length
     * @param nnl near neighbor list
     * @param noi number of interaction partners
     * @param e energy
     * @param dedt time derivative of the energy \f$ \frac{de}{dt} \f$ entry
     * @param cs speed of sound
     * @param rho density \f$ \rho \f$ entry
     * @param p pressure \f$ p \f$ entry
     */
    CUDA_CALLABLE_MEMBER Particles(integer *numParticles, integer *numNodes, real *mass, real *x, real *y, real *z,
                                   real *vx, real *vy, real *vz, real *ax, real *ay, real *az,
                                   idInteger *uid, integer *materialId, real *sml,
                                   integer *nnl, integer *noi, real *e, real *dedt,
                                   real *cs, real *rho, real *p);


    /**
     * @brief Setter (`DIM = 3`) assigning variables to pointer members.
     *
     * Setter as addition to constructor in order to allow Particles class instance creation
     * and subsequent assignment of member variables.
     *
     * @param numParticles number of particles
     * @param numNodes number of nodes
     * @param mass mass entry
     * @param x position \f$ x \f$ entry
     * @param y position \f$ y \f$ entry
     * @param z position \f$ x \f$ entry
     * @param vx velocity \f$ v_x \f$ entry
     * @param vy velocity \f$ v_y \f$ entry
     * @param vz velocity \f$ v_z \f$ entry
     * @param ax acceleration \f$ a_x \f$ entry
     * @param ay acceleration \f$ a_y \f$ entry
     * @param az acceleration \f$ a_z \f$ entry
     * @param uid unique identifier
     * @param materialId material identifier
     * @param sml smoothing length
     * @param nnl near neighbor list
     * @param noi number of interaction partners
     * @param e energy
     * @param dedt time derivative of the energy \f$ \frac{de}{dt} \f$ entry
     * @param cs speed of sound
     * @param rho density \f$ \rho \f$ entry
     * @param p pressure \f$ p \f$ entry
     */
    CUDA_CALLABLE_MEMBER void set(integer *numParticles, integer *numNodes, real *mass, real *x, real *y, real *z,
                                  real *vx, real *vy, real *vz, real *ax, real *ay, real *az,
                                  integer *level, idInteger *uid, integer *materialId,
                                  real *sml, integer *nnl, integer *noi, real *e, real *dedt, real *cs,
                                  real *rho, real *p);
#endif

#if DIM == 1
    CUDA_CALLABLE_MEMBER void setGravity(real *g_ax);
#elif DIM == 2
    CUDA_CALLABLE_MEMBER void setGravity(real *g_ax, real *g_ay);
#else
    /**
     * @brief Constructor (`DIM = 3`) assigning gravitational acceleration to member variables.
     *
     * @note acceleration and gravitational acceleration separated due to decoupled gravity.
     *
     * @param g_ax gravitational acceleration \f$ a_{x, grav} \f$ entry
     * @param g_ay gravitational acceleration \f$ a_{y, grav} \f$ entry
     * @param g_az gravitational acceleration \f$ a_{z, grav} \f$ entry
     */
    CUDA_CALLABLE_MEMBER void setGravity(real *g_ax, real *g_ay, real *g_az);
#endif

#if DIM == 1
    CUDA_CALLABLE_MEMBER void setLeapfrog(real *ax_old, real *g_ax_old);
#elif DIM == 2
    CUDA_CALLABLE_MEMBER void setLeapfrog(real *ax_old, real *ay_old, real *g_ax_old, real *g_ay_old);
#else

    CUDA_CALLABLE_MEMBER void setLeapfrog(real *ax_old, real *ay_old, real *az_old, real *g_ax_old, real *g_ay_old,
                                          real *g_az_old);
#endif

#if MESHLESS_FINITE_METHOD

    //TODO: Implement 1D and 2D (although 1D is not applicable as gradient estimation would fail)

#if DIM == 3
    /**
     * @brief Setter for MFV/MFM related quantities
     *
     * @param omega inverse of effective volume of particles
     * @param psix x-component of least-squares-fit vector weight
     * @param psiy y-component of least-squares-fit vector weight
     * @param psiz z-component of least-squares-fit vector weight
     * @param massFlux summed mass flux through all effective faces of neighboring particles
     * @param vxFlux summed x-velocity flux through all effective faces of neighboring particles
     * @param vyFlux summed y-velocity flux through all effective faces of neighboring particles
     * @param vzFlux summed z-velocity flux through all effective faces of neighboring particles
     * @param energyFlux summed energy flux through all effective faces of neighboring particles
     * @param Ncond condition number on invertibility of matrix E_i
     * @param rhoGrad gradient of rho
     * @param vxGrad gradient of x-velocity
     * @param vyGrad gradient of x-velocity
     * @param vzGrad gradient of x-velocity
     * @param pGrad gradient of pressure
     */
    CUDA_CALLABLE_MEMBER void setMeshlessFinite(real *omega, real *psix, real *psiy, real *psiz,
                                                real *massFlux, real *vxFlux, real *vyFlux, real *vzFlux,
                                                real *energyFlux, real *Ncond, real *rhoGrad, real *vxGrad,
                                                real *vyGrad, real *vzGrad, real *pGrad);

#endif
#endif // MESHLESS_FINITE_METHOD

    /**
     * @brief Setter for energy.
     *
     * @param u energy
     */
    CUDA_CALLABLE_MEMBER void setU(real *u);

    /**
     * @brief Setter for node type.
     *
     * node types are
     *
     * * particle
     * * pseudo-particle
     * * (lowest) domain list node
     *
     * @param nodeType node type
     */
    CUDA_CALLABLE_MEMBER void setNodeType(integer *nodeType);

    /**
     * @brief Setter for artificial viscosity (entry).
     *
     * @param muijmax
     */
    CUDA_CALLABLE_MEMBER void setArtificialViscosity(real *muijmax);

#if BALSARA_SWITCH
    CUDA_CALLABLE_MEMBER void setDivCurl(real *divv, real *curlv);
#endif

//#if INTEGRATE_DENSITY
    /**
     * @brief Setter in dependence of `INTEGRATE_DENSITY`.
     *
     * @param drhodt time derivative of density
     */
    CUDA_CALLABLE_MEMBER void setIntegrateDensity(real *drhodt);
//#endif
#if VARIABLE_SML || INTEGRATE_SML
    /**
     * @brief Setter, in dependence of `VARIABLE_SML`.
     *
     * @param dsmldt time derivative of smoothing length
     */
    CUDA_CALLABLE_MEMBER void setVariableSML(real *dsmldt);
#endif
#if SML_CORRECTION
    CUDA_CALLABLE_MEMBER void setSMLCorrection(real *sml_omega);
#endif
#if NAVIER_STOKES
    CUDA_CALLABLE_MEMBER void setNavierStokes(real *Tshear, real *eta);
#endif
#if SOLID
    /**
     * Setter, in dependence of `SOLID`
     *
     * @param S
     * @param dSdt
     * @param localStrain
     */
    CUDA_CALLABLE_MEMBER void setSolid(real *S, real *dSdt, real *localStrain);
#endif
#if SOLID || NAVIER_STOKES
    /**
     * Setter, in dependence of `SOLID` or `NAVIER_STOKES`
     *
     * @param sigma
     */
    CUDA_CALLABLE_MEMBER void setSolidNavierStokes(real *sigma);
#endif
#if ARTIFICIAL_STRESS
    /**
     * Setter, in dependence of `ARTIFICIAL_STRESS
     * `
     * @param R
     */
    CUDA_CALLABLE_MEMBER void setArtificialStress(real *R);
#endif
#if POROSITY
    /**
     * Setter, in dependence of `POROSITY`
     *
     * @param pold
     * @param alpha_jutzi
     * @param alpha_jutzi_old
     * @param dalphadt
     * @param dalphadp
     * @param dp
     * @param dalphadrho
     * @param f
     * @param delpdelrho
     * @param delpdele
     * @param cs_old
     * @param alpha_epspor
     * @param dalpha_epspordt
     * @param epsilon_v
     * @param depsilon_vdt
     */
    CUDA_CALLABLE_MEMBER void setPorosity(real *pold, real *alpha_jutzi, real *alpha_jutzi_old, real *dalphadt,
                                          real *dalphadp, real *dp, real *dalphadrho, real *f, real *delpdelrho,
                                          real *delpdele, real *cs_old, real *alpha_epspor, real *dalpha_epspordt,
                                          real *epsilon_v, real *depsilon_vdt);
#endif
#if ZERO_CONSISTENCY
    /**
     * Setter, in dependence of `ZERO_CONSISTENCY`
     *
     * @param shepardCorrection
     */
    CUDA_CALLABLE_MEMBER void setZeroConsistency(real *shepardCorrection);
#endif
#if LINEAR_CONSISTENCY
    /**
     * Setter, in dependence of `tensorialCorrectionMatrix`
     *
     * @param tensorialCorrectionMatrix
     */
    CUDA_CALLABLE_MEMBER void setLinearConsistency(real *tensorialCorrectionMatrix);
#endif
#if FRAGMENTATION
    /**
     * Setter, in dependence of `FRAGMENTATION`
     *
     * @param d
     * @param damage_total
     * @param dddt
     * @param numFlaws
     * @param maxNumFlaws
     * @param numActiveFlaws
     * @param flaws
     */
    CUDA_CALLABLE_MEMBER void setFragmentation(real *d, real *damage_total, real *dddt, integer *numFlaws,
                                               integer *maxNumFlaws, integer *numActiveFlaws, real *flaws);
#if PALPHA_POROSITY
    /**
     * Setter, in dependence of `PALPHA_POROSITY`
     *
     * @param damage_porjutzi
     * @param ddamage_porjutzidt
     */
    CUDA_CALLABLE_MEMBER void setPalphaPorosity(real *damage_porjutzi, real *ddamage_porjutzidt);
#endif
#endif

    /**
     * @brief Reset (specific) entries.
     *
     * Reset entries to default values.
     *
     * @param index index of entry to be reset
     */
    CUDA_CALLABLE_MEMBER void reset(integer index);

    /**
     * @brief Distance of two particles.
     *
     * Calculates the euclidian distance \f$ r = || \vec{x} - \vec{y} ||_2 = \sqrt{\sum_{i=1}^{DIM}(\vec{x}_i - \vec{y}_i)^2}\f$
     * of two particles at positions \f$ \vec{x} \f$ and \f$ \vec{y} \f$.
     *
     * @param index_1 index of particle 1
     * @param index_2 index of particle 2
     * @return distance of the two particles
     */
    CUDA_CALLABLE_MEMBER real distance(integer index_1, integer index_2);

    CUDA_CALLABLE_MEMBER real weightedEntry(integer index, Entry::Name entry);

    /**
     * Destructor
     */
    CUDA_CALLABLE_MEMBER ~Particles();

};

namespace ParticlesNS {

    namespace Kernel {

        /**
         * Debug function to search for *NANs* in the entries.
         *
         * @param particles Particle class instance
         * @param n number of particles (to be searched)
         */
        __global__ void check4nans(Particles *particles, integer n);

        /**
         * Debug/Info Kernel (for debugging purposes)
         *
         * @param particles
         * @param n
         * @param m
         * @param k
         */
        __global__ void info(Particles *particles, integer n, integer m, integer k);

        namespace Launch {

            /**
             * Wrapper function for ParticlesNS::check4nans().
             *
             * @return Wall time of execution.
             */
            real check4nans(Particles *particles, integer n);

            /**
             * Info Kernel Wrapper (for debugging purposes)
             *
             * @param particles
             * @param n
             * @param m
             * @param k
             * @return
             */
            real info(Particles *particles, integer n, integer m, integer k);
        }

#if DIM == 1

        __global__ void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x,
                            real *vx, real *ax, integer *level, idInteger *uid, integer *materialId,
                            real *sml, integer *nnl, integer *noi, real *e, real *dedt, real *cs, real *rho, real *p);

        namespace Launch {

            void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x, real *vx,
                     real *ax, integer *level, idInteger *uid, integer *materialId, real *sml, integer *nnl,
                     integer *noi, real *e, real *dedt, real *cs, real *rho, real *p);
        }

#elif DIM == 2

        __global__ void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x,
                            real *y, real *vx, real *vy, real *ax, real *ay, integer *level,
                            idInteger *uid, integer *materialId, real *sml, integer *nnl, integer *noi, real *e,
                            real *dedt, real *cs, real *rho, real *p);

        namespace Launch {

            void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x, real *y,
                     real *vx, real *vy, real *ax, real *ay, integer *level, idInteger *id,
                     integer *materialId, real *sml, integer *nnl, integer *noi, real *e, real *dedt, real *cs,
                     real *rho, real *p);
        }

#else

        __global__ void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x,
                            real *y, real *z, real *vx, real *vy, real *vz, real *ax, real *ay, real *az,
                            integer *level, idInteger *uid, integer *materialId, real *sml,
                            integer *nnl, integer *noi, real *e,
                            real *dedt, real *cs, real *rho, real *p);

        namespace Launch {

            void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x, real *y,
                     real *z, real *vx, real *vy, real *vz, real *ax, real *ay, real *az,
                     integer *level, idInteger *uid, integer *materialId, real *sml,
                     integer *nnl, integer *noi, real *e, real *dedt, real *cs, real *rho, real *p);
        }

#endif

#if DIM == 1
        __global__ void setGravity(Particles *particles, real *g_ax);

        namespace Launch {
            void setGravity(Particles *particles, real *g_ax);
        }

#elif DIM == 2
        __global__ void setGravity(Particles *particles, real *g_ax, real *g_ay);

        namespace Launch {
            void setGravity(Particles *particles, real *g_ax, real *g_ay);
        }
#else
        __global__ void setGravity(Particles *particles, real *g_ax, real *g_ay, real *g_az);

        namespace Launch {
            void setGravity(Particles *particles, real *g_ax, real *g_ay, real *g_az);
        }
#endif

#if DIM == 1
        __global__ void setLeapfrog(Particles *particles, real *ax_old, real *g_ax_old);

        namespace Launch {
            void setLeapfrog(Particles *particles, real *ax_old, real *g_ax_old);
        }

#elif DIM == 2
        __global__ void setLeapfrog(Particles *particles, real *ax_old, real *ay_old, real *g_ax_old, real *g_ay_old);

        namespace Launch {
            void setLeapfrog(Particles *particles, real *ax_old, real *ay_old, real *g_ax_old, real *g_ay_old);
        }
#else
        __global__ void setLeapfrog(Particles *particles, real *ax_old, real *ay_old, real *az_old, real *g_ax_old,
                                    real *g_ay_old, real *g_az_old);

        namespace Launch {
            void setLeapfrog(Particles *particles, real *ax_old, real *ay_old, real *az_old, real *g_ax_old,
                             real *g_ay_old, real *g_az_old);
        }
#endif

#if MESHLESS_FINITE_METHOD
    __global__ void setMeshlessFinite(Particles *particles, real *omega, real *psix, real* psiy, real *psiz,
                                      real *massFlux, real *vxFlux, real *vyFlux, real *vzFlux, real *energyFlux,
                                      real *Ncond,real *rhoGrad, real *vxGrad, real *vyGrad, real *vzGrad, real *pGrad);
        namespace Launch {
            void setMeshlessFinite(Particles *particles, real *omega, real *psix, real* psiy, real *psiz,
                                   real *massFlux, real *vxFlux, real *vyFlux, real *vzFlux, real *energyFlux,
                                   real *Ncond, real *rhoGrad, real *vxGrad, real *vyGrad, real *vzGrad, real *pGrad);
        }
#endif

        __global__ void setU(Particles *particles, real *u);
        namespace Launch {
            void setU(Particles *particles, real *u);
        }

        __global__ void setNodeType(Particles *particles, integer *nodeType);
        namespace Launch {
            void setNodeType(Particles *particles, integer *nodeType);
        }

        __global__ void setArtificialViscosity(Particles *particles, real *muijmax);
        namespace Launch {
            void setArtificialViscosity(Particles *particles, real *muijmax);
        }

#if BALSARA_SWITCH
        __global__ void setDivCurl(Particles *particles, real *divv, real *curlv);
        namespace Launch {
            void setDivCurl(Particles *particles, real *divv, real *curlv);
        }
#endif

//#if INTEGRATE_DENSITY
        /**
         * Kernel call setter, in dependence of `INTEGRATE_DENSITY`
         *
         * @param particles
         * @param drhodt
         */
        __global__ void setIntegrateDensity(Particles *particles, real *drhodt);
        namespace Launch {
            /**
             * Wrapped kernel call setter, in dependence of `INTEGRATE_DENSITY`
             *
             * @param particles
             * @param drhodt
             */
            void setIntegrateDensity(Particles *particles, real *drhodt);
        }
//#endif
#if VARIABLE_SML || INTEGRATE_SML
        /**
         * Kernel call to setter, in dependence of `VARIABLE_SML`
         *
         * @param particles
         * @param dsmldt
         */
        __global__ void setVariableSML(Particles *particles, real *dsmldt);
        namespace Launch {
            /**
            * Wrapped kernel call to setter, in dependence of `VARIABLE_SML`
            *
            * @param particles
            * @param dsmldt
            */
            void setVariableSML(Particles *particles, real *dsmldt);
        }
#endif
#if SML_CORRECTION
        __global__ void setSMLCorrection(Particles *particles, real *sml_omega);

        namespace Launch {
            void setSMLCorrection(Particles *particles, real *sml_omega);
        }
#endif
#if NAVIER_STOKES
        __global__ void setNavierStokes(Particles *particles, real *Tshear, real *eta);
        namespace Launch {
            void setNavierStokes(Particles *particles, real *Tshear, real *eta);
        }
#endif
#if SOLID
        /**
         * Kernel call to setter, in dependence of `SOLID`
         *
         * @param particles
         * @param S
         * @param dSdt
         * @param localStrain
         */
        __global__ void setSolid(Particles *particles, real *S, real *dSdt, real *localStrain);
        namespace Launch {
            /**
             * Wrapped kernel call to setter, in dependence of `SOLID`
             *
             * @param particles
             * @param S
             * @param dSdt
             * @param localStrain
             */
            void setSolid(Particles *particles, real *S, real *dSdt, real *localStrain);
        }
#endif
#if SOLID || NAVIER_STOKES
        /**
         * Kernel call to setter, in dependence of `SOLID` or `NAVIER_STOKES`
         *
         * @param particles
         * @param sigma
         */
        __global__ void setSolidNavierStokes(Particles *particles, real *sigma);
        namespace Launch {
            /**
             * Wrapped kernel call to setter, in dependence of `SOLID` or `NAVIER_STOKES`
             *
             * @param particles
             * @param sigma
             */
            void setSolidNavierStokes(Particles *particles, real *sigma);
        }
#endif
#if ARTIFICIAL_STRESS
        /**
         * Kernel call to setter, in dependence of `ARTIFICIAL_STRESS`
         *
         * @param particles
         * @param R
         */
        __global__ void setArtificialStress(Particles *particles, real *R);
        namespace Launch {
            /**
             * Wrapped kernel call to setter, in dependence of `ARTIFICIAL_STRESS`
             *
             * @param particles
             * @param R
             */
            void setArtificialStress(Particles *particles, real *R);
        }
#endif
#if POROSITY
        /**
         * Kernel call to setter, in dependence of `POROSITY`
         *
         * @param particles
         * @param pold
         * @param alpha_jutzi
         * @param alpha_jutzi_old
         * @param dalphadt
         * @param dalphadp
         * @param dp
         * @param dalphadrho
         * @param f
         * @param delpdelrho
         * @param delpdele
         * @param cs_old
         * @param alpha_epspor
         * @param dalpha_epspordt
         * @param epsilon_v
         * @param depsilon_vdt
         */
        __global__ void setPorosity(Particles *particles, real *pold, real *alpha_jutzi, real *alpha_jutzi_old, real *dalphadt,
                                    real *dalphadp, real *dp, real *dalphadrho, real *f, real *delpdelrho,
                                    real *delpdele, real *cs_old, real *alpha_epspor, real *dalpha_epspordt,
                                    real *epsilon_v, real *depsilon_vdt);
        namespace Launch {
            /**
             * Wrapped kernel call to setter, in dependence of `POROSITY`
             *
             * @param particles
             * @param pold
             * @param alpha_jutzi
             * @param alpha_jutzi_old
             * @param dalphadt
             * @param dalphadp
             * @param dp
             * @param dalphadrho
             * @param f
             * @param delpdelrho
             * @param delpdele
             * @param cs_old
             * @param alpha_epspor
             * @param dalpha_epspordt
             * @param epsilon_v
             * @param depsilon_vdt
             */
            void setPorosity(Particles *particles, real *pold, real *alpha_jutzi, real *alpha_jutzi_old, real *dalphadt,
                             real *dalphadp, real *dp, real *dalphadrho, real *f, real *delpdelrho,
                             real *delpdele, real *cs_old, real *alpha_epspor, real *dalpha_epspordt,
                             real *epsilon_v, real *depsilon_vdt);
        }
#endif
#if ZERO_CONSISTENCY
        /**
         * Kernel call to setter, in dependence of `ZERO_CONSISTENCY`
         *
         * @param particles
         * @param shepardCorrection
         */
        __global__ void setZeroConsistency(Particles *particles, real *shepardCorrection);
        namespace Launch {
            /**
             * Wrapped kernel call to setter, in dependence of `ZERO_CONSISTENCY`
             *
             * @param particles
             * @param shepardCorrection
             */
            void setZeroConsistency(Particles *particles, real *shepardCorrection);
        }
#endif
#if LINEAR_CONSISTENCY
        /**
         * Kernel call to setter, in dependencee of `LINEAR_CONSISTENCY`
         *
         * @param particles
         * @param tensorialCorrectionMatrix
         */
        __global__ void setLinearConsistency(Particles *particles, real *tensorialCorrectionMatrix);
        namespace Launch {
            /**
             * Wrapped kernel call to setter, in dependencee of `LINEAR_CONSISTENCY`
             *
             * @param particles
             * @param tensorialCorrectionMatrix
             */
            void setLinearConsistency(Particles *particles, real *tensorialCorrectionMatrix);
        }
#endif
#if FRAGMENTATION
        /**
         * Kernel call to setter, in dependence of `FRAGMENTATION`
         *
         * @param particles
         * @param d
         * @param damage_total
         * @param dddt
         * @param numFlaws
         * @param maxNumFlaws
         * @param numActiveFlaws
         * @param flaws
         */
        __global__ void setFragmentation(Particles *particles, real *d, real *damage_total, real *dddt, integer *numFlaws,
                                         integer *maxNumFlaws, integer *numActiveFlaws, real *flaws);
        namespace Launch {
            /**
            * Wrapped kernel call to setter, in dependence of `FRAGMENTATION`
            *
            * @param particles
            * @param d
            * @param damage_total
            * @param dddt
            * @param numFlaws
            * @param maxNumFlaws
            * @param numActiveFlaws
            * @param flaws
            */
            void setFragmentation(Particles *particles, real *d, real *damage_total, real *dddt, integer *numFlaws,
                                  integer *maxNumFlaws, integer *numActiveFlaws, real *flaws);
        }
#if PALPHA_POROSITY
        /**
         * Kernel call to setter, in dependence of `PALPHA_POROSITY`
         *
         * @param particles
         * @param damage_porjutzi
         * @param ddamage_porjutzidt
         */
        __global__ void setPalphaPorosity(Particles *particles, real *damage_porjutzi, real *ddamage_porjutzidt);
        namespace Launch {
            /**
             * Wrapped kernel call to setter, in dependence of `PALPHA_POROSITY`
             *
             * @param particles
             * @param damage_porjutzi
             * @param ddamage_porjutzidt
             */
            void setPalphaPorosity(Particles *particles, real *damage_porjutzi, real *ddamage_porjutzidt);
        }
#endif
#endif

        /**
         * Test kernel (for debugging purposes)
         *
         * @param particles
         */
        __global__ void test(Particles *particles);

        namespace Launch {
            /**
             * Wrapped test kernel (for debugging purposes)
             *
             * @param particles
             */
            real test(Particles *particles, bool time=false);
        }
    }

}

/**
 * Class for buffering particle information needed for integration
 *
 * Multiple instances of this class can be used in dependence of the integrator's order.
 */
class IntegratedParticles {

public:

    /// unique identifier
    idInteger *uid;

    ///
    real *x;
    real *vx;
    real *ax;
#if DIM > 1
    ///
    real *y;
    real *vy;
    real *ay;
#if DIM == 3
    ///
    real *z;
    real *vz;
    real *az;
#endif
#endif

    //integer *level;
    //integer *nodeType;

    real *rho;
    real *e;
    real *dedt;
    real *p;
    real *cs;

    real *sml;

//#if INTEGRATE_DENSITY
    real *drhodt;
//#endif

#if VARIABLE_SML || INTEGRATE_SML
    real *dsmldt;
#endif

    /**
     * Default constructor
     */
    CUDA_CALLABLE_MEMBER IntegratedParticles();

#if DIM == 1

    CUDA_CALLABLE_MEMBER IntegratedParticles(idInteger *uid, real *rho, real *e, real *dedt, real *p, real *cs, real *x,
                                             real *vx, real *ax);

    CUDA_CALLABLE_MEMBER void set(idInteger *uid, real *rho, real *e, real *dedt, real *p, real *cs, real *x,
                                  real *vx, real *ax);

#elif DIM == 2

    CUDA_CALLABLE_MEMBER IntegratedParticles(idInteger *uid, real *rho, real *e, real *dedt, real *p, real *cs, real *x,
                                             real *y, real *vx, real *vy, real *ax, real *ay);

    CUDA_CALLABLE_MEMBER void set(integer *uid, real *rho, real *e, real *dedt, real *p, real *cs, real *x,
                                  real *y, real *vx, real *vy, real *ax, real *ay);

#else

    CUDA_CALLABLE_MEMBER IntegratedParticles(idInteger *uid, real *rho, real *e, real *dedt, real *p, real *cs, real *x,
                                             real *y, real *z, real *vx, real *vy, real *vz, real *ax,
                                             real *ay, real *az);

    CUDA_CALLABLE_MEMBER void set(idInteger *uid, real *rho, real *e, real *dedt, real *p, real *cs, real *x,
                                  real *y, real *z, real *vx, real *vy, real *vz, real *ax,
                                  real *ay, real *az);

#endif

    //CUDA_CALLABLE_MEMBER void setLevel(integer *level);
    //CUDA_CALLABLE_MEMBER void setNodeType(integer *nodeType);

    CUDA_CALLABLE_MEMBER void setSML(real *sml);

//#if INTEGRATE_DENSITY
    CUDA_CALLABLE_MEMBER void setIntegrateDensity(real *drhodt);
//#endif

#if VARIABLE_SML || INTEGRATE_SML
    CUDA_CALLABLE_MEMBER void setIntegrateSML(real *dsmldt);
#endif

    /**
     * Reset (specific) entries
     *
     * @param index index to be resetted
     */
    CUDA_CALLABLE_MEMBER void reset(integer index);

    /**
     * Destructor
     */
    CUDA_CALLABLE_MEMBER ~IntegratedParticles();

};

namespace IntegratedParticlesNS {

    namespace Kernel {

#if DIM == 1

        __global__ void set(IntegratedParticles *integratedParticles, idInteger *uid, real *rho, real *e, real *dedt,
                            real *p, real *cs, real *x, real *vx, real *ax);

        namespace Launch {

            void set(IntegratedParticles *integratedParticles, idInteger *uid, real *rho, real *e, real *dedt,
                     real *p, real *cs, real *x, real *vx, real *ax);
        }

#elif DIM == 2

        __global__ void set(IntegratedParticles *integratedParticles, idInteger *uid, real *rho, real *e, real *dedt,
                            real *p, real *cs, real *x, real *y, real *vx, real *vy, real *ax, real *ay);

        namespace Launch {

            void set(IntegratedParticles *integratedParticles, idInteger *uid, real *rho, real *e, real *dedt,
                     real *p, real *cs, real *x, real *y, real *vx, real *vy, real *ax, real *ay);
        }

#else


        __global__ void set(IntegratedParticles *integratedParticles, idInteger *uid, real *rho, real *e, real *dedt,
                            real *p, real *cs, real *x, real *y, real *z, real *vx, real *vy, real *vz, real *ax,
                            real *ay, real *az);

        namespace Launch {

            void set(IntegratedParticles *integratedParticles, idInteger *uid, real *rho, real *e, real *dedt,
                     real *p, real *cs, real *x, real *y, real *z, real *vx, real *vy, real *vz, real *ax,
                     real *ay, real *az);
        }

#endif

        //__global__ void setLevel(IntegratedParticles *integratedParticles, integer *level);
        //namespace Launch {
        //    void setLevel(IntegratedParticles *integratedParticles, integer *level);
        //}

        //__global__ void setNodeType(IntegratedParticles *integratedParticles, integer *nodeType);
        //namespace Launch {
        //    void setNodeType(IntegratedParticles *integratedParticles, integer *nodeType);
        //}

        __global__ void setSML(IntegratedParticles *integratedParticles, real *sml);

        namespace Launch {
            void setSML(IntegratedParticles *integratedParticles, real *sml);
        }

//#if INTEGRATE_DENSITY
        __global__ void setIntegrateDensity(IntegratedParticles *integratedParticles, real *drhodt);

        namespace Launch {

            void setIntegrateDensity(IntegratedParticles *integratedParticles, real *drhodt);

        }
//#endif

#if VARIABLE_SML || INTEGRATE_SML
        __global__ void setIntegrateSML(IntegratedParticles *integratedParticles, real *dsmldt);

        namespace Launch {
            void setIntegrateSML(IntegratedParticles *integratedParticles, real *dsmldt);
        }
#endif

    }
}

#endif //MILUPHPC_PARTICLES_CUH
