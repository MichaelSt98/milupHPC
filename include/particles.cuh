#ifndef MILUPHPC_PARTICLES_CUH
#define MILUPHPC_PARTICLES_CUH

#include "parameter.h"
#include "cuda_utils/cuda_utilities.cuh"

/**
 * Particle(s) class based on SoA (Structur of Arrays).
 */
class Particles {

public:

    integer *numParticles;
    integer *numNodes;

    /// (pointer to) mass (array)
    real *mass;
    /// (pointer to) x position (array)
    real *x;
    /// (pointer to) x velocity (array)
    real *vx;
    /// (pointer to) x acceleration (array)
    real *ax;
#if DIM > 1
    /// (pointer to) y position (array)
    real *y;
    /// (pointer to) y velocity (array)
    real *vy;
    /// (pointer to) y acceleration (array)
    real *ay;
#if DIM == 3
    /// (pointer to) z position (array)
    real *z;
    /// (pointer to) z velocity (array)
    real *vz;
    /// (pointer to) z acceleration (array)
    real *az;
#endif
#endif

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

    /// (pointer to) sound of speed (array)
    real *cs; // soundspeed

    // simplest hydro
    /// (pointer to) density (array)
    real *rho; // density
    /// (pointer to) pressure (array)
    real *p; // pressure

#if INTEGRATE_DENSITY
    // integrated density
    /// (pointer to) time derivative of density (array)
    real *drhodt;
#endif

#if VARIABLE_SML
    // integrate/variable smoothing length
    /// (pointer to) time derivative of smoothing length (array)
    real *dsmldt;
#endif

#if SOLID
    /// (pointer to) deviatoric stress tensor (array)
    real *S; // deviatoric stress tensor (DIM * DIM)
    /// (pointer to) time derivative of deviatoric stress tensor (array)
    real *dSdt;
    /// (pointer to) local strain (array)
    real *localStrain; // local strain
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

    /**
     * Constructor, passing pointer to member variables
     *
     * @param numParticles number of Particles
     * @param numNodes number of nodes
     * @param mass mass
     * @param x x-position
     * @param vx x-velocity
     * @param ax x-acceleration
     * @param uid unique identifier
     * @param materialId material identifier
     * @param sml smoothing length
     * @param nnl near neighbour list
     * @param noi number of interaction
     * @param e energy
     * @param dedt time derivative of energy
     * @param cs speed of sound
     * @param rho density
     * @param p pressure
     */
    CUDA_CALLABLE_MEMBER Particles(integer *numParticles, integer *numNodes, real *mass, real *x, real *vx, real *ax,
                                   idInteger *uid, integer *materialId, real *sml, integer *nnl, integer *noi, real *e,
                                   real *dedt, real *cs, real *rho, real *p);

    /**
     * Setter, passing pointer to member variables
     *
     * @param numParticles number of Particles
     * @param numNodes number of nodes
     * @param mass mass
     * @param x x-position
     * @param vx x-velocity
     * @param ax x-acceleration
     * @param uid unique identifier
     * @param materialId material identifier
     * @param sml smoothing length
     * @param nnl near neighbour list
     * @param noi number of interaction
     * @param e energy
     * @param dedt time derivative of energy
     * @param cs speed of sound
     * @param rho density
     * @param p pressure
     */
    CUDA_CALLABLE_MEMBER void set(integer *numParticles, integer *numNodes, real *mass, real *x, real *vx, real *ax,
                                  idInteger *uid, integer *materialId, real *sml, integer *nnl, integer *noi, real *e,
                                  real *dedt, real *cs, real *rho, real *p);
#if DIM > 1
    /**
     * Constructor, passing pointer to member variables
     *
     * @param numParticles number of Particles
     * @param numNodes number of nodes
     * @param mass mass
     * @param x x-position
     * @param y y-position
     * @param vx x-velocity
     * @param vy y-velocity
     * @param ax x-acceleration
     * @param ay y-acceleration
     * @param uid unique identifier
     * @param materialId material identifier
     * @param sml smoothing length
     * @param nnl near neighbour list
     * @param noi number of interaction
     * @param e energy
     * @param dedt time derivative of energy
     * @param cs speed of sound
     * @param rho density
     * @param p pressure
     */
    CUDA_CALLABLE_MEMBER Particles(integer *numParticles, integer *numNodes, real *mass, real *x, real *y, real *vx,
                                   real *vy, real *ax, real *ay, idInteger *uid, integer *materialId, real *sml,
                                   integer *nnl, integer *noi, real *e, real *dedt, real *cs, real *rho, real *p);

    /**
     * Setter, passing pointer to member variables
     *
     * @param numParticles number of Particles
     * @param numNodes number of nodes
     * @param mass mass
     * @param x x-position
     * @param y y-position
     * @param vx x-velocity
     * @param vy y-velocity
     * @param ax x-acceleration
     * @param ay y-acceleration
     * @param uid unique identifier
     * @param materialId material identifier
     * @param sml smoothing length
     * @param nnl near neighbour list
     * @param noi number of interaction
     * @param e energy
     * @param dedt time derivative of energy
     * @param cs speed of sound
     * @param rho density
     * @param p pressure
     */
    CUDA_CALLABLE_MEMBER void set(integer *numParticles, integer *numNodes, real *mass, real *x, real *y, real *vx,
                                  real *vy, real *ax, real *ay, idInteger *uid, integer *materialId, real *sml,
                                  integer *nnl, integer *noi, real *e, real *dedt, real *cs, real *rho, real *p);
#if DIM == 3
    /**
     * Constructor, passing pointer to member variables
     *
     * @param numParticles number of Particles
     * @param numNodes number of nodes
     * @param mass mass
     * @param x x-position
     * @param y y-position
     * @param z z-position
     * @param vx x-velocity
     * @param vy y-velocity
     * @param vz z-velocity
     * @param ax x-acceleration
     * @param ay y-acceleration
     * @param az z-acceleration
     * @param uid unique identifier
     * @param materialId material identifier
     * @param sml smoothing length
     * @param nnl near neighbour list
     * @param noi number of interaction
     * @param e energy
     * @param dedt time derivative of energy
     * @param cs speed of sound
     * @param rho density
     * @param p pressure
     */
    CUDA_CALLABLE_MEMBER Particles(integer *numParticles, integer *numNodes, real *mass, real *x, real *y, real *z,
                                   real *vx, real *vy, real *vz, real *ax, real *ay, real *az, idInteger *uid,
                                   integer *materialId, real *sml, integer *nnl, integer *noi, real *e, real *dedt,
                                   real *cs, real *rho, real *p);

    /**
    * Setter, passing pointer to member variables
    *
    * @param numParticles number of Particles
    * @param numNodes number of nodes
    * @param mass mass
    * @param x x-position
    * @param y y-position
    * @param z z-position
    * @param vx x-velocity
    * @param vy y-velocity
    * @param vz z-velocity
    * @param ax x-acceleration
    * @param ay y-acceleration
    * @param az z-acceleration
    * @param uid unique identifier
    * @param materialId material identifier
    * @param sml smoothing length
    * @param nnl near neighbour list
    * @param noi number of interaction
    * @param e energy
    * @param dedt time derivative of energy
    * @param cs speed of sound
    * @param rho density
    * @param p pressure
    */
    CUDA_CALLABLE_MEMBER void set(integer *numParticles, integer *numNodes, real *mass, real *x, real *y, real *z,
                                  real *vx, real *vy, real *vz, real *ax, real *ay, real *az, idInteger *uid,
                                  integer *materialId, real *sml, integer *nnl, integer *noi, real *e, real *dedt,
                                  real *cs, real *rho, real *p);
#endif
#endif

#if INTEGRATE_DENSITY
    /**
     * Setter in dependence of `INTEGRATE_DENSITY`
     *
     * @param drhodt time derivative of density
     */
    CUDA_CALLABLE_MEMBER void setIntegrateDensity(real *drhodt);
#endif
#if VARIABLE_SML
    /**
     * Setter, in dependence of `VARIABLE_SML`
     *
     * @param dsmldt time derivative of smoothing length
     */
    CUDA_CALLABLE_MEMBER void setVariableSML(real *dsmldt);
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
    CUDA_CALLABLE_MEMBER void setNavierStokes(real *sigma);
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
     * Reset (specific) entries
     *
     * @param index index of entry to be resetted
     */
    CUDA_CALLABLE_MEMBER void reset(integer index);

    /**
     * Distance of two particles
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
         * Kernel call to setter
         *
         * @param particles
         * @param numParticles
         * @param numNodes
         * @param mass
         * @param x
         * @param vx
         * @param ax
         * @param uid
         * @param materialId
         * @param sml
         * @param nnl
         * @param noi
         * @param e
         * @param dedt
         * @param cs
         * @param rho
         * @param p
         */
        __global__ void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x,
                            real *vx, real *ax, idInteger *uid, integer *materialId, real *sml, integer *nnl,
                            integer *noi, real *e, real *dedt, real *cs, real *rho, real *p);

        namespace Launch {
            /**
             * Wrapped Kernel call to setter
             *
             * @param particles
             * @param numParticles
             * @param numNodes
             * @param mass
             * @param x
             * @param vx
             * @param ax
             * @param uid
             * @param materialId
             * @param sml
             * @param nnl
             * @param noi
             * @param e
             * @param dedt
             * @param cs
             * @param rho
             * @param p
             */
            void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x, real *vx,
                     real *ax, idInteger *uid, integer *materialId, real *sml, integer *nnl, integer *noi, real *e,
                     real *dedt, real *cs, real *rho, real *p);
        }

        /**
         * Info Kernel (for debugging purposes)
         *
         * @param particles
         * @param n
         * @param m
         * @param k
         */
        __global__ void info(Particles *particles, integer n, integer m, integer k);

        namespace Launch {
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

#if DIM > 1
        /**
         * Kernel call to setter
         *
         * @param particles
         * @param numParticles
         * @param numNodes
         * @param mass
         * @param x
         * @param y
         * @param vx
         * @param vy
         * @param ax
         * @param ay
         * @param uid
         * @param materialId
         * @param sml
         * @param nnl
         * @param noi
         * @param e
         * @param dedt
         * @param cs
         * @param rho
         * @param p
         */
        __global__ void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x,
                            real *y, real *vx, real *vy, real *ax, real *ay, idInteger *uid, integer *materialId,
                            real *sml, integer *nnl, integer *noi, real *e, real *dedt, real *cs, real *rho, real *p);

        namespace Launch {
            /**
             * Wrapped kernel call to setter
             *
             * @param particles
             * @param numParticles
             * @param numNodes
             * @param mass
             * @param x
             * @param y
             * @param vx
             * @param vy
             * @param ax
             * @param ay
             * @param id
             * @param materialId
             * @param sml
             * @param nnl
             * @param noi
             * @param e
             * @param dedt
             * @param cs
             * @param rho
             * @param p
             */
            void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x, real *y,
                     real *vx, real *vy, real *ax, real *ay, idInteger *id, integer *materialId, real *sml, integer *nnl,
                     integer *noi, real *e, real *dedt, real *cs, real *rho, real *p);
        }

#if DIM == 3

        /**
         * Kernel call to setter
         *
         * @param particles
         * @param numParticles
         * @param numNodes
         * @param mass
         * @param x
         * @param y
         * @param z
         * @param vx
         * @param vy
         * @param vz
         * @param ax
         * @param ay
         * @param az
         * @param uid
         * @param materialId
         * @param sml
         * @param nnl
         * @param noi
         * @param e
         * @param dedt
         * @param cs
         * @param rho
         * @param p
         */
        __global__ void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x,
                            real *y, real *z, real *vx, real *vy, real *vz, real *ax, real *ay, real *az,
                            idInteger *uid, integer *materialId, real *sml, integer *nnl, integer *noi, real *e,
                            real *dedt, real *cs, real *rho, real *p);

        namespace Launch {
            /**
             * Wrapped kernel call to setter
             *
             * @param particles
             * @param numParticles
             * @param numNodes
             * @param mass
             * @param x
             * @param y
             * @param z
             * @param vx
             * @param vy
             * @param vz
             * @param ax
             * @param ay
             * @param az
             * @param id
             * @param materialId
             * @param sml
             * @param nnl
             * @param noi
             * @param e
             * @param dedt
             * @param cs
             * @param rho
             * @param p
             */
            void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x, real *y,
                     real *z, real *vx, real *vy, real *vz, real *ax, real *ay, real *az, idInteger *id,
                     integer *materialId, real *sml, integer *nnl, integer *noi, real *e, real *dedt, real *cs,
                     real *rho, real *p);
        }

#endif
#endif

#if INTEGRATE_DENSITY
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
#endif
#if VARIABLE_SML
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
        __global__ void setNavierStokes(Particles *particles, real *sigma);
        namespace Launch {
            /**
             * Wrapped kernel call to setter, in dependence of `SOLID` or `NAVIER_STOKES`
             *
             * @param particles
             * @param sigma
             */
            void setNavierStokes(Particles *particles, real *sigma);
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
    integer *uid;

    /// time derivative of the density
    real *drhodt;

    /// time derivative of the x position and velocity
    real *dxdt, *dvxdt;
#if DIM > 1
    /// time derivative of the y position and velocity
    real *dydt, *dvydt;
#if DIM == 3
    /// time derivative of the z position and velocity
    real *dzdt, *dvzdt;
#endif
#endif

    /**
     * Default constructor
     */
    CUDA_CALLABLE_MEMBER IntegratedParticles();

    /**
     * Constructor
     *
     * @param uid unique identifier
     * @param drhodt time derivative of density
     * @param dxdt time derivative of the x position
     * @param dvxdt time derivative of the x velocity
     */
    CUDA_CALLABLE_MEMBER IntegratedParticles(integer *uid, real *drhodt, real *dxdt, real *dvxdt);

    /**
     * Setter
     *
     * @param uid unique identifier
     * @param drhodt time derivative of density
     * @param dxdt time derivative of the x position
     * @param dvxdt time derivative of the y velocity
     */
    CUDA_CALLABLE_MEMBER void set(integer *uid, real *drhodt, real *dxdt, real *dvxdt);

#if DIM > 1
    /**
     * Constructor
     *
     * @param uid unique identifier
     * @param drhodt time derivative of density
     * @param dxdt time derivative of the x position
     * @param dydt time derivative of the y position
     * @param dvxdt time derivative of the x velocity
     * @param dvydt time derivative of the y velocity
     */
    CUDA_CALLABLE_MEMBER IntegratedParticles(integer *uid, real *drhodt, real *dxdt, real *dydt, real *dvxdt,
                                             real *dvydt);

    /**
     * Setter
     *
     * @param uid unique identifier
     * @param drhodt time derivative of density
     * @param dxdt time derivative of the x position
     * @param dydt time derivative of the y position
     * @param dvxdt time derivative of the x velocity
     * @param dvydt time derivative of the y velocity
     */
    CUDA_CALLABLE_MEMBER void set(integer *uid, real *drhodt, real *dxdt, real *dydt, real *dvxdt,
                                  real *dvydt);

#if DIM == 3

    /**
     * Constructor
     *
     * @param uid unique identifier
     * @param drhodt time derivative of density
     * @param dxdt time derivative of the x position
     * @param dydt time derivative of the y position
     * @param dzdt time derivative of the z position
     * @param dvxdt time derivative of the x velocity
     * @param dvydt time derivative of the y velocity
     * @param dvzdt time derivative of the z velocity
     */
    CUDA_CALLABLE_MEMBER IntegratedParticles(integer *uid, real *drhodt, real *dxdt, real *dydt, real *dzdt,
                                             real *dvxdt, real *dvydt, real *dvzdt);

    /**
     * Setter
     *
     * @param uid unique identifier
     * @param drhodt time derivative of density
     * @param dxdt time derivative of the x position
     * @param dydt time derivative of the y position
     * @param dzdt time derivative of the z position
     * @param dvxdt time derivative of the x velocity
     * @param dvydt time derivative of the y velocity
     * @param dvzdt time derivative of the z velocity
     */
    CUDA_CALLABLE_MEMBER void set(integer *uid, real *drhodt, real *dxdt, real *dydt, real *dzdt,
                                  real *dvxdt, real *dvydt, real *dvzdt);

#endif
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
        /**
         * Kernel call to setter
         *
         * @param integratedParticles
         * @param uid
         * @param drhodt
         * @param dxdt
         * @param dvxdt
         */
        __global__ void set(IntegratedParticles *integratedParticles, integer *uid, real *drhodt, real *dxdt,
                            real *dvxdt);

        namespace Launch {
            /**
            * Wrapped kernel call to setter
            *
            * @param integratedParticles
            * @param uid
            * @param drhodt
            * @param dxdt
            * @param dvxdt
            */
            void set(IntegratedParticles *integratedParticles, integer *uid, real *drhodt, real *dxdt,
                     real *dvxdt);
        }

#if DIM > 1
        /**
         * Kernel call to setter
         *
         * @param integratedParticles
         * @param uid
         * @param drhodt
         * @param dxdt
         * @param dydt
         * @param dvxdt
         * @param dvydt
         */
        __global__ void set(IntegratedParticles *integratedParticles, integer *uid, real *drhodt, real *dxdt,
                            real *dydt, real *dvxdt, real *dvydt);

        namespace Launch {
            /**
             * Wrapped kernel call to setter
             *
             * @param integratedParticles
             * @param uid
             * @param drhodt
             * @param dxdt
             * @param dydt
             * @param dvxdt
             * @param dvydt
             */
            void set(IntegratedParticles *integratedParticles, integer *uid, real *drhodt, real *dxdt,
                     real *dydt, real *dvxdt, real *dvydt);
        }

#if DIM == 3

        /**
         * Kernel call to setter
         *
         * @param integratedParticles
         * @param uid
         * @param drhodt
         * @param dxdt
         * @param dydt
         * @param dzdt
         * @param dvxdt
         * @param dvydt
         * @param dvzdt
         */
        __global__ void set(IntegratedParticles *integratedParticles, integer *uid, real *drhodt, real *dxdt,
                            real *dydt, real *dzdt, real *dvxdt, real *dvydt, real *dvzdt);

        namespace Launch {
            /**
             * Wrapped kernel call to setter
             *
             * @param integratedParticles
             * @param uid
             * @param drhodt
             * @param dxdt
             * @param dydt
             * @param dzdt
             * @param dvxdt
             * @param dvydt
             * @param dvzdt
             */
            void set(IntegratedParticles *integratedParticles, integer *uid, real *drhodt, real *dxdt,
                     real *dydt, real *dzdt, real *dvxdt, real *dvydt, real *dvzdt);
        }

#endif
#endif

    }
}

#endif //MILUPHPC_PARTICLES_CUH
