#ifndef MILUPHPC_PARTICLE_HANDLER_H
#define MILUPHPC_PARTICLE_HANDLER_H

#include "parameter.h"
#include "particles.cuh"
#include "utils/logger.h"
#include "cuda_utils/cuda_runtime.h"

class IntegratedParticleHandler;

/**
 * Handling `Particles` class
 *
 * Allocate and handle both host and device arrays and instances
 */
class ParticleHandler {

public:

    bool leapfrog { false }; // TODO: how to change that?

    /// (host) number of particles
    integer numParticles;
    /// (host) number of nodes
    integer numNodes;

    /// host mass
    real *h_mass;
    /// host x position
    real *h_x, *_h_x;
    /// host x velocity
    real *h_vx, *_h_vx;
    /// host x acceleration
    real *h_ax, *_h_ax;
    real *h_g_ax;
    real *h_ax_old, *h_g_ax_old; // only for leapfrog integrator
#if DIM > 1
    /// host y position
    real *h_y, *_h_y;
    /// host y velocity
    real *h_vy, *_h_vy;
    /// host y acceleration
    real *h_ay, *_h_ay;
    real *h_g_ay;
    real *h_ay_old, *h_g_ay_old; // only for leapfrog integrator
#if DIM == 3
    /// host z position
    real *h_z, *_h_z;
    /// host z velocity
    real *h_vz, *_h_vz;
    /// host z acceleration
    real *h_az, *_h_az;
    real *h_g_az;
    real *h_az_old, *h_g_az_old; // only for leapfrog integrator
#endif
#endif

    integer *h_nodeType;

    integer *h_level;
    /// host unique identifier
    idInteger *h_uid, *_h_uid; // unique identifier (unsigned int/long?)
    /// host material identifier
    integer *h_materialId; // material identfier (e.g.: ice, basalt, ...)
    /// host smoothing length
    real *h_sml, *_h_sml; // smoothing length
    /// host near(est) neighbor list
    integer *h_nnl; // max(number of interactions)
    /// host number of interactions
    integer *h_noi; // number of interactions (alternatively initialize nnl with -1, ...)
    /// host internal energy
    real *h_e, *_h_e; // internal energy
    /// host time derivative of internal energy
    real *h_dedt, *_h_dedt;
    /// energy
    real *h_u;
    /// host speed of sound
    real *h_cs, *_h_cs; // soundspeed
    /// host density
    real *h_rho, *_h_rho; // density
    /// host pressure
    real *h_p, *_h_p; // pressure
    /// host max(mu_ij)
    real *h_muijmax;

//#if INTEGRATE_DENSITY
    /// host time derivative of density
    real *h_drhodt, *_h_drhodt;
//#endif
#if VARIABLE_SML || INTEGRATE_SML
    /// host time derivative of smoothing length
    real *h_dsmldt, *_h_dsmldt;
#endif
#if SML_CORRECTION
    real *h_sml_omega;
#endif
#if NAVIER_STOKES
    real *h_Tshear;
    real *h_eta;
#endif
#if SOLID
    /// host deviatoric stress tensor
    real *h_S;
    /// host time deriative of deviatoric stress tensor
    real *h_dSdt;
    /// host local strain
    real *h_localStrain;
#endif
#if SOLID || NAVIER_STOKES
    /// host sigma/stress tensor
    real *h_sigma;
#endif
#if ARTIFICIAL_STRESS
    /// host tensile instability
    real *h_R;
#endif
#if BALSARA_SWITCH
    real *h_divv;
    real *h_curlv;
#endif
#if POROSITY
    /// host pressure of the sph particle after the last successful timestep
    real *h_pold;
    /// host current distension of the sph particle
    real *h_alpha_jutzi;
    /// host distension of the sph particle after the last successful timestep
    real *h_alpha_jutzi_old;
    /// host time derivative of the distension
    real *h_dalphadt;
    /// host partial derivative of the distension with respect to the pressure
    real *h_dalphadp;
    /// host difference in pressure from the last timestep to the current one
    real *h_dp;
    /// host partial derivative of the distension with respect to the density
    real *h_dalphadrho;
    /// host additional factor to reduce the deviatoric stress tensor according to Jutzi
    real *h_f;
    /// host the partial derivative of the pressure with respect to the density
    real *h_delpdelrho;
    /// host the partial derivative of the pressure with respect to the specific internal energy
    real *h_delpdele;
    /// host the sound speed after the last successful timestep
    real *h_cs_old;
    /// host distention in the strain-\alpha model
    real *h_alpha_epspor;
    /// host time derivative of the distension
    real *h_dalpha_epspordt;
    /// host volume change (trace of strain rate tensor)
    real *h_epsilon_v;
    /// host time derivative of volume change
    real *h_depsilon_vdt;
#endif
#if ZERO_CONSISTENCY
    /// host correction (value) for zeroth order consistency
    real *h_shepardCorrection;
#endif
#if LINEAR_CONSISTENCY
    /// host correction matrix for linear order consistency
    real *h_tensorialCorrectionMatrix;
#endif
#if FRAGMENTATION
    /// host DIM-root of tensile damage
    real *h_d;
    /// host tensile damage + porous damage
    real *h_damage_total;
    /// host time derivative of DIM-root of (tensile) damage
    real *h_dddt;
    /// host total number of flaws
    integer *h_numFlaws;
    /// host maximum number of flaws allowed per particle
    integer *h_maxNumFlaws;
    /// host current number of activated flaws
    integer *h_numActiveFlaws;
    /// host values for the strain for each flaw
    real *h_flaws;
#if PALPHA_POROSITY
    /// host DIM-root of porous damage
    real *h_damage_porjutzi;
    /// time derivative of DIM-root of porous damage
    real *h_ddamage_porjutzidt;
#endif
#endif

    /// device number of particles
    integer *d_numParticles;
    /// device number of nodes
    integer *d_numNodes;

    // device particle entries
    /// device mass array
    real *d_mass;
    /// device x position
    real *d_x, *_d_x;
    /// device x velocity
    real *d_vx, *_d_vx;
    /// device x acceleration
    real *d_ax, *_d_ax;
    real *d_g_ax;
    real *d_ax_old, *d_g_ax_old; // only for leapfrog integrator
#if DIM > 1
    /// device y position
    real *d_y, *_d_y;
    /// device y velocity
    real *d_vy, *_d_vy;
    /// device y acceleration
    real *d_ay, *_d_ay;
    real *d_g_ay;
    real *d_ay_old, *d_g_ay_old; // only for leapfrog integrator
#if DIM == 3
    /// device z position
    real *d_z, *_d_z;
    /// device z velocity
    real *d_vz, *_d_vz;
    /// device z acceleration
    real *d_az, *_d_az;
    real *d_g_az;
    real *d_az_old, *d_g_az_old; // only for leapfrog integrator
#endif
#endif

    integer *d_nodeType;

    integer *d_level;
    /// device unique identifier
    idInteger *d_uid, *_d_uid; // unique identifier (unsigned int/long?)
    /// device material identifier
    integer *d_materialId; // material identfier (e.g.: ice, basalt, ...)
    /// device smoothing length
    real *d_sml, *_d_sml; // smoothing length
    /// device near(est) neighbor list
    integer *d_nnl; // max(number of interactions)
    /// device number of interaction
    integer *d_noi; // number of interactions (alternatively initialize nnl with -1, ...)
    /// device internal energy
    real *d_e, *_d_e; // internal energy
    /// device time derivative of internal energy
    real *d_dedt, *_d_dedt;
    /// energy
    real *d_u;
    /// device speed of sound
    real *d_cs, *_d_cs; // soundspeed
    /// device density
    real *d_rho, *_d_rho; // density
    /// device pressure
    real *d_p, *_d_p; // pressure
    /// device max(mu_ij)
    real *d_muijmax;

//#if INTEGRATE_DENSITY
    /// device time derivative of density
    real *d_drhodt, *_d_drhodt;
//#endif
#if VARIABLE_SML || INTEGRATE_SML
    /// device time derivaive of smoothing length
    real *d_dsmldt, *_d_dsmldt;
#endif
#if SML_CORRECTION
    real *d_sml_omega;
#endif
#if NAVIER_STOKES
    real *d_Tshear;
    real *d_eta;
#endif
#if SOLID
    /// device deviatoric stress tensor
    real *d_S;
    /// device time derivative of deviatoric stress tensor
    real *d_dSdt;
    /// device local strain
    real *d_localStrain;
#endif
#if SOLID || NAVIER_STOKES
    /// device sigma/stress tensor
    real *d_sigma;
#endif
#if ARTIFICIAL_STRESS
    /// device tensile instability
    real *d_R;
#endif
#if BALSARA_SWITCH
    real *d_divv;
    real *d_curlv;
#endif
#if POROSITY
    /// device pressure of the sph particle after the last successful timestep
    real *d_pold;
    /// device current distension of the sph particle
    real *d_alpha_jutzi;
    /// device distension of the sph particle after the last successful timestep
    real *d_alpha_jutzi_old;
    /// device time derivative of the distension
    real *d_dalphadt;
    /// device partial derivative of the distension with respect to the pressure
    real *d_dalphadp;
    /// device difference in pressure from the last timestep to the current one
    real *d_dp;
    /// device partial derivative of the distension with respect to the density
    real *d_dalphadrho;
    /// device additional factor to reduce the deviatoric stress tensor according to Jutzi
    real *d_f;
    /// device partial derivative of the pressure with respect to the density
    real *d_delpdelrho;
    /// device partial derivative of the pressure with respect to the specific internal energy
    real *d_delpdele;
    /// device sound speed after the last successful timestep
    real *d_cs_old;
    /// device distention in the strain-\alpha model
    real *d_alpha_epspor;
    /// device time derivative of the distension
    real *d_dalpha_epspordt;
    /// device volume change (trace of strain rate tensor)
    real *d_epsilon_v;
    /// device time derivative of volume change
    real *d_depsilon_vdt;
#endif
#if ZERO_CONSISTENCY
    /// device correction (value) for zeroth order consistency
    real *d_shepardCorrection;
#endif
#if LINEAR_CONSISTENCY
    /// device correction matrix for linear order consistency
    real *d_tensorialCorrectionMatrix;
#endif
#if FRAGMENTATION
    /// device DIM-root of tensile damage
    real *d_d;
    /// device tensile damage + porous damage
    real *d_damage_total;
    /// device time derivative of DIM-root of (tensile) damage
    real *d_dddt;
    /// device total number of flaws
    integer *d_numFlaws;
    /// device maximum number of flaws allowed per particle
    integer *d_maxNumFlaws;
    /// device current number of activated flaws
    integer *d_numActiveFlaws;
    /// device values for the strain for each flaw
    real *d_flaws;
#if PALPHA_POROSITY
    /// device DIM-root of porous damage
    real *d_damage_porjutzi;
    /// device time derivative of DIM-root of porous damage
    real *d_ddamage_porjutzidt;
#endif
#endif

    /// host instance of particles class
    Particles *h_particles;
    /// device instance of particles class
    Particles *d_particles;

    /**
     * Constructor
     *
     * @param numParticles
     * @param numNodes
     */
    ParticleHandler(integer numParticles, integer numNodes);

    void initLeapfrog();
    void freeLeapfrog();

    /**
     * Destructor
     */
    ~ParticleHandler();

    template <typename T>
    T*& getEntry(Entry::Name entry, Execution::Location location = Execution::device);

    void setPointer(IntegratedParticleHandler *integratedParticleHandler);

    void resetPointer();

    /**
     * copy particle's mass(es) to target (host/device)
     *
     * @param target host/device
     */
    void copyMass(To::Target target=To::device, bool includePseudoParticles = false);

    void copyUid(To::Target target=To::device);

    void copyMatId(To::Target target=To::device);

    void copySML(To::Target target=To::device);

    /**
     * copy particle's position(s) to target (host/device)
     *
     * @param target host/device
     */
    void copyPosition(To::Target target=To::device, bool includePseudoParticles = false);
    /**
    * copy particle's velocities to target (host/device)
    *
    * @param target host/device
    */
    void copyVelocity(To::Target target=To::device, bool includePseudoParticles = false);
    /**
     * copy particle's acceleration(s) to target (host/device)
     *
     * @param target host/device
     */
    void copyAcceleration(To::Target target=To::device, bool includePseudoParticles = false);
    /**
     * copy particle's information to target (host/device)
     *
     * @param target host/device
     * @param velocity flag whether velocities should be copied
     * @param acceleration flag whether accelerations should be copied
     */
    void copyDistribution(To::Target target=To::device, bool velocity=true, bool acceleration=true,
                          bool includePseudoParticles = false);

    void copySPH(To::Target target);

};

/**
 * Handling `IntegratedParticles` class
 *
 * Allocate and handle both host and device arrays and instances
 */
class IntegratedParticleHandler {

public:

    /// (host) number of particles
    integer numParticles;
    /// (host) number of nodes
    integer numNodes;

    /// device unique identifier
    idInteger *d_uid;

    real *d_x;
    /// device time derivative of particle's x position
    real *d_vx;
    /// device time derivative of particle's x velocity
    real *d_ax;
#if DIM > 1
    real *d_y;
    /// device time derivative of particle's y position
    real *d_vy;
    /// device time derivative of particle's y velocity
    real *d_ay;
#if DIM == 3
    real *d_z;
    /// device time derivative of particle's z position
    real *d_vz;
    /// device time derivative of particle's z velocity
    real *d_az;
#endif
#endif

    real *d_rho;
    real *d_e;
    real *d_dedt;
    real *d_p;
    real *d_cs;

    real *d_sml;

//#if INTEGRATE_DENSITY
    real *d_drhodt;
//#endif
#if PERIODIC_BOUNDARIES
    /// number of ghost particles
    integer d_numGhosts, h_numGhosts;
    /// using IntegratedParticles instance as container for ghost particles
    integer *d_ghostParticleIndices;
    void copyNumGhosts(To::Target target);
#endif
#if VARIABLE_SML || INTEGRATE_SML
    real *d_dsmldt;
#endif

    /// device instance of `IntegratedParticles` class
    IntegratedParticles *d_integratedParticles;

    /**
     * Constructor
     *
     * @param numParticles
     * @param numNodes
     */
    IntegratedParticleHandler(integer numParticles, integer numNodes);

    /**
     * Destructor
     */
    ~IntegratedParticleHandler();

};

#endif //MILUPHPC_PARTICLE_HANDLER_H
