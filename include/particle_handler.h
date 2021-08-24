//
// Created by Michael Staneker on 12.08.21.
//

#ifndef MILUPHPC_PARTICLE_HANDLER_H
#define MILUPHPC_PARTICLE_HANDLER_H

#include "parameter.h"
#include "particles.cuh"
#include "utils/logger.h"

class ParticleHandler {

public:

    integer numParticles;
    integer numNodes;

    real *h_mass;
    real *h_x, *h_vx, *h_ax;
#if DIM > 1
    real *h_y, *h_vy, *h_ay;
#if DIM == 3
    real *h_z, *h_vz, *h_az;
#endif
#endif

    idInteger *h_uid; // unique identifier (unsigned int/long?)
    integer *h_materialId; // material identfier (e.g.: ice, basalt, ...)
    real *h_sml; // smoothing length
    integer *h_nnl; // max(number of interactions)
    integer *h_noi; // number of interactions (alternatively initialize nnl with -1, ...)
    real *h_e; // internal energy
    real *h_dedt;
    real *h_cs; // soundspeed
    real *h_rho; // density
    real *h_p; // pressure

#if INTEGRATE_DENSITY
    real *h_drhodt;
#endif
#if VARIABLE_SML
    real *h_dsmldt;
#endif
#if SOLID
    real *h_S;
    real *h_dSdt;
    real *h_localStrain;
#endif
#if SOLID || NAVIER_STOKES
    real *h_sigma;
#endif
#if ARTIFICIAL_STRESS
    real *h_R;
#endif
#if POROSITY
    real *h_pold;
    real *h_alpha_jutzi;
    real *h_alpha_jutzi_old;
    real *h_dalphadt;
    real *h_dalphadp;
    real *h_dp;
    real *h_dalphadrho;
    real *h_f;
    real *h_delpdelrho;
    real *h_delpdele;
    real *h_cs_old;
    real *h_alpha_epspor;
    real *h_dalpha_epspordt;
    real *h_epsilon_v;
    real *h_depsilon_vdt;
#endif
#if ZERO_CONSISTENCY
    real *h_shepardCorrection;
#endif
#if LINEAR_CONSISTENCY
    real *h_tensorialCorrectionMatrix;
#endif
#if FRAGMENTATION
    real *h_d;
    real *h_damage_total;
    real *h_dddt;
    integer *h_numFlaws;
    integer *h_maxNumFlaws;
    integer *h_numActiveFlaws;
    real *h_flaws;
#if PALPHA_POROSITY
    real *h_damage_porjutzi;
    real *h_ddamage_porjutzidt;
#endif
#endif

    integer *d_numParticles;
    integer *d_numNodes;

    // device particle entries
    real *d_mass;
    real *d_x, *d_vx, *d_ax;
#if DIM > 1
    real *d_y, *d_vy, *d_ay;
#if DIM == 3
    real *d_z, *d_vz, *d_az;
#endif
#endif

    idInteger *d_uid; // unique identifier (unsigned int/long?)
    integer *d_materialId; // material identfier (e.g.: ice, basalt, ...)
    real *d_sml; // smoothing length
    integer *d_nnl; // max(number of interactions)
    integer *d_noi; // number of interactions (alternatively initialize nnl with -1, ...)
    real *d_e; // internal energy
    real *d_dedt;
    real *d_cs; // soundspeed
    real *d_rho; // density
    real *d_p; // pressure

#if INTEGRATE_DENSITY
    real *d_drhodt;
#endif
#if VARIABLE_SML
    real *d_dsmldt;
#endif
#if SOLID
    real *d_S;
    real *d_dSdt;
    real *d_localStrain;
#endif
#if SOLID || NAVIER_STOKES
    real *d_sigma;
#endif
#if ARTIFICIAL_STRESS
    real *d_R;
#endif
#if POROSITY
    real *d_pold;
    real *d_alpha_jutzi;
    real *d_alpha_jutzi_old;
    real *d_dalphadt;
    real *d_dalphadp;
    real *d_dp;
    real *d_dalphadrho;
    real *d_f;
    real *d_delpdelrho;
    real *d_delpdele;
    real *d_cs_old;
    real *d_alpha_epspor;
    real *d_dalpha_epspordt;
    real *d_epsilon_v;
    real *d_depsilon_vdt;
#endif
#if ZERO_CONSISTENCY
    real *d_shepardCorrection;
#endif
#if LINEAR_CONSISTENCY
    real *d_tensorialCorrectionMatrix;
#endif
#if FRAGMENTATION
    real *d_d;
    real *d_damage_total;
    real *d_dddt;
    integer *d_numFlaws;
    integer *d_maxNumFlaws;
    integer *d_numActiveFlaws;
    real *d_flaws;
#if PALPHA_POROSITY
    real *d_damage_porjutzi;
    real *d_ddamage_porjutzidt;
#endif
#endif

    Particles *h_particles;
    Particles *d_particles;

    ParticleHandler(integer numParticles, integer numNodes);
    ~ParticleHandler();

    void positionToDevice();
    void velocityToDevice();
    void accelerationToDevice();
    void distributionToDevice(bool velocity=true, bool acceleration=true);
    void positionToHost();
    void velocityToHost();
    void accelerationToHost();
    void distributionToHost(bool velocity=true, bool acceleration=true);

};

class IntegratedParticleHandler {

public:

    integer numParticles;
    integer numNodes;

    integer *d_uid;
    real *d_drhodt;

    real *d_dxdt, *d_dvxdt;
#if DIM > 1
    real *d_dydt, *d_dvydt;
#if DIM == 3
    real *d_dzdt, *d_dvzdt;
#endif
#endif

    IntegratedParticles *d_integratedParticles;

    IntegratedParticleHandler(integer numParticles, integer numNodes);
    ~IntegratedParticleHandler();

};

#endif //MILUPHPC_PARTICLE_HANDLER_H
