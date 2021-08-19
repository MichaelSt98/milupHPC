//
// Created by Michael Staneker on 12.08.21.
//

#ifndef MILUPHPC_PARTICLES_CUH
#define MILUPHPC_PARTICLES_CUH

#include "cuda_utils/cuda_utilities.cuh"
//#include "../cuda_utils/cuda_launcher.cuh"
#include "parameter.h"

class IntegratedParticles {

public:
    real *drhodt;

    real *dxdt;
    real *dydt;
    real *dzdt;

    real *dvxdt;
    real *dvydt;
    real *dvzdt;

    //...
    //TODO: implement and add Handler class
};

class Particles {

public:

    integer *numParticles;
    integer *numNodes;

    real *mass;
    real *x, *vx, *ax;
#if DIM > 1
    real *y, *vy, *ay;
#if DIM == 3
    real *z, *vz, *az;
#endif
#endif

    /*integer *uid; // unique identifier (unsigned int/long?)
    integer *materialId; // material identfier (e.g.: ice, basalt, ...)
    real *sml; // smoothing length

    integer *nnl; // max(number of interactions)
    integer *noi; // number of interactions (alternatively initialize nnl with -1, ...)

    real *e; // internal energy
    real *dedt;

    real *cs; // soundspeed

    // simplest hydro
    real *rho; // density
    real *p; // pressure

#if INTEGRATE_DENSITY
    // integrated density
    real *drhodt;
#endif

#if VARIABLE_SML
    // integrate/variable smoothing length
    real *dsmldt;
#endif

#if SOLID
    real *S; // deviatoric stress tensor (DIM * DIM)
    real *dSdt;
    real *localStrain; // local strain
#endif
#if SOLID || NAVIER_STOKES
    real *sigma; // stress tensor (DIM * DIM)
#endif

#if ARTIFICIAL_STRESS
    real *R; // tensile instability, tensor for correction (DIM * DIM)
#endif

#if POROSITY;
//#if PALPHA_POROSITY
    real *pold; // the pressure of the sph particle after the last successful timestep
    real *alpha_jutzi; // the current distension of the sph particle
    real *alpha_jutzi_old; // the distension of the sph particle after the last successful timestep
    real *dalphadt; // the time derivative of the distension
    real *dalphadp; // the partial derivative of the distension with respect to the pressure
    real *dp; // the difference in pressure from the last timestep to the current one
    real *dalphadrho; // the partial derivative of the distension with respect to the density
    real *f; // additional factor to reduce the deviatoric stress tensor according to Jutzi
    real *delpdelrho; // the partial derivative of the pressure with respect to the density
    real *delpdele; // the partial derivative of the pressure with respect to the specific internal energy
	real *cs_old; // the sound speed after the last successful timestep
//#endif
//#if EPSALPHA_POROSITY
    real *alpha_epspor; // distention in the strain-\alpha model
    real *dalpha_epspordt; // time derivative of the distension
    real *epsilon_v; //  volume change (trace of strain rate tensor)
    real *depsilon_vdt; // time derivative of volume change
//#endif
#endif

#if ZERO_CONSISTENCY
    real *shepardCorrection; // correction (value) for zeroth order consistency
#endif

#if LINEAR_CONSISTENCY
    real *tensorialCorrectionMatrix; // correction matrix for linear order consistency (DIM*DIM)
#endif


#if FRAGMENTATION
    real *d; // DIM-root of tensile damage
    real *damage_total; // tensile damage + porous damage (directly, not DIM-root)
    real *dddt; // the time derivative of DIM-root of (tensile) damage
    integer *numFlaws; // the total number of flaws
    integer maxNumFlaws; // the maximum number of flaws allowed per particle
    integer *numActiveFlaws; // the current number of activated flaws
    real *flaws; // the values for the strain for each flaw (array of size maxNumFlaws)
#if PALPHA_POROSITY
    real *damage_porjutzi; // DIM-root of porous damage
    real *ddamage_porjutzidt; // time derivative of DIM-root of porous damage
#endif
#endif*/


    CUDA_CALLABLE_MEMBER Particles();

    //TODO: wouldn't be necessary but better for compilation?
    CUDA_CALLABLE_MEMBER Particles(integer *numParticles, integer *numNodes, real *mass, real *x, real *vx, real *ax);

    CUDA_CALLABLE_MEMBER void set(integer *numParticles, integer *numNodes, real *mass,
                                          real *x, real *vx, real *ax);
#if DIM > 1
    CUDA_CALLABLE_MEMBER Particles(integer *numParticles, integer *numNodes, real *mass, real *x, real *y, real *vx, real *vy, real *ax, real *ay);

    CUDA_CALLABLE_MEMBER void set(integer *numParticles, integer *numNodes, real *mass, real *x, real *y, real *vx, real *vy,
                                          real *ax, real *ay);
#if DIM == 3
    CUDA_CALLABLE_MEMBER Particles(integer *numParticles, integer *numNodes, real *mass, real *x, real *y, real *z, real *vx,
                                   real *vy, real *vz, real *ax, real *ay, real *az);

    CUDA_CALLABLE_MEMBER void set(integer *numParticles, integer *numNodes, real *mass, real *x, real *y, real *z, real *vx,
                                          real *vy, real *vz, real *ax, real *ay, real *az);
#endif
#endif

    CUDA_CALLABLE_MEMBER void reset(integer index);

    CUDA_CALLABLE_MEMBER ~Particles();

};

namespace ParticlesNS {

    namespace Kernel {

        __global__ void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x,
                            real *vx, real *ax);

        namespace Launch {
            void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x, real *vx,
                      real *ax);
        }

#if DIM > 1

        __global__ void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x,
                            real *y, real *vx, real *vy, real *ax, real *ay);

        namespace Launch {
            void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x, real *y,
                     real *vx, real *vy, real *ax, real *ay);
        }

#if DIM == 3

        __global__ void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x,
                            real *y, real *z, real *vx, real *vy, real *vz, real *ax, real *ay, real *az);

        namespace Launch {
            void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x, real *y,
                     real *z, real *vx, real *vy, real *vz, real *ax, real *ay, real *az);
        }

#endif
#endif

        __global__ void test(Particles *particles);

        namespace Launch {
            real test(Particles *particles, bool time=false);
        }
    }

}

#endif //MILUPHPC_PARTICLES_CUH
