//
// Created by Michael Staneker on 12.08.21.
//

#ifndef MILUPHPC_PARTICLES_CUH
#define MILUPHPC_PARTICLES_CUH

#include "cuda_utils/cuda_utilities.cuh"
//#include "../cuda_utils/cuda_launcher.cuh"
#include "parameter.h"

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

    idInteger *uid; // unique identifier (unsigned int/long?)
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

#if POROSITY
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
    integer *maxNumFlaws; // the maximum number of flaws allowed per particle
    integer *numActiveFlaws; // the current number of activated flaws
    real *flaws; // the values for the strain for each flaw (array of size maxNumFlaws)
#if PALPHA_POROSITY
    real *damage_porjutzi; // DIM-root of porous damage
    real *ddamage_porjutzidt; // time derivative of DIM-root of porous damage
#endif
#endif


    CUDA_CALLABLE_MEMBER Particles();

    CUDA_CALLABLE_MEMBER Particles(integer *numParticles, integer *numNodes, real *mass, real *x, real *vx, real *ax,
                                   idInteger *uid, integer *materialId, real *sml, integer *nnl, integer *noi, real *e,
                                   real *dedt, real *cs, real *rho, real *p);

    CUDA_CALLABLE_MEMBER void set(integer *numParticles, integer *numNodes, real *mass, real *x, real *vx, real *ax,
                                  idInteger *uid, integer *materialId, real *sml, integer *nnl, integer *noi, real *e,
                                  real *dedt, real *cs, real *rho, real *p);
#if DIM > 1
    CUDA_CALLABLE_MEMBER Particles(integer *numParticles, integer *numNodes, real *mass, real *x, real *y, real *vx,
                                   real *vy, real *ax, real *ay, idInteger *uid, integer *materialId, real *sml,
                                   integer *nnl, integer *noi, real *e, real *dedt, real *cs, real *rho, real *p);

    CUDA_CALLABLE_MEMBER void set(integer *numParticles, integer *numNodes, real *mass, real *x, real *y, real *vx,
                                  real *vy, real *ax, real *ay, idInteger *uid, integer *materialId, real *sml,
                                  integer *nnl, integer *noi, real *e, real *dedt, real *cs, real *rho, real *p);
#if DIM == 3
    CUDA_CALLABLE_MEMBER Particles(integer *numParticles, integer *numNodes, real *mass, real *x, real *y, real *z,
                                   real *vx, real *vy, real *vz, real *ax, real *ay, real *az, idInteger *uid,
                                   integer *materialId, real *sml, integer *nnl, integer *noi, real *e, real *dedt,
                                   real *cs, real *rho, real *p);

    CUDA_CALLABLE_MEMBER void set(integer *numParticles, integer *numNodes, real *mass, real *x, real *y, real *z,
                                  real *vx, real *vy, real *vz, real *ax, real *ay, real *az, idInteger *uid,
                                  integer *materialId, real *sml, integer *nnl, integer *noi, real *e, real *dedt,
                                  real *cs, real *rho, real *p);
#endif
#endif

#if INTEGRATE_DENSITY
    CUDA_CALLABLE_MEMBER void setIntegrateDensity(real *drhodt);
#endif
#if VARIABLE_SML
    CUDA_CALLABLE_MEMBER void setVariableSML(real *dsmldt);
#endif
#if SOLID
    CUDA_CALLABLE_MEMBER void setSolid(real *S, real *dSdt, real *localStrain);
#endif
#if SOLID || NAVIER_STOKES
    CUDA_CALLABLE_MEMBER void setNavierStokes(real *sigma);
#endif
#if ARTIFICIAL_STRESS
    CUDA_CALLABLE_MEMBER void setArtificialStress(real *R);
#endif
#if POROSITY
    CUDA_CALLABLE_MEMBER void setPorosity(real *pold, real *alpha_jutzi, real *alpha_jutzi_old, real *dalphadt,
                                          real *dalphadp, real *dp, real *dalphadrho, real *f, real *delpdelrho,
                                          real *delpdele, real *cs_old, real *alpha_epspor, real *dalpha_epspordt,
                                          real *epsilon_v, real *depsilon_vdt);
#endif
#if ZERO_CONSISTENCY
    CUDA_CALLABLE_MEMBER void setZeroConsistency(real *shepardCorrection);
#endif
#if LINEAR_CONSISTENCY
    CUDA_CALLABLE_MEMBER void setLinearConsistency(real *tensorialCorrectionMatrix);
#endif
#if FRAGMENTATION
    CUDA_CALLABLE_MEMBER void setFragmentation(real *d, real *damage_total, real *dddt, integer *numFlaws,
                                               integer *maxNumFlaws, integer *numActiveFlaws, real *flaws);
#if PALPHA_POROSITY
    CUDA_CALLABLE_MEMBER void setPalphaPorosity(real *damage_porjutzi, real *ddamage_porjutzidt);
#endif
#endif

    CUDA_CALLABLE_MEMBER void reset(integer index);

    CUDA_CALLABLE_MEMBER real distance(integer index_1, integer index_2);

    CUDA_CALLABLE_MEMBER ~Particles();

};

namespace ParticlesNS {

    namespace Kernel {

        __global__ void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x,
                            real *vx, real *ax, idInteger *uid, integer *materialId, real *sml, integer *nnl,
                            integer *noi, real *e, real *dedt, real *cs, real *rho, real *p);

        namespace Launch {
            void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x, real *vx,
                     real *ax, idInteger *uid, integer *materialId, real *sml, integer *nnl, integer *noi, real *e,
                     real *dedt, real *cs, real *rho, real *p);
        }

#if DIM > 1

        __global__ void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x,
                            real *y, real *vx, real *vy, real *ax, real *ay, idInteger *uid, integer *materialId,
                            real *sml, integer *nnl, integer *noi, real *e, real *dedt, real *cs, real *rho, real *p);

        namespace Launch {
            void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x, real *y,
                     real *vx, real *vy, real *ax, real *ay, idInteger *id, integer *materialId, real *sml, integer *nnl,
                     integer *noi, real *e, real *dedt, real *cs, real *rho, real *p);
        }

#if DIM == 3

        __global__ void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x,
                            real *y, real *z, real *vx, real *vy, real *vz, real *ax, real *ay, real *az,
                            idInteger *uid, integer *materialId, real *sml, integer *nnl, integer *noi, real *e,
                            real *dedt, real *cs, real *rho, real *p);

        namespace Launch {
            void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x, real *y,
                     real *z, real *vx, real *vy, real *vz, real *ax, real *ay, real *az, idInteger *id,
                     integer *materialId, real *sml, integer *nnl, integer *noi, real *e, real *dedt, real *cs,
                     real *rho, real *p);
        }

#endif
#endif

#if INTEGRATE_DENSITY
        __global__ void setIntegrateDensity(Particles *particles, real *drhodt);
        namespace Launch {
            void setIntegrateDensity(Particles *particles, real *drhodt);
        }
#endif
#if VARIABLE_SML
        __global__ void setVariableSML(Particles *particles, real *dsmldt);
        namespace Launch {
            void setVariableSML(Particles *particles, real *dsmldt);
        }
#endif
#if SOLID
        __global__ void setSolid(Particles *particles, real *S, real *dSdt, real *localStrain);
        namespace Launch {
            void setSolid(Particles *particles, real *S, real *dSdt, real *localStrain);
        }
#endif
#if SOLID || NAVIER_STOKES
        __global__ void setNavierStokes(Particles *particles, real *sigma);
        namespace Launch {
            void setNavierStokes(Particles *particles, real *sigma);
        }
#endif
#if ARTIFICIAL_STRESS
        __global__ void setArtificialStress(Particles *particles, real *R);
        namespace Launch {
            void setArtificialStress(Particles *particles, real *R);
        }
#endif
#if POROSITY
        __global__ void setPorosity(Particles *particles, real *pold, real *alpha_jutzi, real *alpha_jutzi_old, real *dalphadt,
                                    real *dalphadp, real *dp, real *dalphadrho, real *f, real *delpdelrho,
                                    real *delpdele, real *cs_old, real *alpha_epspor, real *dalpha_epspordt,
                                    real *epsilon_v, real *depsilon_vdt);
        namespace Launch {
            void setPorosity(Particles *particles, real *pold, real *alpha_jutzi, real *alpha_jutzi_old, real *dalphadt,
                             real *dalphadp, real *dp, real *dalphadrho, real *f, real *delpdelrho,
                             real *delpdele, real *cs_old, real *alpha_epspor, real *dalpha_epspordt,
                             real *epsilon_v, real *depsilon_vdt);
        }
#endif
#if ZERO_CONSISTENCY
        __global__ void setZeroConsistency(Particles *particles, real *shepardCorrection);
        namespace Launch {
            void setZeroConsistency(Particles *particles, real *shepardCorrection);
        }
#endif
#if LINEAR_CONSISTENCY
        __global__ void setLinearConsistency(Particles *particles, real *tensorialCorrectionMatrix);
        namespace Launch {
            void setLinearConsistency(Particles *particles, real *tensorialCorrectionMatrix);
        }
#endif
#if FRAGMENTATION
        __global__ void setFragmentation(Particles *particles, real *d, real *damage_total, real *dddt, integer *numFlaws,
                                         integer *maxNumFlaws, integer *numActiveFlaws, real *flaws);
        namespace Launch {
            void setFragmentation(Particles *particles, real *d, real *damage_total, real *dddt, integer *numFlaws,
                                  integer *maxNumFlaws, integer *numActiveFlaws, real *flaws);
        }
#if PALPHA_POROSITY
        __global__ void setPalphaPorosity(Particles *particles, real *damage_porjutzi, real *ddamage_porjutzidt);
        namespace Launch {
            void setPalphaPorosity(Particles *particles, real *damage_porjutzi, real *ddamage_porjutzidt);
        }
#endif
#endif

        __global__ void test(Particles *particles);

        namespace Launch {
            real test(Particles *particles, bool time=false);
        }
    }

}

class IntegratedParticles {

public:

    integer *uid;

    real *drhodt;

    real *dxdt, *dvxdt;
#if DIM > 1
    real *dydt, *dvydt;
#if DIM == 3
    real *dzdt, *dvzdt;
#endif
#endif

    CUDA_CALLABLE_MEMBER IntegratedParticles();

    CUDA_CALLABLE_MEMBER IntegratedParticles(integer *uid, real *drhodt, real *dxdt, real *dvxdt);

    CUDA_CALLABLE_MEMBER void set(integer *uid, real *drhodt, real *dxdt, real *dvxdt);

#if DIM > 1

    CUDA_CALLABLE_MEMBER IntegratedParticles(integer *uid, real *drhodt, real *dxdt, real *dydt, real *dvxdt,
                                             real *dvydt);

    CUDA_CALLABLE_MEMBER void set(integer *uid, real *drhodt, real *dxdt, real *dydt, real *dvxdt,
                                  real *dvydt);

#if DIM == 3

    CUDA_CALLABLE_MEMBER IntegratedParticles(integer *uid, real *drhodt, real *dxdt, real *dydt, real *dzdt,
                                             real *dvxdt, real *dvydt, real *dvzdt);

    CUDA_CALLABLE_MEMBER void set(integer *uid, real *drhodt, real *dxdt, real *dydt, real *dzdt,
                                  real *dvxdt, real *dvydt, real *dvzdt);

#endif
#endif

    CUDA_CALLABLE_MEMBER void reset(integer index);

    CUDA_CALLABLE_MEMBER ~IntegratedParticles();

};

namespace IntegratedParticlesNS {

    namespace Kernel {

        __global__ void set(IntegratedParticles *integratedParticles, integer *uid, real *drhodt, real *dxdt,
                            real *dvxdt);

        namespace Launch {
            void set(IntegratedParticles *integratedParticles, integer *uid, real *drhodt, real *dxdt,
                     real *dvxdt);
        }

#if DIM > 1

        __global__ void set(IntegratedParticles *integratedParticles, integer *uid, real *drhodt, real *dxdt,
                            real *dydt, real *dvxdt, real *dvydt);

        namespace Launch {
            void set(IntegratedParticles *integratedParticles, integer *uid, real *drhodt, real *dxdt,
                     real *dydt, real *dvxdt, real *dvydt);
        }

#if DIM == 3

        __global__ void set(IntegratedParticles *integratedParticles, integer *uid, real *drhodt, real *dxdt,
                            real *dydt, real *dzdt, real *dvxdt, real *dvydt, real *dvzdt);

        namespace Launch {
            void set(IntegratedParticles *integratedParticles, integer *uid, real *drhodt, real *dxdt,
                     real *dydt, real *dzdt, real *dvxdt, real *dvydt, real *dvzdt);
        }

#endif
#endif

    }
}

#endif //MILUPHPC_PARTICLES_CUH
