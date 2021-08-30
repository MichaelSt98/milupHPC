//
// Created by Michael Staneker on 12.08.21.
//

#include "../include/particles.cuh"
#include "../include/cuda_utils/cuda_launcher.cuh"

CUDA_CALLABLE_MEMBER Particles::Particles() {

}

CUDA_CALLABLE_MEMBER Particles::Particles(integer *numParticles, integer *numNodes, real *mass, real *x, real *vx,
                                          real *ax, idInteger *uid, integer *materialId, real *sml, integer *nnl,
                                          integer *noi, real *e, real *dedt, real *cs, real *rho, real *p) :
                                          numParticles(numParticles), numNodes(numNodes), mass(mass), x(x), vx(vx),
                                          ax(ax), uid(uid), materialId(materialId), sml(sml), nnl(nnl), noi(noi),
                                          e(e), dedt(dedt), cs(cs), rho(rho), p(p) {

}

CUDA_CALLABLE_MEMBER void Particles::set(integer *numParticles, integer *numNodes, real *mass, real *x, real *vx, real *ax,
                                         idInteger *uid, integer *materialId,
                                         real *sml, integer *nnl, integer *noi, real *e, real *dedt,
                                         real *cs, real *rho, real *p) {

    this->numParticles = numParticles;
    this->numNodes = numNodes;
    this->mass = mass;
    this->x = x;
    this->vx = vx;
    this->ax = ax;
    this->uid = uid;
    this->materialId = materialId;
    this->sml = sml;
    this->nnl = nnl;
    this->noi = noi;
    this->e = e;
    this->dedt = dedt;
    this->cs = cs;
    this->rho = rho;
    this->p = p;

}

#if DIM > 1
CUDA_CALLABLE_MEMBER Particles::Particles(integer *numParticles, integer *numNodes, real *mass, real *x, real *y,
                                          real *vx, real *vy, real *ax, real *ay, idInteger *uid, integer *materialId,
                                          real *sml, integer *nnl, integer *noi, real *e, real *dedt, real *cs,
                                          real *rho, real *p) : numParticles(numParticles), numNodes(numNodes),
                                          mass(mass), x(x), y(y), vx(vx), vy(vy), ax(ax), ay(ay), uid(uid),
                                          materialId(materialId), sml(sml), nnl(nnl), noi(noi), e(e), dedt(dedt),
                                          cs(cs), rho(rho), p(p) {

}

CUDA_CALLABLE_MEMBER void Particles::set(integer *numParticles, integer *numNodes, real *mass, real *x, real *y,
                                         real *vx, real *vy, real *ax, real *ay, idInteger *uid, integer *materialId,
                                         real *sml, integer *nnl, integer *noi, real *e, real *dedt,
                                         real *cs, real *rho, real *p) {

    this->numParticles = numParticles;
    this->numNodes = numNodes;
    this->mass = mass;
    this->x = x;
    this->y = y;
    this->vx = vx;
    this->vy = vy;
    this->ax = ax;
    this->ay = ay;
    this->uid = uid;
    this->materialId = materialId;
    this->sml = sml;
    this->nnl = nnl;
    this->noi = noi;
    this->e = e;
    this->dedt = dedt;
    this->cs = cs;
    this->rho = rho;
    this->p = p;
}
#if DIM == 3
CUDA_CALLABLE_MEMBER Particles::Particles(integer *numParticles, integer *numNodes, real *mass, real *x, real *y,
                                          real *z, real *vx, real *vy, real *vz, real *ax, real *ay, real *az,
                                          idInteger *uid, integer *materialId, real *sml, integer *nnl, integer *noi,
                                          real *e, real *dedt, real *cs, real *rho, real *p) :
                                          numParticles(numParticles), numNodes(numNodes), mass(mass), x(x), y(y), z(z),
                                          vx(vx), vy(vy), vz(vz), ax(ax), ay(ay), az(az), uid(uid),
                                          materialId(materialId), sml(sml), nnl(nnl), noi(noi), e(e), dedt(dedt),
                                          cs(cs), rho(rho), p(p) {

}

CUDA_CALLABLE_MEMBER void Particles::set(integer *numParticles, integer *numNodes, real *mass, real *x, real *y,
                                         real *z, real *vx, real *vy, real *vz, real *ax, real *ay, real *az,
                                         idInteger *uid, integer *materialId, real *sml, integer *nnl, integer *noi,
                                         real *e, real *dedt, real *cs, real *rho, real *p) {

    this->numParticles = numParticles;
    this->numNodes = numNodes;
    this->mass = mass;
    this->x = x;
    this->y = y;
    this->z = z;
    this->vx = vx;
    this->vy = vy;
    this->vz = vz;
    this->ax = ax;
    this->ay = ay;
    this->az = az;
    this->uid = uid;
    this->materialId = materialId;
    this->sml = sml;
    this->nnl = nnl;
    this->noi = noi;
    this->e = e;
    this->dedt = dedt;
    this->cs = cs;
    this->rho = rho;
    this->p = p;
}
#endif
#endif

#if INTEGRATE_DENSITY
    CUDA_CALLABLE_MEMBER void Particles::setIntegrateDensity(real *drhodt) {
        this->drhodt = drhodt;
    }
#endif
#if VARIABLE_SML
    CUDA_CALLABLE_MEMBER void Particles::setVariableSML(real *dsmldt) {
        this->dsmldt = dsmldt;
    }
#endif
#if SOLID
    CUDA_CALLABLE_MEMBER void Particles::setSolid(real *S, real *dSdt, real *localStrain) {
        this->S = S;
        this->dSdt = dSdt;
        this->localStrain = localStrain;
    }
#endif
#if SOLID || NAVIER_STOKES
    CUDA_CALLABLE_MEMBER void Particles::setNavierStokes(real *sigma) {
        this->sigma = sigma;
    }
#endif
#if ARTIFICIAL_STRESS
    CUDA_CALLABLE_MEMBER void Particles::setArtificialStress(real *R) {
        this->R = R;
    }
#endif
#if POROSITY
CUDA_CALLABLE_MEMBER void Particles::setPorosity(real *pold, real *alpha_jutzi, real *alpha_jutzi_old, real *dalphadt,
                                      real *dalphadp, real *dp, real *dalphadrho, real *f, real *delpdelrho,
                                      real *delpdele, real *cs_old, real *alpha_epspor, real *dalpha_epspordt,
                                      real *epsilon_v, real *depsilon_vdt) {
        this->pold = pold;
        this->alpha_jutzi = alpha_jutzi;
        this->alpha_jutzi_old = alpha_jutzi_old;
        this->dalphadt = dalphadt;
        this->dalphadp = dalphadp;
        this->dp = dp;
        this->dalphadrho = dalphadrho;
        this->f = f;
        this->delpdelrho = delpdelrho;
        this->delpdele = delpdele;
        this->cs_old = cs_old;
        this->alpha_epspor = alpha_epspor;
        this->dalpha_epspordt = dalpha_epspordt;
        this->epsilon_v = epsilon_v;
        this->depsilon_vdt = depsilon_vdt;
    }
#endif
#if ZERO_CONSISTENCY
    CUDA_CALLABLE_MEMBER void Particles::setZeroConsistency(real *shepardCorrection) {
        this->shepardCorrection = shepardCorrection;
    }
#endif
#if LINEAR_CONSISTENCY
    CUDA_CALLABLE_MEMBER void Particles::setLinearConsistency(real *tensorialCorrectionMatrix) {
        this->tensorialCorrectionMatrix = tensorialCorrectionMatrix;
    }
#endif
#if FRAGMENTATION
    CUDA_CALLABLE_MEMBER void Particles::setFragmentation(real *d, real *damage_total, real *dddt, integer *numFlaws,
                                           integer *maxNumFlaws, integer *numActiveFlaws, real *flaws) {
        this->d = d;
        this->damage_total = damage_total;
        this->dddt = dddt;
        this->numFlaws = numFlaws;
        this->maxNumFlaws = maxNumFlaws;
        this->numActiveFlaws = numActiveFlaws;
        this->flaws = flaws;
    }
#if PALPHA_POROSITY
    CUDA_CALLABLE_MEMBER void Particles::setPalphaPorosity(real *damage_porjutzi, real *ddamage_porjutzidt) {
        this->damage_porjutzi = damage_porjutzi;
        this->ddamage_porjutzidt = ddamage_porjutzidt;
    }
#endif
#endif

CUDA_CALLABLE_MEMBER void Particles::reset(integer index) {
    x[index] = 0;
#if DIM > 1
    y[index] = 0;
#if DIM == 3
    z[index] = 0;
#endif
#endif
    mass[index] = 0;
}

CUDA_CALLABLE_MEMBER real Particles::distance(integer index_1, integer index_2) {
    
    float dx;
    if (x[index_1] < x[index_2]) {
        dx = x[index_2] - x[index_1];
    }
    else if (x[index_1] > x[index_2]) {
        dx = x[index_1] - x[index_2];
    }
    else {
        dx = 0.f;
    }
#if DIM > 1
    float dy;
    if (y[index_1] < y[index_2]) {
        dy = y[index_2] - y[index_1];
    }
    else if (y[index_1] > y[index_2]) {
        dy = y[index_1] - y[index_2];
    }
    else {
        dy = 0.f;
    }
#if DIM == 3
    float dz;
    if (z[index_1] < z[index_2]) {
        dz = z[index_2] - z[index_1];
    }
    else if (z[index_1] > z[index_2]) {
        dz = z[index_1] - z[index_2];
    }
    else {
        dz = 0.f;
    }
#endif
#endif

#if DIM == 1
    return sqrtf(dx*dx);
#elif DIM == 2
    return sqrtf(dx*dx + dy*dy);
#else
    return sqrtf(dx*dx + dy*dy + dz*dz);
#endif

}

CUDA_CALLABLE_MEMBER Particles::~Particles() {

}


namespace ParticlesNS {

    namespace Kernel {

        __global__ void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x,
                            real *vx, real *ax, idInteger *uid, integer *materialId, real *sml, integer *nnl,
                            integer *noi, real *e, real *dedt, real *cs, real *rho, real *p) {

            particles->set(numParticles, numNodes, mass, x, vx, ax, uid, materialId, sml, nnl, noi, e, dedt, cs,
                           rho, p);

        }

        void Launch::set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x, real *vx,
                         real *ax, idInteger *uid, integer *materialId, real *sml, integer *nnl, integer *noi, real *e,
                         real *dedt, real *cs, real *rho, real *p) {

            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::set, particles, numParticles, numNodes, mass,
                         x, vx, ax, uid, materialId, sml, nnl, noi, e, dedt, cs, rho, p);

        }

        __global__ void info(Particles *particles, integer n, integer m, integer k) {
            int bodyIndex = threadIdx.x + blockDim.x * blockIdx.x;
            int stride = blockDim.x * gridDim.x;
            int offset = 0;

            while ((bodyIndex + offset) < n) {
                if ((bodyIndex + offset) % 100 == 0) {
                    printf("x[%i] = (%f, %f, %f) mass = %f\n", bodyIndex + offset, particles->x[bodyIndex + offset],
                           particles->y[bodyIndex + offset],
                           particles->z[bodyIndex + offset],
                           particles->mass[bodyIndex + offset]);
                }
                offset += stride;
            }

            offset = m;
            while ((bodyIndex + offset) < k && (bodyIndex + offset) > m) {
                if ((bodyIndex + offset) % 100 == 0) {
                    printf("x[%i] = (%f, %f, %f) mass = %f\n", bodyIndex + offset, particles->x[bodyIndex + offset],
                           particles->y[bodyIndex + offset],
                           particles->z[bodyIndex + offset],
                           particles->mass[bodyIndex + offset]);
                }
                offset += stride;
            }

        }

        real Launch::info(Particles *particles, integer n, integer m, integer k) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::ParticlesNS::Kernel::info, particles, n, m, k);
        }


#if DIM > 1

        __global__ void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x,
                            real *y, real *vx, real *vy, real *ax, real *ay, idInteger *uid, integer *materialId,
                            real *sml, integer *nnl, integer *noi, real *e, real *dedt,
                            real *cs, real *rho, real *p) {

            particles->set(numParticles, numNodes, mass, x, y, vx, vy, ax, ay, uid, materialId, sml, nnl, noi, e,
                           dedt, cs, rho, p);

        }

        void Launch::set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x, real *y,
                         real *vx, real *vy, real *ax, real *ay, idInteger *uid, integer *materialId, real *sml,
                         integer *nnl, integer *noi, real *e, real *dedt, real *cs, real *rho, real *p) {

            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::set, particles, numParticles, numNodes,
                         mass, x, y, vx, vy, ax, ay, uid, materialId, sml, nnl, noi, e, dedt, cs, rho, p);

        }


#if DIM == 3

        __global__ void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x,
                            real *y, real *z, real *vx, real *vy, real *vz, real *ax, real *ay, real *az,
                            idInteger *uid, integer *materialId, real *sml, integer *nnl, integer *noi, real *e,
                            real *dedt, real *cs, real *rho, real *p) {

            particles->set(numParticles, numNodes, mass, x, y, z, vx, vy, vz, ax, ay, az, uid, materialId, sml, nnl,
                           noi, e, dedt, cs, rho, p);

        }

        void Launch::set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x, real *y,
                         real *z, real *vx, real *vy, real *vz, real *ax, real *ay, real *az, idInteger *uid,
                         integer *materialId, real *sml, integer *nnl, integer *noi, real *e, real *dedt, real *cs,
                         real *rho, real *p) {

            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::set, particles, numParticles, numNodes,
                         mass, x, y, z, vx, vy, vz, ax, ay, az, uid, materialId, sml, nnl, noi, e, dedt, cs, rho, p);
            //setKernel<<<1, 1>>>(particles, count, mass, x, y, z, vx, vy, vz, ax, ay, az);

        }

#endif
#endif

#if INTEGRATE_DENSITY
        __global__ void setIntegrateDensity(Particles *particles, real *drhodt) {
            particles->setIntegrateDensity(drhodt);
        }
        void Launch::setIntegrateDensity(Particles *particles, real *drhodt) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::setIntegrateDensity, particles, drhodt);
        }
#endif
#if VARIABLE_SML
        __global__ void setVariableSML(Particles *particles, real *dsmldt) {
            particles->setVariableSML(dsmldt);
        }
        void Launch::setVariableSML(Particles *particles, real *dsmldt) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::setVariableSML, particles, dsmldt);
        }
#endif
#if SOLID
        __global__ void setSolid(Particles *particles, real *S, real *dSdt, real *localStrain) {
            particles->setSolid(S, dSdt, localStrain);
        }
        void Launch::setSolid(Particles *particles, real *S, real *dSdt, real *localStrain) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::setSolid, particles, S, dSdt, localStrain);
        }
#endif
#if SOLID || NAVIER_STOKES
        __global__ void setNavierStokes(Particles *particles, real *sigma) {
            particles->setNavierStokes(sigma);
        }
        void Launch::setNavierStokes(Particles *particles, real *sigma) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::setNavierStokes, particles, sigma);
        }
#endif
#if ARTIFICIAL_STRESS
        __global__ void setArtificialStress(Particles *particles, real *R) {
            particles->setArtificialStress(R);
        }
        void Launch::setArtificialStress(Particles *particles, real *R) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::setArtificialStress, particles, R);
        }
#endif
#if POROSITY
        __global__ void setPorosity(Particles *particles, real *pold, real *alpha_jutzi, real *alpha_jutzi_old, real *dalphadt,
                                    real *dalphadp, real *dp, real *dalphadrho, real *f, real *delpdelrho,
                                    real *delpdele, real *cs_old, real *alpha_epspor, real *dalpha_epspordt,
                                    real *epsilon_v, real *depsilon_vdt) {
            particles->setPorosity(pold, alpha_jutzi, alpha_jutzi_old, dalphadt,
                                   dalphadp, dp, dalphadrho, f, delpdelrho,
                                   delpdele, cs_old, alpha_epspor, dalpha_epspordt,
                                   epsilon_v, depsilon_vdt);
        }
        void Launch::setPorosity(Particles *particles, real *pold, real *alpha_jutzi, real *alpha_jutzi_old, real *dalphadt,
                                 real *dalphadp, real *dp, real *dalphadrho, real *f, real *delpdelrho,
                                 real *delpdele, real *cs_old, real *alpha_epspor, real *dalpha_epspordt,
                                 real *epsilon_v, real *depsilon_vdt) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::setPorosity, particles, pold, alpha_jutzi,
                         alpha_jutzi_old, dalphadt, dalphadp, dp, dalphadrho, f, delpdelrho, delpdele, cs_old,
                         alpha_epspor, dalpha_epspordt, epsilon_v, depsilon_vdt);
        }

#endif
#if ZERO_CONSISTENCY
        __global__ void setZeroConsistency(Particles *particles, real *shepardCorrection) {
            particles->setZeroConsistency(shepardCorrection);
        }
        void Launch::setZeroConsistency(Particles *particles, real *shepardCorrection) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::setZeroConsistency, particles,
                         shepardCorrection);
        }
#endif
#if LINEAR_CONSISTENCY
        __global__ void setLinearConsistency(Particles *particles, real *tensorialCorrectionMatrix) {
            particles->setLinearConsistency(tensorialCorrectionMatrix);
        }
        void Launch::setLinearConsistency(Particles *particles, real *tensorialCorrectionMatrix) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::setLinearConsistency, particles,
                         tensorialCorrectionMatrix);
        }
#endif
#if FRAGMENTATION
        __global__ void setFragmentation(Particles *particles, real *d, real *damage_total, real *dddt, integer *numFlaws,
                                         integer *maxNumFlaws, integer *numActiveFlaws, real *flaws) {
            particles->setFragmentation(d, damage_total, dddt, numFlaws, maxNumFlaws, numActiveFlaws, flaws);
        }
        void Launch::setFragmentation(Particles *particles, real *d, real *damage_total, real *dddt, integer *numFlaws,
                                integer *maxNumFlaws, integer *numActiveFlaws, real *flaws) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::setFragmentation, particles, d, damage_total,
                         dddt, numFlaws, maxNumFlaws, numActiveFlaws, flaws);
        }
#if PALPHA_POROSITY
        __global__ void setPalphaPorosity(Particles *particles, real *damage_porjutzi, real *ddamage_porjutzidt) {
            particles->setPalphaPorosity(damage_porjutzi, ddamage_porjutzidt);
        }
        void Launch::setPalphaPorosity(Particles *particles, real *damage_porjutzi, real *ddamage_porjutzidt) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::setPalphaPorosity, particles, damage_porjutzi,
                         ddamage_porjutzidt);
        }
#endif
#endif



        __global__ void test(Particles *particles) {

            int bodyIndex = threadIdx.x + blockDim.x * blockIdx.x;
            int stride = blockDim.x * gridDim.x;
            int offset = 0;

            if (bodyIndex == 0) {
                printf("device: numParticles = %i\n", *particles->numParticles);
            }

            while ((bodyIndex + offset) < *particles->numParticles) {
                if ((bodyIndex + offset) % 10000 == 0) {
                    printf("device: x[%i] = (%f, %f, %f)\n", bodyIndex + offset, particles->x[bodyIndex + offset],
                           particles->y[bodyIndex + offset],
                           particles->z[bodyIndex + offset]);
                }
                offset += stride;
            }


        }

        real Launch::test(Particles *particles, bool time) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(time, executionPolicy, ::ParticlesNS::Kernel::test, particles);
            //testKernel<<<256, 256>>>(particles);
        }
    }

}

CUDA_CALLABLE_MEMBER IntegratedParticles::IntegratedParticles() {

}

CUDA_CALLABLE_MEMBER IntegratedParticles::IntegratedParticles(integer *uid, real *drhodt, real *dxdt, real *dvxdt) :
                                                                uid(uid), drhodt(drhodt), dxdt(dxdt), dvxdt(dvxdt) {

}

CUDA_CALLABLE_MEMBER void IntegratedParticles::set(integer *uid, real *drhodt, real *dxdt, real *dvxdt) {
    this->uid = uid;
    this->drhodt = drhodt;
    this->dxdt = dxdt;
    this->dvxdt = dvxdt;
}

#if DIM > 1

CUDA_CALLABLE_MEMBER IntegratedParticles::IntegratedParticles(integer *uid, real *drhodt, real *dxdt, real *dydt,
                                                              real *dvxdt, real *dvydt) : uid(uid), drhodt(drhodt),
                                                              dxdt(dxdt), dydt(dydt), dvxdt(dvxdt), dvydt(dvydt) {

}

CUDA_CALLABLE_MEMBER void IntegratedParticles::set(integer *uid, real *drhodt, real *dxdt, real *dydt, real *dvxdt,
                                                   real *dvydt) {
    this->uid = uid;
    this->drhodt = drhodt;
    this->dxdt = dxdt;
    this->dydt = dydt;
    this->dvxdt = dvxdt;
    this->dvydt = dvydt;
}

#if DIM == 3

CUDA_CALLABLE_MEMBER IntegratedParticles::IntegratedParticles(integer *uid, real *drhodt, real *dxdt, real *dydt,
                                                              real *dzdt, real *dvxdt, real *dvydt, real *dvzdt) :
                                                              uid(uid), drhodt(drhodt), dxdt(dxdt), dydt(dydt),
                                                              dzdt(dzdt), dvxdt(dvxdt), dvydt(dvydt), dvzdt(dvzdt) {

}

CUDA_CALLABLE_MEMBER void IntegratedParticles::set(integer *uid, real *drhodt, real *dxdt, real *dydt, real *dzdt,
                                                   real *dvxdt, real *dvydt, real *dvzdt) {
    this->uid = uid;
    this->drhodt = drhodt;
    this->dxdt = dxdt;
    this->dydt = dydt;
    this->dzdt = dzdt;
    this->dvxdt = dvxdt;
    this->dvydt = dvydt;
    this->dvzdt = dvzdt;
}

#endif
#endif

CUDA_CALLABLE_MEMBER void IntegratedParticles::reset(integer index) {
    uid[index] = 0;
    dxdt[index] = 0.;
    dvxdt[index] = 0.;
#if DIM > 1
    dydt[index] = 0.;
    dvydt[index] = 0.;
#if DIM == 3
    dzdt[index] = 0.;
    dvzdt[index] = 0.;
#endif
#endif
}

CUDA_CALLABLE_MEMBER IntegratedParticles::~IntegratedParticles() {

}

namespace IntegratedParticlesNS {

    namespace Kernel {

        __global__ void set(IntegratedParticles *integratedParticles, integer *uid, real *drhodt, real *dxdt,
                            real *dvxdt) {
            integratedParticles->set(uid, drhodt, dxdt, dvxdt);
        }

        void Launch::set(IntegratedParticles *integratedParticles, integer *uid, real *drhodt, real *dxdt,
                         real *dvxdt) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::IntegratedParticlesNS::Kernel::set, integratedParticles, uid,
                         drhodt, dxdt, dvxdt);
        }

#if DIM > 1

        __global__ void set(IntegratedParticles *integratedParticles, integer *uid, real *drhodt, real *dxdt,
                            real *dydt, real *dvxdt, real *dvydt) {
            integratedParticles->set(uid, drhodt, dxdt, dydt, dvxdt, dvydt);
        }

        void Launch::set(IntegratedParticles *integratedParticles, integer *uid, real *drhodt, real *dxdt, real *dydt,
                         real *dvxdt, real *dvydt) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::IntegratedParticlesNS::Kernel::set, integratedParticles, uid,
                         drhodt, dxdt, dydt, dvxdt, dvydt);
        }

#if DIM == 3

        __global__ void set(IntegratedParticles *integratedParticles, integer *uid, real *drhodt, real *dxdt,
                            real *dydt, real *dzdt, real *dvxdt, real *dvydt, real *dvzdt) {
            integratedParticles->set(uid, drhodt, dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt);
        }

        void Launch::set(IntegratedParticles *integratedParticles, integer *uid, real *drhodt, real *dxdt, real *dydt,
                         real *dzdt, real *dvxdt, real *dvydt, real *dvzdt) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::IntegratedParticlesNS::Kernel::set, integratedParticles, uid,
                         drhodt, dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt);
        }

#endif
#endif
    }
}
