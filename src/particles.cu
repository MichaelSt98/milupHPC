#include "../include/particles.cuh"
#include "../include/cuda_utils/cuda_launcher.cuh"

CUDA_CALLABLE_MEMBER Particles::Particles() {

}

#if DIM == 1

CUDA_CALLABLE_MEMBER Particles::Particles(integer *numParticles, integer *numNodes, real *mass, real *x, real *vx,
                                          real *ax, integer *level, idInteger *uid, integer *materialId,
                                          real *sml, integer *nnl, integer *noi, real *e, real *dedt, real *cs,
                                          real *rho, real *p) :
                                          numParticles(numParticles), numNodes(numNodes), mass(mass), x(x), vx(vx),
                                          ax(ax), level(level), uid(uid), materialId(materialId), sml(sml),
                                          nnl(nnl), noi(noi), e(e), dedt(dedt), cs(cs), rho(rho), p(p) {

}

CUDA_CALLABLE_MEMBER void Particles::set(integer *numParticles, integer *numNodes, real *mass, real *x, real *vx,
                                         real *ax, integer *level, idInteger *uid, integer *materialId,
                                         real *sml, integer *nnl, integer *noi, real *e, real *dedt,
                                         real *cs, real *rho, real *p) {

    this->numParticles = numParticles;
    this->numNodes = numNodes;
    this->mass = mass;
    this->x = x;
    this->vx = vx;
    this->ax = ax;
    this->level = level;
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

#elif DIM == 2

CUDA_CALLABLE_MEMBER Particles::Particles(integer *numParticles, integer *numNodes, real *mass, real *x, real *y,
                                          real *vx, real *vy, real *ax, real *ay, idInteger *uid,
                                          integer *materialId, real *sml, integer *nnl, integer *noi,
                                          real *e, real *dedt, real *cs, real *rho, real *p) :
                                          numParticles(numParticles), numNodes(numNodes),
                                          mass(mass), x(x), y(y), vx(vx), vy(vy), ax(ax), ay(ay),
                                          uid(uid), materialId(materialId), sml(sml), nnl(nnl), noi(noi),
                                          e(e), dedt(dedt), cs(cs), rho(rho), p(p) {

}

CUDA_CALLABLE_MEMBER void Particles::set(integer *numParticles, integer *numNodes, real *mass, real *x, real *y,
                                         real *vx, real *vy, real *ax, real *ay, integer *level, idInteger *uid,
                                         integer *materialId, real *sml, integer *nnl, integer *noi, real *e,
                                         real *dedt, real *cs, real *rho, real *p) {

    this->numParticles = numParticles;
    this->numNodes = numNodes;
    this->mass = mass;
    this->x = x;
    this->y = y;
    this->vx = vx;
    this->vy = vy;
    this->ax = ax;
    this->ay = ay;
    this->level = level;
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

#else

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
                                         integer *level, idInteger *uid, integer *materialId, real *sml, integer *nnl,
                                         integer *noi, real *e, real *dedt, real *cs, real *rho, real *p) {

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
    this->level = level;
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

#if DIM == 1
    CUDA_CALLABLE_MEMBER void Particles::setGravity(real *g_ax) {
        this->g_ax = g_ax;
    }
#elif DIM == 2
    CUDA_CALLABLE_MEMBER void Particles::setGravity(real *g_ax, real *g_ay) {
        this->g_ax = g_ax;
        this->g_ay = g_ay;
    }
#else
    CUDA_CALLABLE_MEMBER void Particles::setGravity(real *g_ax, real *g_ay, real *g_az) {
        this->g_ax = g_ax;
        this->g_ay = g_ay;
        this->g_az = g_az;
    }
#endif

CUDA_CALLABLE_MEMBER void Particles::setU(real *u) {
    this->u = u;
}

CUDA_CALLABLE_MEMBER void Particles::setArtificialViscosity(real *muijmax) {
    this->muijmax = muijmax;
}

//#if INTEGRATE_DENSITY
    CUDA_CALLABLE_MEMBER void Particles::setIntegrateDensity(real *drhodt) {
        this->drhodt = drhodt;
    }
//#endif
#if VARIABLE_SML || INTEGRATE_SML
    CUDA_CALLABLE_MEMBER void Particles::setVariableSML(real *dsmldt) {
        this->dsmldt = dsmldt;
    }
#endif
#if SML_CORRECTION
    CUDA_CALLABLE_MEMBER void Particles::setSMLCorrection(real *sml_omega) {
        this->sml_omega = sml_omega;
    }
#endif
#if NAVIER_STOKES
    CUDA_CALLABLE_MEMBER void Particles::setNavierStokes(real *Tshear, real *eta) {
        this->Tshear = Tshear;
        this->eta = eta;
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
    CUDA_CALLABLE_MEMBER void Particles::setSolidNavierStokes(real *sigma) {
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
    level[index] = -1;
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
    
    real dx;
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
    real dy;
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
    real dz;
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
    return sqrt(dx*dx);
#elif DIM == 2
    return sqrt(dx*dx + dy*dy);
#else
    return sqrt(dx*dx + dy*dy + dz*dz);
#endif

}

CUDA_CALLABLE_MEMBER real Particles::weightedEntry(integer index, Entry::Name entry) {
    switch (entry) {
        case Entry::x: {
            return x[index] * mass[index];
        } break;
#if DIM > 1
        case Entry::y: {
            return y[index] * mass[index];
        } break;
#if DIM == 3
        case Entry::z: {
            return z[index] * mass[index];
        }
#endif
#endif
        default: {
            printf("Entry not available!\n");
            return (real)0;
        }
    }
}

CUDA_CALLABLE_MEMBER Particles::~Particles() {

}


namespace ParticlesNS {

    namespace Kernel {

        /*__global__ void check4nans(Particles *particles, integer n) {
            int bodyIndex = threadIdx.x + blockDim.x * blockIdx.x;
            int stride = blockDim.x * gridDim.x;
            int offset = 0;

            while ((bodyIndex + offset) < n) {
                if (std::isnan(particles->x[bodyIndex + offset]) || std::isnan(particles->mass[bodyIndex + offset])
#if DIM > 1
                    || std::isnan(particles->y[bodyIndex + offset])
#if DIM == 3
                    || std::isnan(particles->z[bodyIndex + offset])
#endif
#endif
                ) {
#if DIM == 1
                    printf("NAN for index: %i (%f) %f\n", bodyIndex + offset,
                           particles->x[bodyIndex + offset],
                           particles->mass[bodyIndex + offset]);
#elif DIM == 2
                    printf("NAN for index: %i (%f, %f) %f\n", bodyIndex + offset,
                           particles->x[bodyIndex + offset],
                           particles->y[bodyIndex + offset],
                           particles->mass[bodyIndex + offset]);
#else
                    printf("NAN for index: %i (%f, %f, %f) %f\n", bodyIndex + offset,
                           particles->x[bodyIndex + offset],
                           particles->y[bodyIndex + offset],
                           particles->z[bodyIndex + offset],
                           particles->mass[bodyIndex + offset]);
#endif
                    assert(0);


                }

                if (particles->mass[bodyIndex + offset] == 0. || particles->sml[bodyIndex + offset] == 0.) {
#if DIM == 1
                    printf("ATTENTION for index: %i (%f) %f\n", bodyIndex + offset,
                           particles->x[bodyIndex + offset],
                           particles->mass[bodyIndex + offset]);
#elif DIM == 2
                    printf("ATTENTION for index: %i (%f, %f) %f\n", bodyIndex + offset,
                           particles->x[bodyIndex + offset],
                           particles->y[bodyIndex + offset],
                           particles->mass[bodyIndex + offset]);
#else
                    printf("ATTENTION for index: %i (%e, %e, %e) %e sml = %e\n", bodyIndex + offset,
                           particles->x[bodyIndex + offset],
                           particles->y[bodyIndex + offset],
                           particles->z[bodyIndex + offset],
                           particles->mass[bodyIndex + offset],
                           particles->sml[bodyIndex + offset]);
#endif
                    assert(0);
                }



                offset += stride;
            }
        }*/

        __global__ void check4nans(Particles *particles, integer n) {
            int bodyIndex = threadIdx.x + blockDim.x * blockIdx.x;
            int stride = blockDim.x * gridDim.x;
            int offset = 0;

            while ((bodyIndex + offset) < n) {
                if (std::isnan(particles->x[bodyIndex + offset]) || std::isnan(particles->mass[bodyIndex + offset])
                    #if DIM > 1
                    || std::isnan(particles->y[bodyIndex + offset])
                    #if DIM == 3
                    || std::isnan(particles->z[bodyIndex + offset])
#endif
#endif
                        ) {
#if DIM == 1
                    printf("NAN for index: %i (%f) %f\n", bodyIndex + offset,
                           particles->x[bodyIndex + offset],
                           particles->mass[bodyIndex + offset]);
#elif DIM == 2
                    printf("NAN for index: %i (%f, %f) %f\n", bodyIndex + offset,
                           particles->x[bodyIndex + offset],
                           particles->y[bodyIndex + offset],
                           particles->mass[bodyIndex + offset]);
#else
                    printf("NAN for index: %i (%f, %f, %f) %f\n", bodyIndex + offset,
                           particles->x[bodyIndex + offset],
                           particles->y[bodyIndex + offset],
                           particles->z[bodyIndex + offset],
                           particles->mass[bodyIndex + offset]);
#endif
                    assert(0);


                }

                if (particles->mass[bodyIndex + offset] == 0. || particles->sml[bodyIndex + offset] == 0.) {
#if DIM == 1
                    printf("ATTENTION for index: %i (%f) %f\n", bodyIndex + offset,
                           particles->x[bodyIndex + offset],
                           particles->mass[bodyIndex + offset]);
#elif DIM == 2
                    printf("ATTENTION for index: %i (%f, %f) %f\n", bodyIndex + offset,
                           particles->x[bodyIndex + offset],
                           particles->y[bodyIndex + offset],
                           particles->mass[bodyIndex + offset]);
#else
                    printf("ATTENTION for index: %i (%e, %e, %e) %e sml = %e\n", bodyIndex + offset,
                           particles->x[bodyIndex + offset],
                           particles->y[bodyIndex + offset],
                           particles->z[bodyIndex + offset],
                           particles->mass[bodyIndex + offset],
                           particles->sml[bodyIndex + offset]);
#endif
                    assert(0);
                }

                if (particles->x[bodyIndex + offset] > 1.e250
                    #if DIM > 1
                    || particles->y[bodyIndex + offset] > 1.e250
                    #if DIM == 3
                    || particles->z[bodyIndex + offset] > 1.e250
#endif
#endif
                        ) {
                    printf("HUGE entry for index: %i (%e, %e, %e) %e\n", bodyIndex + offset,
                           particles->x[bodyIndex + offset],
                           particles->y[bodyIndex + offset],
                           particles->z[bodyIndex + offset],
                           particles->mass[bodyIndex + offset]);
                    assert(0);
                }

                /*if (bodyIndex + offset == 128121) {
                    printf("INFO for index: %i (%e, %e, %e) %e sml = %e\n", bodyIndex + offset,
                           particles->x[bodyIndex + offset],
                           particles->y[bodyIndex + offset],
                           particles->z[bodyIndex + offset],
                           particles->mass[bodyIndex + offset],
                           particles->sml[bodyIndex + offset]);
                }*/

                /*if (particles->sml[bodyIndex + offset] < 1.e-20) {
                    printf("sml = %e\n", particles->sml[bodyIndex + offset]);
                    assert(0);
                }*/

                offset += stride;
            }
        }

        __global__ void info(Particles *particles, integer n, integer m, integer k) {
            int bodyIndex = threadIdx.x + blockDim.x * blockIdx.x;
            int stride = blockDim.x * gridDim.x;
            int offset = 0;

#if DIM == 1
            printf("not implemented yet for DIM == 1...\n");
#elif DIM == 2
            printf("not implemented yet for DIM == 2...\n");
#else
            while ((bodyIndex + offset) < n) {
                //if ((bodyIndex + offset) % 100 == 0) {
                printf("x[%i] = (%f, %f, %f) v = (%f, %f, %f) a = (%f, %f, %f) mass = %f\n", bodyIndex + offset,
                       particles->x[bodyIndex + offset], particles->y[bodyIndex + offset], particles->z[bodyIndex + offset],
                       particles->vx[bodyIndex + offset], particles->vy[bodyIndex + offset], particles->vz[bodyIndex + offset],
                       particles->ax[bodyIndex + offset], particles->ay[bodyIndex + offset], particles->az[bodyIndex + offset],
                       particles->mass[bodyIndex + offset]);
                //}
                offset += stride;
            }

            offset = m;
            while ((bodyIndex + offset) < k && (bodyIndex + offset) > m) {
                //if ((bodyIndex + offset) % 100 == 0) {
                printf("x[%i] = (%f, %f, %f) mass = %f\n", bodyIndex + offset, particles->x[bodyIndex + offset],
                       particles->y[bodyIndex + offset],
                       particles->z[bodyIndex + offset],
                       particles->mass[bodyIndex + offset]);
                //}
                offset += stride;
            }
#endif

        }

        real Launch::check4nans(Particles *particles, integer n) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::ParticlesNS::Kernel::check4nans, particles, n);
        }

        real Launch::info(Particles *particles, integer n, integer m, integer k) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::ParticlesNS::Kernel::info, particles, n, m, k);
        }

#if DIM == 1

        __global__ void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x,
                            real *vx, real *ax, integer *level, idInteger *uid, integer *materialId,
                            real *sml, integer *nnl, integer *noi, real *e, real *dedt, real *cs, real *rho, real *p) {

            particles->set(numParticles, numNodes, mass, x, vx, ax, level, uid, materialId, sml, nnl, noi, e,
                           dedt, cs, rho, p);

        }

        void Launch::set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x, real *vx,
                         real *ax, integer *level, idInteger *uid, integer *materialId, real *sml,
                         integer *nnl, integer *noi, real *e, real *dedt, real *cs, real *rho, real *p) {

            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::set, particles, numParticles, numNodes, mass,
                         x, vx, ax, level, uid, materialId, sml, nnl, noi, e, dedt, cs, rho, p);

        }

#elif DIM == 2

        __global__ void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x,
                            real *y, real *vx, real *vy, real *ax, real *ay, integer *level,
                            idInteger *uid, integer *materialId, real *sml, integer *nnl, integer *noi, real *e,
                            real *dedt, real *cs, real *rho, real *p) {

            particles->set(numParticles, numNodes, mass, x, y, vx, vy, ax, ay, level, uid, materialId, sml,
                           nnl, noi, e, dedt, cs, rho, p);

        }

        void Launch::set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x, real *y,
                         real *vx, real *vy, real *ax, real *ay, integer *level, idInteger *uid,
                         integer *materialId, real *sml, integer *nnl, integer *noi, real *e, real *dedt, real *cs,
                         real *rho, real *p) {

            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::set, particles, numParticles, numNodes,
                         mass, x, y, vx, vy, ax, ay, level, uid, materialId, sml, nnl, noi, e, dedt,
                         cs, rho, p);

        }


#else

        __global__ void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x,
                            real *y, real *z, real *vx, real *vy, real *vz, real *ax, real *ay, real *az,
                            integer *level, idInteger *uid, integer *materialId, real *sml, integer *nnl,
                            integer *noi, real *e, real *dedt, real *cs, real *rho, real *p) {

            particles->set(numParticles, numNodes, mass, x, y, z, vx, vy, vz, ax, ay, az, level, uid,
                           materialId, sml, nnl, noi, e, dedt, cs, rho, p);

        }

        void Launch::set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x, real *y,
                         real *z, real *vx, real *vy, real *vz, real *ax, real *ay, real *az, integer *level,
                         idInteger *uid, integer *materialId, real *sml, integer *nnl, integer *noi, real *e,
                         real *dedt, real *cs, real *rho, real *p) {

            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::set, particles, numParticles, numNodes,
                         mass, x, y, z, vx, vy, vz, ax, ay, az, level, uid, materialId, sml,
                         nnl, noi, e, dedt, cs, rho, p);
            //setKernel<<<1, 1>>>(particles, count, mass, x, y, z, vx, vy, vz, ax, ay, az);

        }

#endif

#if DIM == 1
        __global__ void setGravity(Particles *particles, real *g_ax) {
            particles->setGravity(g_ax);
        }

        namespace Launch {
            void setGravity(Particles *particles, real *g_ax) {
                ExecutionPolicy executionPolicy(1, 1);
                cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::setGravity, particles, g_ax);
            }
        }

#elif DIM == 2
        __global__ void setGravity(Particles *particles, real *g_ax, real *g_ay) {
            particles->setGravity(g_ax, g_ay);
        }

        namespace Launch {
            void setGravity(Particles *particles, real *g_ax, real *g_ay) {
                ExecutionPolicy executionPolicy(1, 1);
                cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::setGravity, particles,
                             g_ax, g_ay);
            }
        }
#else
        __global__ void setGravity(Particles *particles, real *g_ax, real *g_ay, real *g_az) {
            particles->setGravity(g_ax, g_ay, g_az);
        }

        namespace Launch {
            void setGravity(Particles *particles, real *g_ax, real *g_ay, real *g_az) {
                ExecutionPolicy executionPolicy(1, 1);
                cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::setGravity, particles,
                             g_ax, g_ay, g_az);
            }
        }
#endif

        __global__ void setU(Particles *particles, real *u) {
            particles->setU(u);
        }

        namespace Launch {
            void setU(Particles *particles, real *u) {
                ExecutionPolicy executionPolicy(1, 1);
                cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::setU, particles, u);
            }
        }

        __global__ void setArtificialViscosity(Particles *particles, real *muijmax) {
            particles->setArtificialViscosity(muijmax);
        }

        namespace Launch {
            void setArtificialViscosity(Particles *particles, real *muijmax) {
                ExecutionPolicy executionPolicy(1, 1);
                cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::setArtificialViscosity, particles,
                             muijmax);
            }
        }

//#if INTEGRATE_DENSITY
        __global__ void setIntegrateDensity(Particles *particles, real *drhodt) {
            particles->setIntegrateDensity(drhodt);
        }
        void Launch::setIntegrateDensity(Particles *particles, real *drhodt) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::setIntegrateDensity, particles, drhodt);
        }
//#endif
#if VARIABLE_SML || INTEGRATE_SML
        __global__ void setVariableSML(Particles *particles, real *dsmldt) {
            particles->setVariableSML(dsmldt);
        }
        void Launch::setVariableSML(Particles *particles, real *dsmldt) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::setVariableSML, particles, dsmldt);
        }
#endif
#if SML_CORRECTION
        __global__ void setSMLCorrection(Particles *particles, real *sml_omega) {
            particles->setSMLCorrection(sml_omega);
        }

        void Launch::setSMLCorrection(Particles *particles, real *sml_omega) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::setSMLCorrection, particles, sml_omega);
        }
#endif
#if NAVIER_STOKES
        __global__ void setNavierStokes(Particles *particles, real *Tshear, real *eta) {
            particles->setNavierStokes(Tshear, eta);
        }
        namespace Launch {
            void setNavierStokes(Particles *particles, real *Tshear, real *eta) {
                ExecutionPolicy executionPolicy(1, 1);
                cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::setNavierStokes, particles, Tshear, eta);
            }
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
        __global__ void setSolidNavierStokes(Particles *particles, real *sigma) {
            particles->setSolidNavierStokes(sigma);
        }
        void Launch::setSolidNavierStokes(Particles *particles, real *sigma) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::setSolidNavierStokes, particles, sigma);
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

#if DIM == 1
            printf("not implemented yet for DIM == 1...\n");
#elif DIM == 2
            printf("not implemented yet for DIM == 2...\n");
#else
            while ((bodyIndex + offset) < *particles->numParticles) {
                if ((bodyIndex + offset) % 10000 == 0) {
                    printf("device: x[%i] = (%f, %f, %f)\n", bodyIndex + offset, particles->x[bodyIndex + offset],
                           particles->y[bodyIndex + offset],
                           particles->z[bodyIndex + offset]);
                }
                offset += stride;
            }
#endif


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

#if DIM == 1

CUDA_CALLABLE_MEMBER IntegratedParticles::IntegratedParticles(idInteger *uid, real *rho, real *e, real *dedt, real *p,
                                                              real *cs, real *x, real *vx, real *ax) :
                                                              uid(uid), rho(rho), e(e), dedt(dedt), p(p), cs(cs),
                                                              x(x), vx(vx), ax(ax) {

}

CUDA_CALLABLE_MEMBER void IntegratedParticles::set(idInteger *uid, real *rho, real *e, real *dedt, real *p, real *cs,
                                                   real *x, real *vx, real *ax) {
    this->uid = uid;
    this->rho = rho;
    this->e = e;
    this->dedt = dedt;
    this->p = p;
    this->cs = cs;
    this->x = x;
    this->vx = vx;
    this->ax = ax;
}

#elif DIM == 2

CUDA_CALLABLE_MEMBER IntegratedParticles::IntegratedParticles(idInteger *uid, real *rho, real *e, real *dedt, real *p,
                                                              real *cs, real *x, real *y, real *vx, real *vy, real *ax,
                                                              real *ay) : uid(uid), rho(rho), e(e), dedt(dedt),
                                                              p(p), cs(cs), x(x), y(y), vx(vx), vy(vy), ax(ax),
                                                              ay(ay) {

}

CUDA_CALLABLE_MEMBER void IntegratedParticles::set(idInteger *uid, real *rho, real *e, real *dedt, real *p,
                                                   real *cs, real *x, real *y, real *vx, real *vy, real *ax,
                                                   real *ay) {
    this->uid = uid;
    this->rho = rho;
    this->e = e;
    this->dedt = dedt;
    this->p = p;
    this->cs = cs;
    this->x = x;
    this->y = y;
    this->vx = vx;
    this->vy = vy;
    this->ax = ax;
    this->ay = ay;
}

#else

CUDA_CALLABLE_MEMBER IntegratedParticles::IntegratedParticles(idInteger *uid, real *rho, real *e, real *dedt, real *p,
                                                              real *cs, real *x, real *y, real *z, real *vx, real *vy,
                                                              real *vz, real *ax, real *ay, real *az) :
                                                              uid(uid), rho(rho), e(e), dedt(dedt), p(p), cs(cs),
                                                              x(x), y(y), z(z), vx(vx), vy(vy), vz(vz), ax(ax), ay(ay),
                                                              az(az) {

}

CUDA_CALLABLE_MEMBER void IntegratedParticles::set(idInteger *uid, real *rho, real *e, real *dedt, real *p,
                                                   real *cs, real *x, real *y, real *z, real *vx, real *vy,
                                                   real *vz, real *ax, real *ay, real *az) {
    this->uid = uid;
    this->rho = rho;
    this->e = e;
    this->dedt = dedt;
    this->p = p;
    this->cs = cs;
    this->x = x;
    this->y = y;
    this->z = z;
    this->vx = vx;
    this->vy = vy;
    this->vz = vz;
    this->ax = ax;
    this->ay = ay;
    this->az = az;
}

#endif

CUDA_CALLABLE_MEMBER void IntegratedParticles::setSML(real *sml) {
    this->sml = sml;
}

//#if INTEGRATE_DENSITY
CUDA_CALLABLE_MEMBER void IntegratedParticles::setIntegrateDensity(real *drhodt) {
    this->drhodt = drhodt;
}
//#endif

#if VARIABLE_SML || INTEGRATE_SML
CUDA_CALLABLE_MEMBER void IntegratedParticles::setIntegrateSML(real *dsmldt) {
    this->dsmldt = dsmldt;
}
#endif

CUDA_CALLABLE_MEMBER void IntegratedParticles::reset(integer index) {

    //TODO: what to reset?
    uid[index] = 0;
    vx[index] = 0.;
    ax[index] = 0.;
#if DIM > 1
    vy[index] = 0.;
    ay[index] = 0.;
#if DIM == 3
    vz[index] = 0.;
    az[index] = 0.;
#endif
#endif
}

CUDA_CALLABLE_MEMBER IntegratedParticles::~IntegratedParticles() {

}

namespace IntegratedParticlesNS {

    namespace Kernel {

#if DIM == 1

        __global__ void set(IntegratedParticles *integratedParticles, idInteger *uid, real *rho, real *e, real *dedt,
                            real *p, real *cs, real *x, real *vx, real *ax) {
            integratedParticles->set(uid, rho, e, dedt, p, cs, x, vx, ax);
        }

        void Launch::set(IntegratedParticles *integratedParticles, idInteger *uid, real *rho, real *e, real *dedt,
                         real *p, real *cs, real *x, real *vx, real *ax) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::IntegratedParticlesNS::Kernel::set, integratedParticles, uid,
                         rho, e, dedt, p, cs, x, vx, ax);
        }

#elif DIM == 2

        __global__ void set(IntegratedParticles *integratedParticles, idInteger *uid, real *rho, real *e, real *dedt,
                            real *p, real *cs, real *x, real *y, real *vx, real *vy, real *ax, real *ay) {
            integratedParticles->set(uid, rho, e, dedt, p, cs, x, y, vx, vy, ax, ay);
        }

        void Launch::set(IntegratedParticles *integratedParticles, idInteger *uid, real *rho, real *e, real *dedt,
                         real *p, real *cs, real *x, real *y, real *vx, real *vy, real *ax, real *ay) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::IntegratedParticlesNS::Kernel::set, integratedParticles, uid,
                         rho, e, dedt, p, cs, x, y, vx, vy, ax, ay);
        }

#else

        __global__ void set(IntegratedParticles *integratedParticles, idInteger *uid, real *rho, real *e, real *dedt,
                            real *p, real *cs, real *x, real *y, real *z, real *vx, real *vy, real *vz, real *ax,
                            real *ay, real *az) {
            integratedParticles->set(uid, rho, e, dedt, p, cs, x, y, z, vx, vy, vz, ax, ay, az);
        }

        void Launch::set(IntegratedParticles *integratedParticles, idInteger *uid, real *rho, real *e, real *dedt,
                         real *p, real *cs, real *x, real *y, real *z, real *vx, real *vy, real *vz, real *ax,
                         real *ay, real *az) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::IntegratedParticlesNS::Kernel::set, integratedParticles, uid,
                         rho, e, dedt, p, cs, x, y, z, vx, vy, vz, ax, ay, az);
        }

#endif

        __global__ void setSML(IntegratedParticles *integratedParticles, real *sml) {
            integratedParticles->setSML(sml);
        }

        namespace Launch {
            void setSML(IntegratedParticles *integratedParticles, real *sml) {
                ExecutionPolicy executionPolicy(1, 1);
                cuda::launch(false, executionPolicy, ::IntegratedParticlesNS::Kernel::setSML,
                             integratedParticles, sml);
            }
        }

//#if INTEGRATE_DENSITY
        __global__ void setIntegrateDensity(IntegratedParticles *integratedParticles, real *drhodt) {
            integratedParticles->setIntegrateDensity(drhodt);
        }

        namespace Launch {

            void setIntegrateDensity(IntegratedParticles *integratedParticles, real *drhodt) {
                ExecutionPolicy executionPolicy(1, 1);
                cuda::launch(false, executionPolicy, ::IntegratedParticlesNS::Kernel::setIntegrateDensity,
                             integratedParticles, drhodt);
            }

        }
//#endif

#if VARIABLE_SML || INTEGRATE_SML
        __global__ void setIntegrateSML(IntegratedParticles *integratedParticles, real *dsmldt) {
            integratedParticles->setIntegrateSML(dsmldt);
        }

        namespace Launch {
            void setIntegrateSML(IntegratedParticles *integratedParticles, real *dsmldt) {
                ExecutionPolicy executionPolicy(1, 1);
                cuda::launch(false, executionPolicy, ::IntegratedParticlesNS::Kernel::setIntegrateSML,
                             integratedParticles, dsmldt);
            }
        }
#endif
    }
}
