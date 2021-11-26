#include "../../include/integrator/device_predictor_corrector_euler.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

namespace PredictorCorrectorEulerNS {


    CUDA_CALLABLE_MEMBER Shared::Shared() {

    }
    CUDA_CALLABLE_MEMBER Shared::Shared(real *forces, real *courant, real *artVisc) {
        this->forces = forces;
        this->courant = courant;
        this->artVisc = artVisc;
    }
    CUDA_CALLABLE_MEMBER Shared::~Shared() {

    }
    CUDA_CALLABLE_MEMBER void Shared::set(real *forces, real *courant, real *artVisc) {
        this->forces = forces;
        this->courant = courant;
        this->artVisc= artVisc;
    }
    CUDA_CALLABLE_MEMBER void Shared::setE(real *e) {
        this->e = e;
    }
    CUDA_CALLABLE_MEMBER void Shared::setRho(real *rho) {
        this->rho = rho;
    }
    CUDA_CALLABLE_MEMBER void Shared::setVmax(real *vmax) {
        this->vmax = vmax;
    }
    namespace SharedNS {
        __global__ void set(Shared *shared, real *forces, real *courant, real *artVisc) {
            shared->set(forces, courant, artVisc);
        }
        __global__ void setE(Shared *shared, real *e) {
            shared->setE(e);
        }
        __global__ void setRho(Shared *shared, real *rho) {
            shared->setRho(rho);
        }
        __global__ void setVmax(Shared *shared, real *vmax) {
            shared->setVmax(vmax);
        }
        namespace Launch {
            void set(Shared *shared, real *forces, real *courant, real *artVisc) {
                ExecutionPolicy executionPolicy(1, 1);
                cuda::launch(false, executionPolicy, ::PredictorCorrectorEulerNS::SharedNS::set, shared,
                             forces, courant, artVisc);
            }
            void setE(Shared *shared, real *e) {
                ExecutionPolicy executionPolicy(1, 1);
                cuda::launch(false, executionPolicy, ::PredictorCorrectorEulerNS::SharedNS::setE, shared, e);
            }
            void setRho(Shared *shared, real *rho) {
                ExecutionPolicy executionPolicy(1, 1);
                cuda::launch(false, executionPolicy, ::PredictorCorrectorEulerNS::SharedNS::setRho, shared, rho);
            }
            void setVmax(Shared *shared, real *vmax) {
                ExecutionPolicy executionPolicy(1, 1);
                cuda::launch(false, executionPolicy, ::PredictorCorrectorEulerNS::SharedNS::setVmax, shared, vmax);
            }
        }
    }

    CUDA_CALLABLE_MEMBER BlockShared::BlockShared() {

    }
    CUDA_CALLABLE_MEMBER BlockShared::BlockShared(real *forces, real *courant, real *artVisc) {
        this->forces = forces;
        this->courant = courant;
        this->artVisc = artVisc;
    }
    CUDA_CALLABLE_MEMBER BlockShared::~BlockShared() {

    }
    CUDA_CALLABLE_MEMBER void BlockShared::set(real *forces, real *courant, real *artVisc) {
        this->forces = forces;
        this->courant = courant;
        this->artVisc= artVisc;
    }
    CUDA_CALLABLE_MEMBER void BlockShared::setE(real *e) {
        this->e = e;
    }
    CUDA_CALLABLE_MEMBER void BlockShared::setRho(real *rho) {
        this->rho = rho;
    }
    CUDA_CALLABLE_MEMBER void BlockShared::setVmax(real *vmax) {
        this->vmax = vmax;
    }
    namespace BlockSharedNS {
        __global__ void set(BlockShared *blockShared, real *forces, real *courant, real *artVisc) {
            blockShared->set(forces, courant, artVisc);
        }
        __global__ void setE(BlockShared *blockShared, real *e) {
            blockShared->setE(e);
        }
        __global__ void setRho(BlockShared *blockShared, real *rho) {
            blockShared->setRho(rho);
        }
        __global__ void setVmax(BlockShared *blockShared, real *vmax) {
            blockShared->setVmax(vmax);
        }

        namespace Launch {
            void set(BlockShared *blockShared, real *forces, real *courant, real *artVisc) {
                ExecutionPolicy executionPolicy(1, 1);
                cuda::launch(false, executionPolicy, ::PredictorCorrectorEulerNS::BlockSharedNS::set, blockShared,
                             forces, courant, artVisc);
            }
            void setE(BlockShared *blockShared, real *e) {
                ExecutionPolicy executionPolicy(1, 1);
                cuda::launch(false, executionPolicy, ::PredictorCorrectorEulerNS::BlockSharedNS::setE, blockShared, e);
            }
            void setRho(BlockShared *blockShared, real *rho) {
                ExecutionPolicy executionPolicy(1, 1);
                cuda::launch(false, executionPolicy, ::PredictorCorrectorEulerNS::BlockSharedNS::setRho,
                             blockShared, rho);
            }
            void setVmax(BlockShared *blockShared, real *vmax) {
                ExecutionPolicy executionPolicy(1, 1);
                cuda::launch(false, executionPolicy, ::PredictorCorrectorEulerNS::BlockSharedNS::setVmax,
                             blockShared, vmax);
            }
        }
    }

    namespace Kernel {

        __global__ void corrector(Particles *particles, IntegratedParticles *predictor, real dt, int numParticles) {

            int i;
            // particle loop
            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i+= blockDim.x * gridDim.x) {

// just for debugging purposes!!!
/*
                particles->vx[i] += dt * (particles->ax[i] + particles->g_ax[i]);
#if DIM > 1
                particles->vy[i] += dt * (particles->ay[i] + particles->g_ay[i]);
#if DIM == 3
                particles->vz[i] += dt * (particles->az[i] + particles->g_az[i]);
#endif
#endif

                // calculating/updating the positions
                particles->x[i] += dt * particles->vx[i];
#if DIM > 1
                particles->y[i] += dt * particles->vy[i];
#if DIM == 3
                particles->z[i] += dt * particles->vz[i];
#endif
#endif
*/
// end: just for debugging purposes!!!

                particles->x[i] = particles->x[i] + dt/2 * (predictor->vx[i] + particles->vx[i]);
                //if (i == 12) { //(i % 1000 == 0) {
                //    printf("corrector: x[%i] = %e + %e/2 * (%e + %e)\n", i, particles->x[i], dt, predictor->vx[i],
                //           particles->vx[i]);
                //}
                particles->vx[i] = particles->vx[i] + dt/2 * (predictor->ax[i] + particles->ax[i] + 2 * particles->g_ax[i]);
                //if (i == 12) { //(i % 1000 == 0) {
                //    printf("corrector: vx[%i] = %e + %e/2 * (%e + %e + 2 * %e)\n", i, particles->vx[i], dt, predictor->ax[i],
                //           particles->ax[i], particles->g_ax);
                //}
                particles->ax[i] = 0.5 * (predictor->ax[i] + particles->ax[i]) + particles->g_ax[i];
                //if (i == 12) { //(i % 1000 == 0) {
                //    printf("corrector: ax[%i] = 1/2 * (%e + %e) + %e)\n", i, predictor->ax[i], particles->ax[i], particles->g_ax);
                //}
#if DIM > 1
                particles->y[i] = particles->y[i] + dt/2 * (predictor->vy[i] + particles->vy[i]);
                particles->vy[i] = particles->vy[i] + dt/2 * (predictor->ay[i] + particles->ay[i] + 2 * particles->g_ay[i]);
                particles->ay[i] = 0.5 * (predictor->ay[i] + particles->ay[i]) + particles->g_ay[i];
#if DIM == 3
                particles->z[i] = particles->z[i] + dt/2 * (predictor->vz[i] + particles->vz[i]);
                particles->vz[i] = particles->vz[i] + dt/2 * (predictor->az[i] + particles->az[i] + 2 * particles->g_az[i]);
                particles->az[i] = 0.5 * (predictor->az[i] + particles->az[i]) + particles->g_az[i];
#endif
#endif

// TODO: some SPH flag?
#if INTEGRATE_DENSITY
                particles->rho[i] = particles->rho[i] + dt/2 * (predictor->drhodt[i] + particles->drhodt[i]);
                particles->drhodt[i] = 0.5 * (predictor->drhodt[i] + particles->drhodt[i]);
                if (i == 12) { //(i % 1000 == 0) {
                    printf("corrector: rho[%i] = %e + %e/2 * (%e + %e)\n", i, particles->rho[i], dt, predictor->drhodt[i],
                           particles->drhodt[i]);
                }
#else
                //p.rho[i] = p.rho[i];
#endif
#if INTEGRATE_ENERGY
                particles->e[i] = particles->e[i] + dt/2 * (predictor->dedt[i] + particles->dedt[i]);
                if (particles->e[i] < 1e-6) {
                    particles->e[i] = 1e-6;
                }
                particles->dedt[i] = 0.5 * (predictor->dedt[i] + particles->dedt[i]);
                //if (i == 12) { //(i % 1000 == 0) {
                //    printf("corrector: e[%i] = %e + %e/2 * (%e + %e)\n", i, particles->e[i], dt, predictor->dedt[i],
                //           particles->dedt[i]);
                //}
#endif
#if INTEGRATE_SML
#if DECOUPLE_SML
                particles->sml[i] = particles->sml[i] + dt * particles->dsmldt[i];
                //particles->dsmldt[i] = particles->dsmldt[i];
#else
                particles->sml[i] = particles->sml[i] + dt/2 * (predictor->dsmldt[i] + particles->dsmldt[i]);
                particles->dsmldt[i] = 0.5 * (predictor->dsmldt[i] + particles->dsmldt[i]);
#endif
#else
                particles->sml[i] = predictor->sml[i];
#endif

                predictor->reset(i); //TODO: move somewhere else?
            }
        }

        __global__ void predictor(Particles *particles, IntegratedParticles *predictor, real dt, int numParticles) {

            int i;

            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i+= blockDim.x * gridDim.x) {

                predictor->x[i] = particles->x[i] + dt * particles->vx[i];
                predictor->vx[i] = particles->vx[i] + dt * (particles->ax[i] + particles->g_ax[i]);
#if DIM > 1
                predictor->y[i] = particles->y[i] + dt * particles->vy[i];
                predictor->vy[i] = particles->vy[i] + dt * (particles->ay[i] + particles->g_ay[i]);
#if DIM == 3
                predictor->z[i] = particles->z[i] + dt * particles->vz[i];
                predictor->vz[i] = particles->vz[i] + dt * (particles->az[i] + particles->g_az[i]);
#endif
#endif

// TODO: some SPH flag?
#if INTEGRATE_DENSITY
                predictor->rho[i] = particles->rho[i] + dt * particles->drhodt[i];
                //predictor->drhodt[i] = particles->drhodt[i];
#else
                //predictor->rho[i] = particles->rho[i];
#endif
#if INTEGRATE_ENERGY
                predictor->e[i] = particles->e[i] + dt * particles->dedt[i];
                // TODO: in principle there should not be a energy floor (but needed for sedov)
                if (predictor->e[i] < 1e-6) {
                    predictor->e[i] = 1e-6;
                }
#endif
#if INTEGRATE_SML
#if DECOUPLE_SML
                predictor->sml[i] = particles->sml[i] + dt * particles->dsmldt[i];
#else
                predictor->sml[i] = particles->sml[i];
#endif
#else
                predictor->sml[i] = particles->sml[i];
#endif
            }

        }

        /**
         * Conditions to be applied:
         *
         * * sound waves traveling faster than a fraction of the smoothing length
         *     * $\Delta t \leq C \frac{h}{c + 1.2 (\alpha_{\nu} c + \beta_{\nu} \mu_{max})}$
         *     * where $c$ is the sound speed; $\alpha_{\nu}$ and $\beta_{\nu}$ are the viscosity parameters, $\mu_{max}$ is the maximal value of $\mu_{ij}$ and $C$ is the Courant number
         * * time step constrains the distance a particle travels due to acceleration
         *     * $\Delta t \leq \sqrt{\frac{h}{|\vec{a}|}}$
         * * all other quantities $f$ have to be prevented from growing too fast within one time step
         *     * $\Delta t \leq \begin{cases} a \frac{|f| + f_{min}}{|df|} & |df| > 0 \\ \Delta t_{max} & |df| = 0 \\ \end{cases}$ where $a < 1$
         * * additional constraint regarding parallelization approach:
         *     * particles should not move further than h/2
         *     * $\Delta t \cdot v_{max} < \frac{h}{2} \, \Leftrightarrow \Delta t < \frac{h}{2 v_{max}}$
         */
        __global__ void setTimeStep(SimulationTime *simulationTime, Material *materials, Particles *particles,
                                    BlockShared *blockShared, int *blockCount, real searchRadius, int numParticles) {

#define SAFETY_FIRST 0.1

            __shared__ real sharedForces[NUM_THREADS_LIMIT_TIME_STEP];
            __shared__ real sharedCourant[NUM_THREADS_LIMIT_TIME_STEP];
            __shared__ real sharedArtVisc[NUM_THREADS_LIMIT_TIME_STEP];
            __shared__ real sharede[NUM_THREADS_LIMIT_TIME_STEP];
            __shared__ real sharedrho[NUM_THREADS_LIMIT_TIME_STEP];
            __shared__ real sharedVmax[NUM_THREADS_LIMIT_TIME_STEP];

            int i, j, k, m;
            int d, dd;
            int index;
            bool hasEnergy;
            real forces = DBL_MAX;
            real courant = DBL_MAX;
            real dtx = DBL_MAX;
            real dtrho = DBL_MAX;
            real dte = DBL_MAX;
            real vmax = 0.; //TODO: initial value
            real temp;
            real sml;
            int matId;

            real ax, ay;
#if DIM == 3
            real az;
#endif
            real dtartvisc = DBL_MAX;

            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i+= blockDim.x * gridDim.x) {

                matId = particles->materialId[i];

#if INTEGRATE_ENERGY
            hasEnergy = false;

//          switch  (matEOS[matId]) {
//              case (EOS_TYPE_TILLOTSON):
//                  hasEnergy = true;
//                  break;
//              case (EOS_TYPE_JUTZI):
//                  hasEnergy = true;
//                  break;
//		  	    case (EOS_TYPE_JUTZI_ANEOS):
//		  		    hasEnergy = true;
//		  		    break;
//              case (EOS_TYPE_SIRONO):
//                  hasEnergy = true;
//                  break;
//              case (EOS_TYPE_EPSILON):
//                  hasEnergy = true;
//                  break;
//              case (EOS_TYPE_ANEOS):
//                  hasEnergy = true;
//                  break;
//              default:
//                  hasEnergy = false;
//                  break;
//          }
#endif
                ax = particles->ax[i];
#if DIM > 1
                ay = particles->ay[i];
#if DIM == 3
                az = particles->az[i];
#endif
#endif
                temp = ax * ax;
#if DIM > 1
                temp += ay * ay;
#if DIM == 3
                temp += az * az;
#endif
#endif

                //if (i % 10000 == 0) {
                //    printf("i: %i ax = %e, ay = %e, az = %e\n", i, ax, ay, az);
                //}

                sml = particles->sml[i];
                temp = cuda::math::sqrt(sml / cuda::math::sqrt(temp));
                forces = cuda::math::min(forces, temp);
                temp = sml / particles->cs[i];
                courant = cuda::math::min(courant, temp);

                temp = COURANT_FACT * sml / (particles->cs[i] + 1.2 * materials[matId].artificialViscosity.alpha * particles->cs[i] +
                            materials[matId].artificialViscosity.beta * particles->muijmax[i]);
                dtartvisc = min(dtartvisc, temp);

#if DIM == 1
                temp = cuda::math::sqrt(particles->vx[i] * particles->vx[i]);
#elif DIM == 2
                temp = cuda::math::sqrt(particles->vx[i] * particles->vx[i] +
                                        particles->vy[i] * particles->vy[i]);
#else
                temp = cuda::math::sqrt(particles->vx[i] * particles->vx[i] +
                                        particles->vy[i] * particles->vy[i] +
                                        particles->vz[i] * particles->vz[i]);
#endif
                //if (i % 10000 == 0) {
                //    printf("i: %i vx = %e, vy = %e, vz = %e\n", i, particles->vx[i], particles->vy[i], particles->vz[i]);
                //}

                vmax = cuda::math::max(temp, vmax);

#if INTEGRATE_DENSITY
                if (particles->drhodt[i] != 0) {
                    temp = SAFETY_FIRST * (cuda::math::abs(particles->rho[i])+rhomin_d)/cuda::math::abs(particles->drhodt[i]);
                    dtrho = cuda::math::min(temp, dtrho);
                }
#endif
#if INTEGRATE_ENERGY
                if (particles->dedt[i] != 0 && hasEnergy) {
                    //TODO: define emin_d
                    //temp = SAFETY_FIRST * (cuda::math::abs(particles->e[i])+emin_d)/cuda::math::abs(particles->dedt[i]);
                    dte = cuda::math::min(temp, dte);
                }
#endif

            }

            __threadfence();

            i = threadIdx.x;
            sharedForces[i] = forces;
            sharedCourant[i] = courant;
            sharede[i] = dte;
            sharedrho[i] = dtrho;
            sharedArtVisc[i] = dtartvisc;
            sharedVmax[i] = vmax;

            for (j = NUM_THREADS_LIMIT_TIME_STEP / 2; j > 0; j /= 2) {
                __syncthreads();
                if (i < j) {
                    k = i + j;
                    sharedForces[i] = forces = cuda::math::min(forces, sharedForces[k]);
                    sharedCourant[i] = courant = cuda::math::min(courant, sharedCourant[k]);
                    sharede[i] = dte = cuda::math::min(dte, sharede[k]);
                    sharedrho[i] = dtrho = cuda::math::min(dtrho, sharedrho[k]);
                    sharedArtVisc[i] = dtartvisc = cuda::math::min(dtartvisc, sharedArtVisc[k]);
                    sharedVmax[i] = vmax = cuda::math::max(vmax, sharedVmax[k]);
                }
            }
            // write block result to global memory
            if (i == 0) {
                k = blockIdx.x;
                blockShared->forces[k] = forces;
                blockShared->courant[k] = courant;
                blockShared->e[k] = dte;
                blockShared->rho[k] = dtrho;
                blockShared->artVisc[k] = dtartvisc;
                blockShared->vmax[k] = vmax;


                m = gridDim.x - 1;
                if (m == atomicInc((unsigned int *)blockCount, m)) {
                    // last block, so combine all block results
                    for (j = 0; j <= m; j++) {
                        forces = cuda::math::min(forces, blockShared->forces[j]);
                        courant = cuda::math::min(courant, blockShared->courant[j]);
                        dte = cuda::math::min(dte, blockShared->e[j]);
                        dtrho = cuda::math::min(dtrho, blockShared->rho[j]);
                        dtartvisc = cuda::math::min(dtartvisc, blockShared->artVisc[j]);
                        vmax = cuda::math::min(vmax, blockShared->vmax[j]);
                    }
                    // set new timestep
                    *simulationTime->dt = dtx = cuda::math::min(COURANT_FACT*courant, FORCES_FACT*forces);
                    //printf("courant: dt = %e (courant = %e)\n", COURANT_FACT*courant, courant);
                    //printf("force  : dt = %e (forces = %e)\n", FORCES_FACT*forces, forces);

                    if (vmax > 0. && searchRadius > 0.) { // TODO: searchRadius = 0 for 1 process
                        *simulationTime->dt = cuda::math::min(*simulationTime->dt, searchRadius / (2 * vmax));
                        //printf("search : dt = %e (vmax = %e)\n", searchRadius / (2 * vmax), vmax);
                    }
#if INTEGRATE_ENERGY
                    *simulationTime->dt = cuda::math::min(*simulationTime->dt, dte);
#endif
#if INTEGRATE_DENSITY
                    dt = cuda::math::min(dt, dtrho);
#endif

                    *simulationTime->dt = cuda::math::min(*simulationTime->dt, dtartvisc);
                    //printf("viscos : dt = %e\n", dtartvisc);

                    *simulationTime->dt = cuda::math::min(*simulationTime->dt, *simulationTime->endTime - *simulationTime->currentTime);
                    if (*simulationTime->dt > *simulationTime->dt_max) {
                        *simulationTime->dt = *simulationTime->dt_max;
                    }
                    //printf("max    : dt = %e\n", *simulationTime->dt_max);

                    //printf("Time Step Information: dt(v and x): %.17e dtS: %.17e dte: %.17e dtrho: %.17e dtdamage: %.17e dtalpha: %.17e dtalpha_epspor: %.17e dtepsilon_v: %.17e\n", dtx, dtS, dte, dtrho, dtdamage, dtalpha, dtalpha_epspor, dtepsilon_v);
                    //printf("time: %.17e timestep set to %.17e, integrating until %.17e \n", currentTimeD, dt, endTimeD);

                    // reset block count
                    *blockCount = 0;
                }
            }
        }

        real Launch::corrector(Particles *particles, IntegratedParticles *predictor, real dt, int numParticles) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::PredictorCorrectorEulerNS::Kernel::corrector, particles,
                                predictor, dt, numParticles);
        }
        real Launch::predictor(Particles *particles, IntegratedParticles *predictor, real dt, int numParticles) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::PredictorCorrectorEulerNS::Kernel::predictor, particles,
                                predictor, dt, numParticles);
        }

        real Launch::setTimeStep(int multiProcessorCount, SimulationTime *simulationTime, Material *materials, Particles *particles,
                                 BlockShared *blockShared, int *blockCount, real searchRadius, int numParticles) {
            ExecutionPolicy executionPolicy(multiProcessorCount, 256);
            return cuda::launch(true, executionPolicy, ::PredictorCorrectorEulerNS::Kernel::setTimeStep, simulationTime,
                                materials, particles, blockShared, blockCount, searchRadius, numParticles);
        }

        real Launch::pressureChangeCheck() {

        }

    }
}
