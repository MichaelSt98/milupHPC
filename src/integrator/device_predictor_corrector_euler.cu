#include "../../include/integrator/device_predictor_corrector_euler.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

namespace PredictorCorrectorEulerNS {

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
                //if (i % 1000 == 0) {
                //    printf("x[%i] = %f + %f/2 * (%f + %f)\n", i, particles->x[i], dt, predictor->dxdt[i],
                //           particles->vx[i]);
                //}
                particles->vx[i] = particles->vx[i] + dt/2 * (predictor->ax[i] + particles->ax[i] + 2 * particles->g_ax[i]);
                particles->ax[i] = 0.5 * (predictor->ax[i] + particles->ax[i]) + particles->g_ax[i];
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

                predictor->reset(i); //TODO: move somewhere else?

// TODO: some SPH flag?
#if INTEGRATE_DENSITY
                particles->rho[i] = particles->rho[i] + dt/2 * (predictor->drhodt[i] + particles->drhodt[i]);
                particles->drhodt[i] = 0.5 * (predictor->drhodt[i] + particles->drhodt[i]);
#else
                //p.rho[i] = p.rho[i];
#endif
#if INTEGRATE_ENERGY
                particles->e[i] = particles->e[i] + dt/2 * (predictor->dedt[i] + particles->dedt[i]);
                particles->dedt[i] = 0.5 * (predictor->dedt[i] + particles->dedt[i]);
#endif
#if INTEGRATE_SML
                particles->sml[i] = particles->sml[i] + dt/2 * (predictor->dsmldt[i] + particles->dsmldt[i]);
                particles->dsmldt[i] = 0.5 * (predictor->dsmldt[i] + particles->dsmldt[i]);
#else
                particles->sml[i] = predictor->sml[i];
#endif
            }
        }

        __global__ void predictor(Particles *particles, IntegratedParticles *predictor, real dt, int numParticles) {

            int i;

            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i+= blockDim.x * gridDim.x) {

                predictor->x[i] = particles->x[i] + dt * particles->vx[i];
                predictor->vx[i] = particles->vx[i] + dt * (particles->ax[i] + particles->g_ax[i]);
                //predictor->dvxdt[i] = particles->ax[i];
#if DIM > 1
                predictor->y[i] = particles->y[i] + dt * particles->vy[i];
                predictor->vy[i] = particles->vy[i] + dt * (particles->ay[i] + particles->g_ay[i]);
                //predictor->dvydt[i] = particles->ay[i];
#if DIM == 3
                predictor->z[i] = particles->z[i] + dt * particles->vz[i];
                predictor->vz[i] = particles->vz[i] + dt * (particles->az[i] + particles->g_az[i]);
                //predictor->dvzdt[i] = particles->az[i];
#endif
#endif

// TODO: some SPH flag?
#if INTEGRATE_DENSITY
                predictor->rho[i] = particles->rho[i] + dt * particles->drhodt[i];
                predictor->drhodt[i] = particles->drhodt[i];
#else
                predictor->rho[i] = particles->rho[i];
#endif
#if INTEGRATE_ENERGY
                predictor->e[i] = particles->e[i] + dt * particles->dedt[i];
#endif
#if INTEGRATE_SML
                predictor->sml[i] = particles->sml[i] + dt * particles->dsmldt[i];
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
        __global__ void setTimeStep() {

        }

        __global__ void pressureChangeCheck() {

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

        real Launch::setTimeStep() {

        }

        real Launch::pressureChangeCheck() {

        }

    }
}
