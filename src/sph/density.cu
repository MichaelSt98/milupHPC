#include "../../include/sph/density.cuh"
#include "../include/cuda_utils/cuda_launcher.cuh"

namespace SPH {

    namespace Kernel {

        __global__ void calculateDensity(::SPH::SPH_kernel kernel, Tree *tree, Particles *particles, int *interactions, int numParticles) {

            int i;
            int j;
            int inc;
            int ip;
            int d;
            real W, Wj, dx[DIM], dWdx[DIM], dWdr;
            real rho, sml;
            real x;
#if DIM > 1
            real y;
#if DIM == 3
            real z;
#endif
#endif

            inc = blockDim.x * gridDim.x;
            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {

                x = particles->x[i];
#if DIM > 1
                y = particles->y[i];
#if DIM == 3
                z = particles->z[i];
#endif
#endif

                sml = particles->sml[i];

                #pragma unroll
                for (d = 0; d < DIM; d++) {
                    dx[d] = 0;
                }

                kernel(&W, dWdx, &dWdr, dx, sml);
                // "self-density"
                rho = particles->mass[i] * W;

                // sph sum for particle i
                for (j = 0; j < particles->noi[i]; j++) {
                    ip = interactions[i * MAX_NUM_INTERACTIONS + j];

#if (VARIABLE_SML || INTEGRATE_SML)
                    sml = 0.5 * (particles->sml[i] + particles->sml[ip]);
#endif

                    dx[0] = x - particles->x[ip];
#if DIM > 1
                    dx[1] = y - particles->y[ip];
#if DIM > 2
                    dx[2] = z - particles->z[ip];
#endif
#endif

                    kernel(&W, dWdx, &dWdr, dx, sml);
                    rho += particles->mass[ip] * W;
                }

                particles->rho[i] = rho;

                if (particles->rho[i] <= 0.) {
                    cudaTerminate("negative or zero rho! rho[%i] = %e\n", i, particles->rho[i]);
                }
            }
        }

        real Launch::calculateDensity(::SPH::SPH_kernel kernel, Tree *tree, Particles *particles, int *interactions, int numParticles) {
            //ExecutionPolicy executionPolicy(numParticles, ::SPH::Kernel::calculateDensity, kernel, tree, particles,
            //                                interactions, numParticles);
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SPH::Kernel::calculateDensity, kernel, tree, particles, interactions, numParticles);
        }

    }
}
