#include "../../include/mfv/volume_partition.cuh"
#include "../include/cuda_utils/cuda_launcher.cuh"

namespace MFV {

    namespace Kernel {

        __global__ void calculateDensity(::SPH::SPH_kernel kernel, Particles *particles, int *interactions, int numParticles) {

            int i, j, inc, ip, d, noi;
            real W, Wj, dx[DIM], dWdx[DIM], dWdr;
            real sml, omg;
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
                // "self-inverse-volume"
                omg = W;

                noi = particles->noi[i];

                // sph sum for particle i
                for (j = 0; j < noi; j++) {

                    ip = interactions[i * MAX_NUM_INTERACTIONS + j];

#if VARIABLE_SML
                    //TODO: check if this works
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
                    omg += W;
                }

                particles->omega[i] = omg;
                particles->rho[i] = particles->mass[i]*omg;

                if (particles->rho[i] <= 0.) {
                    cudaTerminate("negative or zero rho! rho[%i] = %e\n", i, particles->rho[i]);
                }
            }
        }

        real Launch::calculateDensity(::SPH::SPH_kernel kernel, Particles *particles, int *interactions, int numParticles) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::MFV::Kernel::calculateDensity, kernel, particles, interactions, numParticles);
        }

        __global__ void computeVectorWeights(::SPH::SPH_kernel kernel, Particles *particles, int *interactions,
                                             int numParticles, real *critCondNum){
            int i, j, inc, ip, d, alpha, beta, noi;
            real W, Wj, dx[DIM], dWdx[DIM], dWdr, E[DIM*DIM];
            real sml, omg;
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
                omg = particles->omega[i];

                noi = particles->noi[i];

#pragma unroll
                for(d=0; d < DIM*DIM; d++){
                    E[d] = 0.;
                }

                // construct matrix E_i
                for (j = 0; j < noi; j++) {

                    ip = interactions[i * MAX_NUM_INTERACTIONS + j];

#if VARIABLE_SML
                    //TODO: check if this works
                    sml = 0.5 * (particles->sml[i] + particles->sml[ip]);
#endif

                    dx[0] = particles->x[ip] - x;
#if DIM > 1
                    dx[1] = particles->y[ip] - y;
#if DIM > 2
                    dx[2] = particles->z[ip] - z;
#endif
#endif

                    kernel(&W, dWdx, &dWdr, dx, sml);

#pragma unroll
                    for (alpha=0; alpha<DIM; alpha++){
#pragma unroll
                        for (beta=0; beta<DIM; beta++){
                            E[DIM*alpha+beta] += dx[alpha]*dx[beta]*W/omg;
                        }
                    }
                }

                real B[DIM*DIM];
                if (CudaUtils::invertMatrix(E, B) < 1){

                    printf("ERROR: Matrix E_%i is not invertible (det(E) = 0.).\n");
                    particles->Ncond[i] = DBL_MAX;

                } else {
                    real normE=0., normB=0.;
#pragma unroll
                    for (alpha=0; alpha<DIM; alpha++){
#pragma unroll
                        for (beta=0; beta<DIM; beta++){
                            normE += abs(E[DIM*alpha+beta])*abs(E[DIM*alpha+beta]);
                            normB += abs(B[DIM*alpha+beta])*abs(B[DIM*alpha+beta]);
                        }
                    }
                    particles->Ncond[i] = 1./(real)DIM * sqrt(normB*normE);

                    if (particles->Ncond[i] > *critCondNum){
                        printf("WARNING: N_cond = %f > N_cond^crit = %i\n", particles->Ncond[i], *critCondNum);
                    }
                }

                // compute vector weights psi_j(x_i)
                for (j = 0; j < noi; j++) {

                    ip = interactions[i * MAX_NUM_INTERACTIONS + j];

#if VARIABLE_SML
                    //TODO: check if this works
                    sml = 0.5 * (particles->sml[i] + particles->sml[ip]);
#endif

                    dx[0] = particles->x[ip] - x;
#if DIM > 1
                    dx[1] = particles->y[ip] - y;
#if DIM > 2
                    dx[2] = particles->z[ip] - z;
#endif
#endif

                    kernel(&W, dWdx, &dWdr, dx, sml);

#pragma unroll
                    particles->psix[ip] = 0.;
#if DIM > 1
                    particles->psiy[ip] = 0.;
#if DIM > 2
                    particles->psiz[ip] = 0.;
#endif
#endif
#pragma unroll
                    for (beta=0; beta<DIM; beta++){
                        particles->psix[ip] += B[beta]*dx[beta]*W/omg;
#if DIM > 1
                        particles->psiy[ip] += B[DIM+beta]*dx[beta]*W/omg;
#if DIM > 2
                        particles->psiz[ip] += B[2*DIM+beta]*dx[beta]*W/omg;
#endif
#endif
                    }
                }
            }
        }

        real Launch::computeVectorWeights(::SPH::SPH_kernel kernel, Particles *particles, int *interactions, int numParticles, real *critCondNum) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::MFV::Kernel::computeVectorWeights, kernel, particles, interactions, numParticles, critCondNum);
        }

    }
}