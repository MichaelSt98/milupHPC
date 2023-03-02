#include "../../include/mfv/riemann_fluxes.cuh"
#include "../include/cuda_utils/cuda_launcher.cuh"

namespace MFV {

    CUDA_CALLABLE_MEMBER SlopeLimitingParameters::SlopeLimitingParameters(){}

    CUDA_CALLABLE_MEMBER SlopeLimitingParameters::SlopeLimitingParameters(real critCondNum, real betaMin, real betaMax
#if PAIRWISE_LIMITER
                                                                          ,real psi1, real psi2
#endif
    ) : critCondNum(critCondNum), betaMin(betaMin), betaMax(betaMax)

#if PAIRWISE_LIMITER
    , pis1(psi1), psi2(psi2)
#endif
    {}

    namespace Compute {

        __device__ void quadraturePoint(real x_ij[DIM], int i, int ip, Particles *particles){
            x_ij[0] = (particles->x[i] + particles->x[ip])/2.;
#if DIM > 1
            x_ij[1] = (particles->y[i] + particles->y[ip])/2.;
#if DIM == 3
            x_ij[2] = (particles->z[i] + particles->z[ip])/2.;
#endif
#endif
        }

        __device__ void effectiveFace(real Aij[DIM], int i, int ip, int *interactions, Particles *particles){
            // search neighbor i in interactions[] of ip
            int d, ij;
            for(ij=0; ij<particles->noi[ip]; ij++){
                if (interactions[ij+ip*MAX_NUM_INTERACTIONS] == i) break;
            }

            Aij[0] = 1./particles->omega[i]*particles->psix[ip]
                    - 1./particles->omega[ip]*particles->psix[ij+ip*MAX_NUM_INTERACTIONS];
#if DIM > 1
            Aij[1] = 1./particles->omega[i]*particles->psiy[ip]
                    - 1./particles->omega[ip]*particles->psiy[ij+ip*MAX_NUM_INTERACTIONS];
#if DIM ==3
            Aij[0] = 1./particles->omega[i]*particles->psiz[ip]
                    - 1./particles->omega[ip]*particles->psiz[ij+ip*MAX_NUM_INTERACTIONS];
#endif
#endif
        }

        __device__ void gradient(real grad[DIM], real *f, int i, int *interactions, int noi, Particles *particles,
                                 SlopeLimitingParameters *slopeLimitingParameters){
            int d, j, ip;
#pragma unroll
            for(d=0; d<DIM; d++){
                grad[d] = 0.;
            }

            for (j = 0; j < noi; j++) {
                ip = interactions[i * MAX_NUM_INTERACTIONS + j];

                grad[0] += (f[ip] - f[i]) * particles->psix[ip];
#if DIM > 1
                grad[1] += (f[ip] - f[i]) * particles->psiy[ip];
#if DIM == 3
                grad[2] += (f[ip] - f[i]) * particles->psiz[ip];
#endif
#endif
            }
//#if SLOPE_LIMITING
            // limit slopes by default
            // find maximum value of f amongst all neighbors including the particle itself
            real fMax = f[i];
            real fMin = f[i];

            for (j = 0; j < noi; j++){
                ip = interactions[i * MAX_NUM_INTERACTIONS + j];

                if (f[ip] > fMax){
                    fMax = f[ip];
                } else if (f[ip] < fMin){
                    fMin = f[ip];
                }
            }

            // compute "trustworthyness" parameter of the gradient beta_i
            real beta;
            real Ncrit2cond = slopeLimitingParameters->critCondNum/particles->Ncond[i];
            if (Ncrit2cond < 1.){
                real betaCandidate = Ncrit2cond*slopeLimitingParameters->betaMax;
                if(betaCandidate>slopeLimitingParameters->betaMin){
                    beta = betaCandidate;
                } else {
                    beta = slopeLimitingParameters->betaMin;
                }
            } else {
                beta = slopeLimitingParameters->betaMax;
            }

            // computing length of gradient
            real absGrad = 0.;
#pragma unroll
            for(d=0; d<DIM; d++){
                absGrad += grad[d]*grad[d];
            }
            absGrad = sqrt(absGrad);

            // actually limit gradients
            real fDelta = absGrad*particles->sml[i]/2.;
            real alphaMinCandidate = (f[i] - fMin)/fDelta;
            real alphaMaxCandidate = (fMax - f[i])/fDelta;

            real alpha = alphaMinCandidate < alphaMaxCandidate ? alphaMinCandidate : alphaMaxCandidate;
            alpha = alpha < 1. ? alpha : 1.;

#pragma unroll
            for(d=0; d<DIM; d++){
                grad[d] *= alpha;
            }
//#end
        }
    }

    namespace Kernel {

        __global__ void riemannFluxes(Particles *particles, RiemannSolver riemannSolver, int *interactions,
                                      int numParticles, SlopeLimitingParameters *slopeLimitingParameters){
            int i, j, inc, ip, noi;
            real x, rho, vx, P, vFrame[DIM], rhoGrad[DIM], vxGrad[DIM], pGrad[DIM]; // containers for particle i quantities
            real Aij[DIM]; // effective face of the interface i -> j
#if DIM > 1
            real y, vyGrad[DIM];
#if DIM == 3
            real z, vzGrad[DIM];
#endif
#endif
            /// main loop over particles
            inc = blockDim.x * gridDim.x;
            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
                x = particles->x[i];
#if DIM > 1
                y = particles->y[i];
#if DIM == 3
                z = particles->z[i];
#endif
#endif
                noi = particles->noi[i];

                /// estimate gradients of particle i
                ::MFV::Compute::gradient(rhoGrad, particles->rho, i, interactions, noi, particles, slopeLimitingParameters);
                ::MFV::Compute::gradient(vxGrad, particles->vx, i, interactions, noi, particles, slopeLimitingParameters);
#if DIM > 1
                ::MFV::Compute::gradient(vyGrad, particles->vy, i, interactions, noi, particles, slopeLimitingParameters);
#if DIM == 3
                ::MFV::Compute::gradient(vzGrad, particles->vz, i, interactions, noi, particles, slopeLimitingParameters);
#endif
#endif
                ::MFV::Compute::gradient(pGrad, particles->p, i, interactions, noi, particles, slopeLimitingParameters);

                /// loop over nearest neighbors
                for (j = 0; j < noi; j++) {
                    ip = interactions[i * MAX_NUM_INTERACTIONS + j];

                    ::MFV::Compute::effectiveFace(Aij, i, ip, interactions, particles);



                }


            }
        }

        namespace Launch {
            real riemannFluxes(Particles *particles, RiemannSolver riemannSolver, int *interactions, int numParticles,
                               SlopeLimitingParameters *slopeLimitingParameters) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::MFV::Kernel::riemannFluxes, particles, riemannSolver,
                                    interactions, numParticles, slopeLimitingParameters);
            }
        }
    }
}