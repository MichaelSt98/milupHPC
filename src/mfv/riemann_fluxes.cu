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

        __device__ void frameVelocity(real vFrame[DIM], int i, int ip, Particles *particles){
        vFrame[0] = (particles->vx[i] + particles->vx[ip])/2.;
#if DIM > 1
        vFrame[1] = (particles->vy[i] + particles->vy[ip])/2.;
#if DIM == 3
        vFrame[2] = (particles->vz[i] + particles->vz[ip])/2.;
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

        __device__ void gradient(real *grad, real *f, int i, int *interactions, int noi, Particles *particles,
                                 SlopeLimitingParameters *slopeLimitingParameters){
            int d, j, ip;
            real fMax, fMin, beta, Ncrit2cond, absGrad;
#pragma unroll
            for(d=0; d<DIM; d++){
                *(grad+d) = 0.;
            }

            for (j = 0; j < noi; j++) {
                ip = interactions[i * MAX_NUM_INTERACTIONS + j];

                *grad += (f[ip] - f[i]) * particles->psix[ip];
#if DIM > 1
                *(grad+1) += (f[ip] - f[i]) * particles->psiy[ip];
#if DIM == 3
                *(grad+2) += (f[ip] - f[i]) * particles->psiz[ip];
#endif
#endif
            }
//#if SLOPE_LIMITING
            // limit slopes by default
            // find maximum value of f amongst all neighbors including the particle itself
            fMax = f[i];
            fMin = f[i];

            for (j = 0; j < noi; j++){
                ip = interactions[i * MAX_NUM_INTERACTIONS + j];

                if (f[ip] > fMax){
                    fMax = f[ip];
                } else if (f[ip] < fMin){
                    fMin = f[ip];
                }
            }

            // compute "trustworthyness" parameter of the gradient beta_i
            Ncrit2cond = slopeLimitingParameters->critCondNum/particles->Ncond[i];
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
            absGrad = 0.;
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
                *(grad+d) *= alpha;
            }
//#end
        }
    }

    namespace Kernel {

        __global__ void computeGradients(Particles *particles, int *interactions, int numParticles,
                                         SlopeLimitingParameters *slopeLimitingParameters){

            int i, inc, noi, iGrad;

            /// main loop over particles
            inc = blockDim.x * gridDim.x;
            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {

                noi = particles->noi[i];
                iGrad = i*DIM;
                /// estimate gradients of particle i
                ::MFV::Compute::gradient(&particles->rhoGrad[iGrad], particles->rho, i, interactions, noi, particles,
                                         slopeLimitingParameters);
                ::MFV::Compute::gradient(&particles->vxGrad[iGrad], particles->vx, i, interactions, noi, particles,
                                         slopeLimitingParameters);
#if DIM > 1
                ::MFV::Compute::gradient(&particles->vyGrad[iGrad], particles->vy, i, interactions, noi, particles,
                                         slopeLimitingParameters);
#if DIM == 3
                ::MFV::Compute::gradient(&particles->vzGrad[iGrad], particles->vz, i, interactions, noi, particles,
                                         slopeLimitingParameters);
#endif
#endif
                ::MFV::Compute::gradient(&particles->pGrad[iGrad], particles->p, i, interactions, noi, particles,
                                         slopeLimitingParameters);
            }

        }

        __global__ void riemannFluxes(Particles *particles, RiemannSolver riemannSolver, int *interactions,
                                      int numParticles, SlopeLimitingParameters *slopeLimitingParameters,
                                      real *dt, Material *materials){
            int i, j, inc, ip, noi, d, iGradR, iGradL;
            real vFrame[DIM]; // frame velocity for Riemann problem
            real xij[DIM]; // quadrature point between particles i and j
            real Aij[DIM], AijNorm, hatAij[DIM]; // effective face of the interface i -> j
            /// container variables for vectors of primitive variables W_R and W_L
            real rhoR, rhoL, vR[DIM], vL[DIM], pR, pL;
            real unitX[DIM], R[DIM*DIM], vBuf[DIM]; // helpers for rotation of states
            unitX[0] = 1.;
#if DIM > 1
            unitX[1] = 0.;
#if DIM == 3
            unitX[2] = 0.;
#endif
#endif
            real viDiv, vjDiv; // velocity divergences for forward prediction in time
            real gamma; // adiabatic index

#if PAIRWISE_LIMITER
            real rhoR0, rhoL0, vxR0, vxL0, pR0, pL0;
#if DIM > 1
            real vyR0, vyL0;
#if DIM == 3
            real vzR0, vzL0;
#endif
#endif
#endif // PAIRWISE_LIMITER

            /// main loop over particles
            inc = blockDim.x * gridDim.x;
            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {

                noi = particles->noi[i];

                /// loop over nearest neighbors
                for (j = 0; j < noi; j++) {
                    ip = interactions[i * MAX_NUM_INTERACTIONS + j];

                    // computing frame velocity
                    ::MFV::Compute::frameVelocity(vFrame, i, ip, particles);

                    /// constructing vector of primitive variables for the Riemann problem
                    rhoR = particles->rho[i];
                    rhoL = particles->rho[ip];

                    /// boost velocities to moving frame of reference
                    vR[0] = particles->vx[i] - vFrame[0];
                    vL[0] = particles->vx[ip] - vFrame[0];
#if DIM > 1
                    vR[1]  = particles->vy[i] - vFrame[1];
                    vL[1] = particles->vy[ip] - vFrame[1];
#if DIM == 3
                    vR[2]  = particles->vz[i] - vFrame[2];
                    vL[2] = particles->vz[ip] - vFrame[2];
#endif
#endif
                    pR = particles->p[i];
                    pL = particles->p[ip];

#if PAIRWISE_LIMITER
                    // store original values boosted to moving frame
                    rhoR0 = rhoR, rhoL0 = rhoL;
                    vxR0 = vR[0], vxL0 = vL[0];
#if DIM > 1
                    vyR0 = vR[1], vyL0 = vL[1];
#if DIM == 3
                    vzR0 = vR[2], vzL0 = vL[2];
#endif
#endif
                    pR0 = pR, pL0 = pL;
#endif // PAIRWISE_LIMITER

                    /// reconstruct vector or primitive variables at effective face between particles

                    ::MFV::Compute::quadraturePoint(xij, i, ip, particles);

                    real deltaXijR[DIM], deltaXijL[DIM];

                    deltaXijR[0] = xij[0] - particles->x[i];
                    deltaXijL[0] = xij[0] - particles->x[ip];
#if DIM > 1
                    deltaXijR[1] = xij[1] - particles->y[i];
                    deltaXijL[1] = xij[1] - particles->y[ip];
#if DIM == 3
                    deltaXijR[2] = xij[2] - particles->z[i];
                    deltaXijL[2] = xij[2] - particles->z[ip];
#endif
#endif

                    iGradR = i*DIM;
                    iGradL = ip*DIM;

                    rhoR += CudaUtils::dotProd(&particles->rhoGrad[iGradR], deltaXijR);
                    rhoL += CudaUtils::dotProd(&particles->rhoGrad[iGradL], deltaXijL);

                    vR[0] += CudaUtils::dotProd(&particles->vxGrad[iGradR], deltaXijR);
                    vL[0] += CudaUtils::dotProd(&particles->vxGrad[iGradL], deltaXijL);
#if DIM > 1
                    vR[1] += CudaUtils::dotProd(&particles->vyGrad[iGradR], deltaXijR);
                    vL[1] += CudaUtils::dotProd(&particles->vyGrad[iGradL], deltaXijL);
#if DIM == 3
                    vR[2] += CudaUtils::dotProd(&particles->vzGrad[iGradR], deltaXijR);
                    vL[2] += CudaUtils::dotProd(&particles->vzGrad[iGradL], deltaXijL);
#endif
#endif
                    pR += CudaUtils::dotProd(&particles->pGrad[iGradR], deltaXijR);
                    pL += CudaUtils::dotProd(&particles->pGrad[iGradL], deltaXijL);


                    /// forward predict quantities by half a timestep

                    // computing velocity divergence for particles i and j
                    viDiv = particles->vxGrad[iGradR];
                    vjDiv = particles->vxGrad[iGradL];
#if DIM > 1
                    viDiv += particles->vyGrad[iGradR+1];
                    vjDiv += particles->vyGrad[iGradL+1];
#if DIM == 3
                    viDiv += particles->vzGrad[iGradR+2];
                    vjDiv += particles->vzGrad[iGradL+2];
#endif
#endif
                    // get adiabatic index of material
                    // TODO: this only works for an ideal gas
                    gamma = materials[particles->materialId[i]].eos.polytropic_gamma;

                    // actual forward prediction dimension by dimension
                    rhoR -= *dt/2. * (particles->rho[i]*viDiv + (particles->vx[i]-vFrame[0])*particles->rhoGrad[iGradR]);
                    rhoL -= *dt/2. * (particles->rho[ip]*vjDiv + (particles->vx[ip]-vFrame[0])*particles->rhoGrad[iGradL]);

                    vR[0] -= *dt/2. * (particles->pGrad[iGradL]/particles->rho[i] + (particles->vx[i]-vFrame[0])*particles->vxGrad[iGradR]);
                    vL[0] -= *dt/2. * (particles->pGrad[iGradL]/particles->rho[ip] + (particles->vx[ip]-vFrame[0])*particles->vxGrad[iGradL]);

                    pR -= *dt/2. * (gamma*particles->p[i] * viDiv + (particles->vx[i]-vFrame[0])*particles->pGrad[iGradR]);
                    pL -= *dt/2. * (gamma*particles->p[ip] * vjDiv + (particles->vx[ip]-vFrame[0])*particles->pGrad[iGradL]);

#if DIM > 1
                    rhoR -= *dt/2. * (particles->vy[i]-vFrame[1])*particles->rhoGrad[iGradR+1];
                    rhoL -= *dt/2. * (particles->vy[ip]-vFrame[1])*particles->rhoGrad[iGradL+1];

                    vR[0] -= *dt/2. * (particles->vy[i]-vFrame[1])*particles->vxGrad[iGradR+1];
                    vL[0] -= *dt/2. * (particles->vy[ip]-vFrame[1])*particles->vxGrad[iGradL+1];

                    vR[1] -= *dt/2. * (particles->pGrad[iGradR+1]/particles->rho[i] + (particles->vx[i]-vFrame[0])*particles->vyGrad[iGradR]
                            + (particles->vy[i]-vFrame[1])*particles->vyGrad[iGradR+1]);
                    vL[1] -= *dt/2. * (particles->pGrad[iGradL+1]/particles->rho[ip] + (particles->vx[ip]-vFrame[0])*particles->vyGrad[iGradL]
                            + (particles->vy[ip]-vFrame[1])*particles->vyGrad[iGradL+1]);

                    pR -= *dt/2. * (particles->vy[i]-vFrame[1])*particles->pGrad[iGradR+1];
                    pL -= *dt/2. * (particles->vy[ip]-vFrame[1])*particles->pGrad[iGradL+1];

#if DIM == 3
                    rhoR -= *dt/2. * (particles->vz[i]-vFrame[2])*particles->rhoGrad[iGradR+2];
                    rhoL -= *dt/2. * (particles->vz[ip]-vFrame[2])*particles->rhoGrad[iGradL+2];

                    vR[0] -= *dt/2. * (particles->vz[i]-vFrame[2])*particles->vxGrad[iGradR+2];
                    vL[0] -= *dt/2. * (particles->vz[ip]-vFrame[2])*particles->vxGrad[iGradL+2];

                    vR[1] -= *dt/2. * (particles->vz[i]-vFrame[2])*particles->vyGrad[iGradR+2];
                    vL[1] -= *dt/2. * (particles->vz[ip]-vFrame[2])*particles->vyGrad[iGradL+2];

                    vR[2] -= *dt/2. * (particles->pGrad[iGradR+2]/particles->rho[i] + (particles->vx[i]-vFrame[0])*particles->vzGrad[iGradR]
                                     + (particles->vy[i]-vFrame[1])*particles->vzGrad[iGradR+1]
                                     + (particles->vz[i]-vFrame[2])*particles->vzGrad[iGradR+2]);
                    vL[2] -= *dt/2. * (particles->pGrad[iGradL+2]/particles->rho[ip] + (particles->vx[ip]-vFrame[0])*particles->vzGrad[iGradL]
                                     + (particles->vy[ip]-vFrame[1])*particles->vzGrad[iGradL+1]
                                     + (particles->vz[ip]-vFrame[2])*particles->vzGrad[iGradL+2]);

                    pR -= *dt/2. * (particles->vz[i]-vFrame[2])*particles->pGrad[iGradR+2];
                    pL -= *dt/2. * (particles->vz[ip]-vFrame[2])*particles->pGrad[iGradL+2];
#endif
#endif

                    /// rotate to effective face
                    ::MFV::Compute::effectiveFace(Aij, i, ip, interactions, particles);
                    AijNorm = sqrt(::CudaUtils::dotProd(Aij, Aij));
#pragma unroll
                    for(d=0; d<DIM; d++){
                        hatAij[d] = AijNorm*Aij[d];
                    }

                    ::CudaUtils::rotationMatrix(R, hatAij, unitX);

                    // rotate right state
#pragma unroll
                    for(d=0; d<DIM; d++){
                        vBuf[d] = vR[d];
                    }
                    ::CudaUtils::multiplyMatVec(vR, R, vBuf);

                    //rotate left state
#pragma unroll
                    for(d=0; d<DIM; d++){
                        vBuf[d] = vL[d];
                    }
                    ::CudaUtils::multiplyMatVec(vL, R, vBuf);

                }

                // TODO: RIEMANN SOLVER function call

            }
        }

        namespace Launch {
            real computeGradients(Particles *particles, int *interactions, int numParticles,
                                  SlopeLimitingParameters *slopeLimitingParameters) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::MFV::Kernel::computeGradients, particles,
                                    interactions, numParticles, slopeLimitingParameters);
            }

            real riemannFluxes(Particles *particles, RiemannSolver riemannSolver, int *interactions, int numParticles,
                               SlopeLimitingParameters *slopeLimitingParameters, real *dt, Material *materials) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::MFV::Kernel::riemannFluxes, particles, riemannSolver,
                                    interactions, numParticles, slopeLimitingParameters, dt, materials);
            }
        }
    }
}