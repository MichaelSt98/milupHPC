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
    , psi1(psi1), psi2(psi2)
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
#if !MFV_FIX_PARTICLES
            vFrame[0] = (particles->vx[i] + particles->vx[ip])/2.;
#if DIM > 1
            vFrame[1] = (particles->vy[i] + particles->vy[ip])/2.;
#if DIM == 3
            vFrame[2] = (particles->vz[i] + particles->vz[ip])/2.;
#endif
#endif
#else
            vFrame[0] = 0.;
#if DIM > 1
            vFrame[1] = 0.;
#if DIM == 3
            vFrame[2] = 0.;
#endif
#endif
#endif
        }

        __device__ void effectiveFace(real Aij[DIM], int i, int ji, int *interactions, Particles *particles){
            // search neighbor i in interactions[] of ip
            int d, ip ,ij;
            ip = interactions[ji];
            for(ij=0; ij<particles->noi[ip]; ij++){
                if (interactions[ij+ip*MAX_NUM_INTERACTIONS] == i) break;
            }

            Aij[0] = 1./particles->omega[i]*particles->psix[ji]
                    - 1./particles->omega[ip]*particles->psix[ij+ip*MAX_NUM_INTERACTIONS];
#if DIM > 1
            Aij[1] = 1./particles->omega[i]*particles->psiy[ji]
                    - 1./particles->omega[ip]*particles->psiy[ij+ip*MAX_NUM_INTERACTIONS];
#if DIM == 3
            Aij[2] = 1./particles->omega[i]*particles->psiz[ji]
                    - 1./particles->omega[ip]*particles->psiz[ij+ip*MAX_NUM_INTERACTIONS];
#endif
#endif
        }

        __device__ void gradient(real *grad, real *f, int i, int *interactions, int noi, Particles *particles,
                                 SlopeLimitingParameters *slopeLimitingParameters){
            int d, j, ji, ip;
            real fMax, fMin, beta, Ncrit2cond, absGrad;
#pragma unroll
            for(d=0; d<DIM; d++){
                grad[d] = 0.;
            }

            for (j = 0; j < noi; j++) {
                ji = i * MAX_NUM_INTERACTIONS + j;
                ip = interactions[ji];

                grad[0] += (f[ip] - f[i]) * particles->psix[ji];
#if DIM > 1
                grad[1] += (f[ip] - f[i]) * particles->psiy[ji];
#if DIM == 3
                grad[2] += (f[ip] - f[i]) * particles->psiz[ji];
#endif
#endif
            }

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
                grad[d] *= alpha;
            }
        }

#if PAIRWISE_LIMITER
        __device__ double pairwiseLimiter(real &phi0, real &phi_i, real &phi_j, real &xijxiAbs, real &xjxiAbs,
                                            SlopeLimitingParameters *slopeLimitingParameters){
            real phi_ = phi_i;

            /// calculate helper values
            real phi_ij = phi_i + xijxiAbs / xjxiAbs * (phi_j - phi_i);
            real phiMin, phiMax;
            if (phi_i < phi_j) {
                phiMin = phi_i;
                phiMax = phi_j;
            } else {
                phiMin = phi_j;
                phiMax = phi_i;
            }
            real delta1 = slopeLimitingParameters->psi1 * abs(phi_i - phi_j);
            real delta2 = slopeLimitingParameters->psi2 * abs(phi_i - phi_j);
            real phiMinus, phiPlus;
            if ((phiMax + delta1 >= 0. && phiMax >= 0.) || (phiMax + delta1 < 0. && phiMax < 0.)) {
                phiPlus = phiMax + delta1;
            } else {
                phiPlus = phiMax / (1. + delta1 / abs(phiMax));
            }
            if ((phiMin - delta1 >= 0. && phiMin >= 0.) || (phiMin - delta1 < 0. && phiMin < 0.)) {
                phiMinus = phiMin - delta1;
            } else {
                phiMinus = phiMin / (1. + delta1 / abs(phiMin));
            }

            /// actually compute the effective face limited value
            if (phi_i < phi_j) {
                real minPhiD2;
                if (phi_ij + delta2 < phi0){
                    minPhiD2 = phi_ij + delta2;
                } else {
                    minPhiD2 = phi0;
                }

                phi_ = phiMinus > minPhiD2 ? phiMinus : minPhiD2;

            } else if (phi_i > phi_j){
                real maxPhiD2;
                if (phi_ij - delta2 > phi0){
                    maxPhiD2 = phi_ij - delta2;
                } else {
                    maxPhiD2 = phi0;
                }

                phi_ = phiPlus < maxPhiD2 ? phiPlus : maxPhiD2;

            }
            return phi_;
        }
#endif // PAIRWISE_LIMITER
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

        __global__ void riemannFluxes(Particles *particles, int *interactions,
                                      int numParticles, SlopeLimitingParameters *slopeLimitingParameters,
                                      real *dt, Material *materials){
            int i, j, inc, ip, noi, d, iGradR, iGradL, flagLR;
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
            real rhoSol, vSol[DIM], pSol; // solution vector of Riemann problem
            real vLab2; // length of velocity solution squared in the laboratory frame

#if PAIRWISE_LIMITER
            real rhoR0, rhoL0, vxR0, vxL0, pR0, pL0;
            real xijxiAbs; // distance of quadrature point x_ij to particles location x_i;
            real xijxjAbs; // distance of quadrature point x_ij to particles location x_j;
            real xjxiAbs; // distance of interaction partner x_j to x_i
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
//                if (i == 0){
//                    printf("Particle %i interacts with %i neighbors:\n", i, noi);
//                }

                // get adiabatic index of material
                // TODO: this only works for an ideal gas
                gamma = materials[particles->materialId[i]].eos.polytropic_gamma;

                /// create Riemann Solver instance with appropriate gamma
                // TODO: implement the possibility to use other Riemann solvers, probably with preprocessor directives
                ExactRiemannSolver riemannSolver;
                riemannSolver.init(gamma);

                /// set fluxes to be collected over interaction partners to zero
                particles->massFlux[i] = 0.;
                particles->vxFlux[i] = 0.;
#if DIM > 1
                particles->vyFlux[i] = 0.;
#if DIM == 3
                particles->vzFlux[i] = 0.;
#endif
#endif
                particles->energyFlux[i] = 0.;

                /// loop over nearest neighbors
                for (j = 0; j < noi; j++) {
                    ip = interactions[i * MAX_NUM_INTERACTIONS + j];

                    /// computing frame velocity
                    ::MFV::Compute::frameVelocity(vFrame, i, ip, particles);

                    /// constructing vector of primitive variables for the Riemann problem
                    rhoR = particles->rho[i];
                    rhoL = particles->rho[ip];

                    /// boost velocities to moving frame of reference
                    vR[0] = particles->vx[i] - vFrame[0];
                    vL[0] = particles->vx[ip] - vFrame[0];
#if DIM > 1
                    vR[1] = particles->vy[i] - vFrame[1];
                    vL[1] = particles->vy[ip] - vFrame[1];
#if DIM == 3
                    vR[2] = particles->vz[i] - vFrame[2];
                    vL[2] = particles->vz[ip] - vFrame[2];
#endif
#endif
                    pR = particles->p[i];
                    pL = particles->p[ip];

                    /// computing quadrature point xij
                    ::MFV::Compute::quadraturePoint(xij, i, ip, particles);

#if PAIRWISE_LIMITER
                    // store original values boosted to moving frame
                    rhoR0 = rhoR, rhoL0 = rhoL;
                    vxR0 = vR[0], vxL0 = vL[0];
                    xijxiAbs = (xij[0] - particles->x[i])*(xij[0] - particles->x[i]);
                    xijxjAbs = (xij[0] - particles->x[ip])*(xij[0] - particles->x[ip]);
                    xjxiAbs = (particles->x[ip] - particles->x[i])*(particles->x[ip] - particles->x[i]);
#if DIM > 1
                    vyR0 = vR[1], vyL0 = vL[1];
                    xijxiAbs += (xij[1] - particles->y[i])*(xij[1] - particles->y[i]);
                    xijxjAbs += (xij[1] - particles->y[ip])*(xij[1] - particles->y[ip]);
                    xjxiAbs += (particles->y[ip] - particles->y[i])*(particles->y[ip] - particles->y[i]);
#if DIM == 3
                    vzR0 = vR[2], vzL0 = vL[2];
                    xijxiAbs += (xij[2] - particles->z[i])*(xij[2] - particles->z[i]);
                    xijxjAbs += (xij[2] - particles->z[ip])*(xij[2] - particles->z[ip]);
                    xjxiAbs += (particles->z[ip] - particles->z[i])*(particles->z[ip] - particles->z[i]);
#endif
#endif
                    pR0 = pR, pL0 = pL;
                    xijxiAbs = sqrt(xijxiAbs), xijxjAbs = sqrt(xijxjAbs), xjxiAbs = sqrt(xjxiAbs);
#endif // PAIRWISE_LIMITER

                    /// reconstruct vector or primitive variables at effective face between particles

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

                    //if (i < 100){
                    //    printf("pGrad[%i] = [%e, %e, %e]\n", i, particles->pGrad[iGradR], particles->pGrad[iGradR+1], particles->pGrad[iGradR+2]);
                    //    printf("pGrad[%i] = [%e, %e, %e]\n", ip, particles->pGrad[iGradL], particles->pGrad[iGradL+1], particles->pGrad[iGradL+2]);
                    //}

                    rhoR += ::CudaUtils::dotProd(&particles->rhoGrad[iGradR], deltaXijR);
                    rhoL += ::CudaUtils::dotProd(&particles->rhoGrad[iGradL], deltaXijL);

                    //if (j == 0 && i < 10){ // only print once for each i
                    //    printf("vxGrad[%i] = [%e, %e, %e]\n", i, particles->vxGrad[iGradR], particles->vxGrad[iGradR+1], particles->vxGrad[iGradR+2]);
                    //}
                    vR[0] += ::CudaUtils::dotProd(&particles->vxGrad[iGradR], deltaXijR);
                    vL[0] += ::CudaUtils::dotProd(&particles->vxGrad[iGradL], deltaXijL);
#if DIM > 1
                    //if (j == 0 && i < 10){
                    //    printf("vyGrad[%i] = [%e, %e, %e]\n", i, particles->vyGrad[iGradR], particles->vyGrad[iGradR+1], particles->vyGrad[iGradR+2]);
                    //}
                    vR[1] += ::CudaUtils::dotProd(&particles->vyGrad[iGradR], deltaXijR);
                    vL[1] += ::CudaUtils::dotProd(&particles->vyGrad[iGradL], deltaXijL);
#if DIM == 3
                    //if (j == 0 && i < 10){
                    //    printf("vzGrad[%i] = [%e, %e, %e]\n", i, particles->vzGrad[iGradR], particles->vzGrad[iGradR+1], particles->vzGrad[iGradR+2]);
                    //}
                    vR[2] += ::CudaUtils::dotProd(&particles->vzGrad[iGradR], deltaXijR);
                    vL[2] += ::CudaUtils::dotProd(&particles->vzGrad[iGradL], deltaXijL);

                    //if (j == 0 && i < 10){
                    //    printf("projected vL = [%e, %e, %e], vR = [%e, %e, %e], %i -> %i\n",
                    //           vL[0], vL[1], vL[2], vR[0], vR[1], vR[2], i, ip);
                    //}

#endif
#endif
                    pR += ::CudaUtils::dotProd(&particles->pGrad[iGradR], deltaXijR);
                    pL += ::CudaUtils::dotProd(&particles->pGrad[iGradL], deltaXijL);

#if PAIRWISE_LIMITER
                    rhoR = ::MFV::Compute::pairwiseLimiter(rhoR, rhoR0, rhoL0, xijxiAbs, xjxiAbs, slopeLimitingParameters);
                    rhoL = ::MFV::Compute::pairwiseLimiter(rhoL, rhoL0, rhoR0, xijxjAbs, xjxiAbs, slopeLimitingParameters);

                    vR[0] = ::MFV::Compute::pairwiseLimiter(vR[0], vxR0, vxL0, xijxiAbs, xjxiAbs, slopeLimitingParameters);
                    vL[0] = ::MFV::Compute::pairwiseLimiter(vL[0], vxL0, vxR0, xijxjAbs, xjxiAbs, slopeLimitingParameters);
#if DIM > 1
                    vR[1] = ::MFV::Compute::pairwiseLimiter(vR[1], vyR0, vyL0, xijxiAbs, xjxiAbs, slopeLimitingParameters);
                    vL[1] = ::MFV::Compute::pairwiseLimiter(vL[1], vyL0, vyR0, xijxjAbs, xjxiAbs, slopeLimitingParameters);
#if DIM == 3
                    vR[2] = ::MFV::Compute::pairwiseLimiter(vR[2], vzR0, vzL0, xijxiAbs, xjxiAbs, slopeLimitingParameters);
                    vL[2] = ::MFV::Compute::pairwiseLimiter(vL[2], vzL0, vzR0, xijxjAbs, xjxiAbs, slopeLimitingParameters);
#endif
#endif
                    pR = ::MFV::Compute::pairwiseLimiter(pR, pR0, pL0, xijxiAbs, xjxiAbs, slopeLimitingParameters);
                    pL = ::MFV::Compute::pairwiseLimiter(pL, pL0, pR0, xijxjAbs, xjxiAbs, slopeLimitingParameters);

#endif // PAIRWISE_LIMITER

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
                    // actual forward prediction dimension by dimension
                    rhoR -= *dt/2. * (particles->rho[i]*viDiv + (particles->vx[i]-vFrame[0])*particles->rhoGrad[iGradR]);
                    rhoL -= *dt/2. * (particles->rho[ip]*vjDiv + (particles->vx[ip]-vFrame[0])*particles->rhoGrad[iGradL]);

                    vR[0] -= *dt/2. * (particles->pGrad[iGradR]/particles->rho[i] + (particles->vx[i]-vFrame[0])*particles->vxGrad[iGradR]);
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

                    if(pR < 0. || pL < 0.){
                        printf("ERROR: Negative pressure after half-timestep prediction for particles %i -> %i\n", i, j);
                        //printf("         xi = [%f, %f, %f], xj = [%f, %f, %f]\n",
                        //       particles->x[i], particles->y[i], particles->z[i],
                        //       particles->x[ip], particles->y[ip], particles->z[ip]);
                    }

                    /// rotate to effective face
                    ::MFV::Compute::effectiveFace(Aij, i, j+i*MAX_NUM_INTERACTIONS, interactions, particles);

//                    if (i < 10){
//                        printf("Aij[%i -> %i] = [%e, %e, %e], xi = [%e, %e, %e], xj = [%e, %e, %e]\n",
//                               i, ip, Aij[0], Aij[1], Aij[2],
//                               particles->x[i], particles->y[i], particles->z[i],
//                               particles->x[ip], particles->y[ip], particles->z[ip]);
//                    }


                    AijNorm = sqrt(::CudaUtils::dotProd(Aij, Aij));
#pragma unroll
                    for(d=0; d<DIM; d++){
                        hatAij[d] = 1./AijNorm*Aij[d];
                    }

#if SAFETY_LEVEL
#if DIM == 3
                    if (isnan(Aij[0]) || isnan(Aij[1]) || isnan(Aij[2])){
                        cudaTerminate("Riemann ERROR: NaN in Aij[%i -> %i] = [%e, %e, %e], Aij_norm = %e\n",
                                      i, ip, Aij[0], Aij[1], Aij[2], AijNorm);
                    }
#endif
                    if (isnan(rhoL) || isnan(vL[0]) || isnan(pL)
                        || isnan(rhoR) || isnan(vR[0]) || isnan(pR)){
                        cudaTerminate("Riemann ERROR: Before rotation. NaN in L or R state. %i -> %i. WL = [%e, %e, %e], WR = [%e, %e, %e]\n",
                                      i, ip, rhoL, vL[0], pL, rhoR, vR[0], pR);
                    }
#endif

                    ::CudaUtils::rotationMatrix(R, hatAij, unitX);

#if SAFETY_LEVEL
#if DIM == 3
                    if(isnan(R[0]) || isnan(R[1]) || isnan(R[2])
                       || isnan(R[3]) || isnan(R[4]) || isnan(R[5])
                       || isnan(R[6]) || isnan(R[7]) || isnan(R[9])){
                        cudaTerminate("Riemann ERROR: NaN in rotation matrix. %i -> %i. [%e, %e, %e, %e, %e, %e, %e, %e, %e]\n",
                                      i, ip, R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8]);
                    }
#endif
#endif

                    // rotate right state
#pragma unroll
                    for(d=0; d<DIM; d++){
                        vBuf[d] = vR[d];
                    }
                    ::CudaUtils::multiplyMatVec(vR, R, vBuf);

                    //printf("hatAij = [%e, %e, %e], vR = [%e, %e, %e], R = [%e, %e, %e, %e, %e, %e, %e, %e, %e]\n",
                    //       hatAij[0], hatAij[1], hatAij[2], vR[0], vR[1], vR[2],
                    //       R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8]);


                    //rotate left state
#pragma unroll
                    for(d=0; d<DIM; d++){
                        vBuf[d] = vL[d];
                    }
                    ::CudaUtils::multiplyMatVec(vL, R, vBuf);

                    //if(i < 10){
                    //    real testRot[DIM];
                    //    ::CudaUtils::multiplyMatVec(testRot, R, hatAij);
                    //    printf("Rotation effective face: [%e, %e, %e]\n", testRot[0], testRot[1], testRot[2]);
                    //}

#if SAFETY_LEVEL
                    if (isnan(rhoL) || isnan(vL[0]) || isnan(pL)
                        || isnan(rhoR) || isnan(vR[0]) || isnan(pR)){
                        cudaTerminate("Riemann ERROR: Final states. NaN in L or R state. %i -> %i. WL = [%e, %e, %e], WR = [%e, %e, %e]\n",
                                      i, ip, rhoL, vL[0], pL, rhoR, vR[0], pR);
                    }
#endif

                    //TODO: R and L state are named the opposite in the exact Riemann solver and the rest of the code
                    flagLR = riemannSolver.solve(rhoR, vR[0], pR,
                                                 rhoL, vL[0], pL,
                                                 rhoSol, vSol[0], pSol, 0.);
                    //flagLR = riemannSolver.solve(rhoL, vL[0], pL,
                    //                             rhoR, vR[0], pR,
                    //                             rhoSol, vSol[0], pSol, 0.);

//                    if (i == 0){
//                        printf("    neighbor %i:\n", ip);
//                        printf("        state L: rhoL = %e, vL = %e, pL = %e\n", rhoL, vL[0], pL);
//                        printf("        state R: rhoR = %e, vR = %e, pR = %e\n", rhoR, vR[0], pR);
//                        printf("        sol: rhoSol = %e, vSol = %e, pSol = %e\n", rhoSol, vSol[0], pSol);
//                    }
#if SAFETY_LEVEL
                    if(isnan(rhoSol) || isnan(vSol[0]) || isnan(pSol)){
                        cudaTerminate("Riemann ERROR: NaN in solution. %i -> %i. rhoSol = %e, vSol = %e, pSol = %e\n",
                                      i, ip, rhoSol, vSol[0], pSol);
                    }
#endif


                    if (flagLR == -1){
#if DIM > 1
                        // right state sampled (naming convention as in this file)
                        vSol[1] = vR[1];
#if DIM == 3
                        vSol[2] = vR[2];
#endif
#endif
                    } else if (flagLR == 1){ // flagLR == -1
#if DIM > 1
                        // left state sampled (naming convention as in this file)
                        vSol[1] = vL[1];
#if DIM ==3
                        vSol[2] = vL[2];
#endif
#endif
                    } else { // flagLR == 0
                        printf("WARNING: Vacuum state has been sampled. %i -> %i. Don't know how to handle this.\n", i, ip);
                    }

                    /// rotate solution state back to simulation frame
                    ::CudaUtils::rotationMatrix(R, unitX, hatAij);

#pragma unroll
                    for(d=0; d<DIM; d++){
                        vBuf[d] = vSol[d];
                    }
                    ::CudaUtils::multiplyMatVec(vSol, R, vBuf);

                    /// collect fluxes through effective face
                    // reuse vBuf[] as vLab[]
#pragma unroll
                    for(d=0; d<DIM; d++){
                        vBuf[d] = vSol[d] + vFrame[d];
                    }

                    vLab2 = ::CudaUtils::dotProd(vBuf, vBuf);

//                    if (i < 200){
//                        printf("    neighbor %i:\n", ip);
//                        printf("        sol: rhoSol = %e, vSol = [%e,%e, %e], pSol = %e\n", rhoSol, vSol[0], vSol[1], vSol[2], pSol);
//                    }

#if MESHLESS_FINITE_METHOD == 2 // MFM
#pragma unroll
                    for(d=0; d<DIM; d++){
                        /*
                         * This effectively supresses the mass flux  between particles and leads to the
                         * the same fluxes as computed in
                         * https://github.com/SWIFTSIM/SWIFT/blob/master/src/riemann/riemann_exact.h
                         * l. 626ff.
                         */
                        vSol[d] = 0.;
                    }
#endif

                    particles->massFlux[i] += Aij[0]*rhoSol*vSol[0];
                    particles->vxFlux[i] += Aij[0]*(rhoSol*vBuf[0]*vSol[0]+pSol);
                    particles->energyFlux[i] += Aij[0]*(vSol[0]*(pSol/(gamma-1.)+rhoSol*.5*vLab2) + pSol*vBuf[0]);
#if DIM > 1
                    particles->massFlux[i] += Aij[1]*rhoSol*vSol[1];
                    particles->vxFlux[i] += Aij[1]*rhoSol*vBuf[0]*vSol[1];
                    particles->vyFlux[i] += Aij[0]*rhoSol*vBuf[1]*vSol[0] + Aij[1]*(rhoSol*vBuf[1]*vSol[1]+pSol);
                    particles->energyFlux[i] += Aij[1]*(vSol[1]*(pSol/(gamma-1.)+rhoSol*.5*vLab2) + pSol*vBuf[1]);
#if DIM == 3
                    particles->massFlux[i] += Aij[2]*rhoSol*vSol[2];
                    particles->vxFlux[i] += Aij[2]*rhoSol*vBuf[0]*vSol[2];
                    particles->vyFlux[i] += Aij[2]*rhoSol*vBuf[1]*vSol[2];
                    particles->vzFlux[i] += Aij[0]*rhoSol*vBuf[2]*vSol[0] + Aij[1]*rhoSol*vBuf[2]*vSol[1] + Aij[2]*(rhoSol*vBuf[2]*vSol[2]+pSol);
                    particles->energyFlux[i] += Aij[2]*(vSol[2]*(pSol/(gamma-1.)+rhoSol*.5*vLab2) + pSol*vBuf[2]);
#endif
#endif
                }

//#if MESHLESS_FINITE_METHOD == 2 // MFM
//                if (particles->massFlux[i] > MFM_MASSFLUX_TOL){
//                    printf("WARNING: massFlux[%i] is %e > %e although finite mass (MFM)\n", i, particles->massFlux[i], MFM_MASS_FLUX_TOL);
//                }
//#endif

                //real r = sqrt(particles->x[i]*particles->x[i]+particles->y[i]*particles->y[i]+particles->z[i]*particles->z[i]);
                //if (r < 1e-2){
                //    printf("massFlux[%i] = %e, vFlux = [%e, %e, %e], energyFlux = %e, r = %e\n", i, particles->massFlux[i],
                //           particles->vxFlux[i], particles->vyFlux[i], particles->vzFlux[i],
                //           particles->energyFlux[i], r);
                //}
            }
        }

        namespace Launch {
            real computeGradients(Particles *particles, int *interactions, int numParticles,
                                  SlopeLimitingParameters *slopeLimitingParameters) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::MFV::Kernel::computeGradients, particles,
                                    interactions, numParticles, slopeLimitingParameters);
            }

            real riemannFluxes(Particles *particles, int *interactions, int numParticles,
                               SlopeLimitingParameters *slopeLimitingParameters, real *dt, Material *materials) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::MFV::Kernel::riemannFluxes, particles,
                                    interactions, numParticles, slopeLimitingParameters, dt, materials);
            }
        }
    }
}