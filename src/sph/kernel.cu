#include "../../include/sph/kernel.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

#define MIN_NUMBER_OF_INTERACTIONS_FOR_TENSORIAL_CORRECTION_TO_WORK 0

__device__ SPH::SPH_kernel spiky_p = SPH::SmoothingKernel::spiky;
__device__ SPH::SPH_kernel cubicSpline_p = SPH::SmoothingKernel::cubicSpline;
__device__ SPH::SPH_kernel wendlandc2_p = SPH::SmoothingKernel::wendlandc2;
__device__ SPH::SPH_kernel wendlandc4_p = SPH::SmoothingKernel::wendlandc4;
__device__ SPH::SPH_kernel wendlandc6_p = SPH::SmoothingKernel::wendlandc6;

namespace SPH {

    __device__ void SmoothingKernel::spiky(real *W, real dWdx[DIM], real *dWdr, real dx[DIM], real sml) {

        real r, q;
        r = 0;
        for (int d = 0; d < DIM; d++) {
            r += dx[d] * dx[d];
            dWdx[d] = 0;
        }
        r = sqrt(r);
        *dWdr = 0;
        *W = 0;
        q = r/sml;

#if DIM == 1
        printf("Error, this kernel can only be used with DIM == 2,3\n");
        //assert(0);
#elif DIM == 2
        if (q > 1) {
            *W = 0;
        } else if (q >= 0.0) {
            *W = 10./(M_PI * sml * sml) * (1-q) * (1-q) * (1-q);
            *dWdr = -30./(M_PI * sml * sml * sml) * (1-q) * (1-q);
        }
#else
        if (q > 1) {
            *W = 0;
        } else if (q >= 0.0) {
            *W = 15./(M_PI * sml * sml * sml) * (1-q) * (1-q) * (1-q);
            *dWdr = -45/(M_PI * sml * sml * sml * sml) * (1-q) * (1-q);
        }
#endif
        for (int d = 0; d < DIM; d++) {
            dWdx[d] = *dWdr/r * dx[d];
        }
    }

    __device__ void SmoothingKernel::cubicSpline(real *W, real dWdx[DIM], real *dWdr, real dx[DIM], real sml) {

        real r, q, f;
        r = 0;
        for (int d = 0; d < DIM; d++) {
            r += dx[d] * dx[d];
            dWdx[d] = 0;
        }
        r = sqrt(r);
        *dWdr = 0;
        *W = 0;
        q = r/sml;

        f = 4./3. * 1./sml;
#if DIM > 1
        f = 40./(7 * M_PI) * 1./(sml * sml);
#if DIM > 2
        f = 8./M_PI * 1./(sml * sml * sml);
#endif
#endif
        if (q > 1) {
            *W = 0;
            *dWdr = 0.0;
        } else if (q > 0.5) {
            *W = 2. * f * (1.-q) * (1.-q) * (1-q);
            *dWdr = -6. * f * 1./sml * (1.-q) * (1.-q);
        } else if (q <= 0.5) {
            *W = f * (6. * q * q * q - 6. * q * q + 1.);
            *dWdr = 6. * f/sml * (3 * q * q - 2 * q);
        }
        for (int d = 0; d < DIM; d++) {
            dWdx[d] = *dWdr/r * dx[d];
        }
    }

    // Wendland C2 from Dehnen & Aly 2012
    __device__ void SmoothingKernel::wendlandc2(real *W, real dWdx[DIM], real *dWdr, real dx[DIM], real sml) {

        real r, q;
        r = 0;
        for (int d = 0; d < DIM; d++) {
            r += dx[d]*dx[d];
            dWdx[d] = 0;
        }
        r = sqrt(r);
        *dWdr = 0;
        *W = 0;
        if (r > sml) {
            *W = 0;
        } else {
            q = r/sml;
#if DIM == 1
            *W = 5./(4. * sml) * (1-q) * (1-q) * (1-q) * (1+3*q) * (q < 1);
            *dWdr = -15/(sml * sml) * q * (1-q) * (1-q) * (q < 1);
#elif DIM == 2
            *W = 7./(M_PI * sml * sml) * (1-q) * (1-q) * (1-q) * (1-q) * (1+4 * q) * (q < 1);
            *dWdr = -140./(M_PI * sml * sml * sml) * q * (1-q) * (1-q) * (1-q) * (q < 1);
#else //DIM == 3
            *W = 21./(2 * M_PI * sml * sml * sml) * (1-q) * (1-q) * (1-q) * (1-q) * (1+4 * q) * (q < 1);
            *dWdr = -210./(M_PI * sml * sml * sml * sml) * q * (1-q) * (1-q) * (1-q) * (q < 1);
#endif
            for (int d = 0; d < DIM; d++) {
                dWdx[d] = *dWdr/r * dx[d];
            }
        }
    }

// Wendland C4 from Dehnen & Aly 2012
    __device__ void SmoothingKernel::wendlandc4(real *W, real dWdx[DIM], real *dWdr, real dx[DIM], real sml) {

        real r, q;
        r = 0;
        for (int d = 0; d < DIM; d++) {
            r += dx[d]*dx[d];
            dWdx[d] = 0;
        }
        r = sqrt(r);
        *dWdr = 0;
        *W = 0;

        if (r > sml) {
            *W = 0;
        } else {
            q = r/sml;
#if DIM == 1
            *W = 3./(2.*sml) * (1-q) * (1-q) * (1-q) * (1-q) * (1-q) * (1+5*q+8*q*q) * (q < 1);
            *dWdr = -21./(sml*sml) * q * (1-q) * (1-q) * (1-q) * (1-q) * (1+4*q) * (q < 1);
#elif DIM == 2
            *W = 9./(M_PI*sml*sml) * (1-q) * (1-q) * (1-q) * (1-q) * (1-q) * (1-q) * (1.+6*q+35./3.*q*q) * (q < 1);
            *dWdr = -54./(M_PI*sml*sml*sml) * (1-q)*(1-q)*(1-q)*(1-q)*(1-q) * (1.-35.*q*q+105.*q*q*q) * (q< 1);
#else //DIM == 3
            *W = 495./(32.*M_PI*sml*sml*sml) * (1-q) * (1-q) * (1-q) * (1-q) * (1-q) * (1-q) * (1.+6.*q+35./3.*q*q) * (q < 1);
            *dWdr = -1485./(16.*M_PI*sml*sml*sml*sml) * (1-q) * (1-q) * (1-q) * (1-q) * (1-q) * (1.-35.*q*q+105.*q*q*q) * (q< 1);
#endif
            for (int d = 0; d < DIM; d++) {
                dWdx[d] = *dWdr/r * dx[d];
            }
        }
    }


    // Wendland C6 from Dehnen & Aly 2012
    __device__ void SmoothingKernel::wendlandc6(real *W, real dWdx[DIM], real *dWdr, real dx[DIM], real sml) {

        real r, q;
        r = 0;
        for (int d = 0; d < DIM; d++) {
            r += dx[d]*dx[d];
            dWdx[d] = 0;
        }
        r = sqrt(r);
        *dWdr = 0;
        *W = 0;
        if (r > sml) {
            *W = 0;
        } else {
            q = r/sml;
#if DIM == 1
            *W = 55./(32.*sml) * (1-q)*(1-q)*(1-q)*(1-q)*(1-q)*(1-q)*(1-q) * (1+7*q+19*q*q+21*q*q*q) * (q < 1);
            *dWdr = -165./(16*sml*sml) * q * (1-q)*(1-q)*(1-q)*(1-q)*(1-q)*(1-q) * (3+18*q+35*q*q) * (q < 1);
#elif DIM == 2
            *W =  78./(7.*M_PI*sml*sml) * (1-q)*(1-q)*(1-q)*(1-q)*(1-q)*(1-q)*(1-q)*(1-q) * (1.+8.*q+25.*q*q+32*q*q*q) * (q < 1);

            *dWdr = -1716./(7.*M_PI*sml*sml*sml) * q * (1-q)*(1-q)*(1-q)*(1-q)*(1-q)*(1-q)*(1-q) * (1.+7*q+16*q*q) * (q < 1);
#else // DIM == 3
            *W = 1365./(64.*M_PI*sml*sml*sml) * (1-q)*(1-q)*(1-q)*(1-q)*(1-q)*(1-q)*(1-q)*(1-q) * (1.+8.*q+25.*q*q+32*q*q*q) * (q < 1);
            *dWdr = -15015./(32.*M_PI*sml*sml*sml*sml) * q * (1-q)*(1-q)*(1-q)*(1-q)*(1-q)*(1-q)*(1-q) *
                    (1.+7*q+16*q*q) * (q < 1);
#endif
            for (int d = 0; d < DIM; d++) {
                dWdx[d] = *dWdr/r * dx[d];
            }
        }
    }


    CUDA_CALLABLE_MEMBER real fixTensileInstability(SPH_kernel kernel, Particles *particles, int p1, int p2) {

        real hbar;
        real dx[DIM];
        real W;
        real W2;
        real dWdr;
        real dWdx[DIM];

        W = 0;
        W2 = 0;
        dWdr = 0;
        for (int d = 0; d < DIM; d++) {
            dx[d] = 0.0;
            dWdx[d] = 0;
        }
        dx[0] = particles->x[p1] - particles->x[p2];
#if DIM > 1
        dx[1] = particles->y[p1] - particles->y[p2];
#if DIM > 2
        dx[2] = particles->z[p1] - particles->z[p2];
#endif
#endif

        hbar = 0.5 * (particles->sml[p1] + particles->sml[p2]);
        // calculate kernel for r and particle_distance
        kernel(&W, dWdx, &dWdr, dx, hbar);
        //TODO: matmean_particle_distance
        //dx[0] = matmean_particle_distance[p_rhs.materialId[a]];
        for (int d = 1; d < DIM; d++) {
            dx[d] = 0;
        }
        kernel(&W2, dWdx, &dWdr, dx, hbar);

        return W/W2;

    }

#if (NAVIER_STOKES || BALSARA_SWITCH || INVISCID_SPH || INTEGRATE_ENERGY)
    __global__ void CalcDivvandCurlv(SPH_kernel kernel, Particles *particles, int *interactions, int numParticles) {

        int i, inc, j, k, m, d, dd;
        // absolute values of div v and curl v */
        real divv;
        real curlv[DIM];
        real W, dWdr;
        real Wj, dWdrj, dWdxj[DIM];
        real dWdx[DIM], dx[DIM];
        real sml;
        real vi[DIM], vj[DIM];
        real r;
        inc = blockDim.x * gridDim.x;
        for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
            //if (EOS_TYPE_IGNORE == matEOS[p_rhs.materialId[i]] || p_rhs.materialId[i] == EOS_TYPE_IGNORE) {
            //    continue;
            //}
            k = particles->noi[i];
            divv = 0;
            for (m = 0; m < DIM; m++) {
                curlv[m] = 0;
                dWdx[m] = 0;
            }
            sml = particles->sml[i];
            // interaction partner loop
            for (m = 0; m < k; m++) {
                j = interactions[i*MAX_NUM_INTERACTIONS + m];
                // get the kernel values
#if VARIABLE_SML
                sml = 0.5 *(particles->sml[i] + particles->sml[j]);
#endif
                dx[0] = particles->x[i] - particles->x[j];
#if DIM > 1
                dx[1] = particles->y[i] - particles->y[j];
#if DIM > 2
                dx[2] = particles->z[i] - particles->z[j];
#endif
#endif


#if AVERAGE_KERNELS
                kernel(&W, dWdx, &dWdr, dx, particles->sml[i]);
                kernel(&Wj, dWdxj, &dWdrj, dx, particles->sml[j]);
# if SHEPARD_CORRECTION
                //TODO: shephard correction
                W /= particles->shepard_correction[i];
                Wj /= particles->shepard_correction[j];
                for (d = 0; d < DIM; d++) {
                    dWdx[d] /= p_rhs.shepard_correction[i];
                    dWdxj[d] /= p_rhs.shepard_correction[j];
                }
# endif
                W = 0.5 * (W + Wj);
                for (d = 0; d < DIM; d++) {
                    dWdx[d] = 0.5 * (dWdx[d] + dWdxj[d]);
                }
#else
                kernel(&W, dWdx, &dWdr, dx, sml);
# if SHEPARD_CORRECTION
                W /= p_rhs.shepard_correction[i];
                for (d = 0; d < DIM; d++) {
                    dWdx[d] /= particles->shepard_correction[i];
                }
# endif
#endif // AVERAGE_KERNELS

                vi[0] = particles->vx[i];
                vj[0] = particles->vx[j];
#if DIM > 1
                vi[1] = particles->vy[i];
                vj[1] = particles->vy[j];
#if DIM > 2
                vi[2] = particles->vz[i];
                vj[2] = particles->vz[j];
#endif
#endif
                r = 0;
                for (d = 0; d < DIM; d++) {
                    r += dx[d]*dx[d];
                }
                r = sqrt(r);
                // divv
                for (d = 0; d < DIM; d++) {
#if TENSORIAL_CORRECTION
                    for (dd = 0; dd < DIM; dd++) {
                    divv += particles->mass[j]/particles->rho[j] * (vj[d] - vi[d]) * particles->tensorialCorrectionMatrix[i*DIM*DIM+d*DIM+dd] * dWdx[dd];
                }
#else
                    divv += particles->mass[j]/particles->rho[j] * (vj[d] - vi[d]) * dWdx[d];
#endif

                }
                /* curlv */
#if (DIM == 1 && BALSARA_SWITCH)
#error unset BALSARA SWITCH in 1D
#elif DIM == 2
                // only one component in 2D
            curlv[0] += particles->mass[j]/particles->rho[i] * ((vi[0] - vj[0]) * dWdx[1]
                        - (vi[1] - vj[1]) * dWdx[0]);
            curlv[1] = 0;
#elif DIM == 3
                curlv[0] += particles->mass[j]/particles->rho[i] * ((vi[1] - vj[1]) * dWdx[2]
                                               - (vi[2] - vj[2]) * dWdx[1]);
                curlv[1] += particles->mass[j]/particles->rho[i] * ((vi[2] - vj[2]) * dWdx[0]
                                               - (vi[0] - vj[0]) * dWdx[2]);
                curlv[2] += particles->mass[j]/particles->rho[i] * ((vi[0] - vj[0]) * dWdx[1]
                                               - (vi[1] - vj[1]) * dWdx[0]);
#endif
            }
            for (d = 0; d < DIM; d++) {
                //TODO: particles or particles_rhs: curlv and divv
                //particles->curlv[i*DIM+d] = curlv[d];
            }
            //particles->divv[i] = divv;
        }
    }
#endif //  (NAVIER_STOKES || BALSARA_SWITCH || INVISCID_SPH)

#if ZERO_CONSISTENCY //SHEPARD_CORRECTION
    // this adds zeroth order consistency but needs one more loop over all neighbours
__global__ void shepardCorrection(SPH_kernel kernel, Particles *particles, int *interactions, int numParticles) {

    register int i, inc, j, m;
    register real dr[DIM], h, dWdr;
    inc = blockDim.x * gridDim.x;
    real W, dWdx[DIM], Wj;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        real shepard_correction;
        W = 0;
        for (m = 0; m < DIM; m++) {
            dr[m] = 0.0;
        }
        kernel(&W, dWdx, &dWdr, dr, particles->sml[i]);
        shepard_correction = particles->mass[i]/particles->rho[i]*W;

        for (m = 0; m < particles->noi[i]; m++) {
            W = 0;
            j = interactions[i*MAX_NUM_INTERACTIONS + m];
            //if (EOS_TYPE_IGNORE == matEOS[p_rhs.materialId[j]] || p_rhs.materialId[j] == EOS_TYPE_IGNORE) {
            //    continue;
            //}
            dr[0] = particles->x[i] - particles->x[j];
#if DIM > 1
            dr[1] = particles->y[i] - particles->y[j];
#if DIM > 2
            dr[2] = particles->z[i] - particles->z[j];
#endif
#endif

#if AVERAGE_KERNELS
            kernel(&W, dWdx, &dWdr, dr, particles->sml[i]);
            Wj = 0;
            kernel(&Wj, dWdx, &dWdr, dr, particles->sml[j]);
            W = 0.5*(W + Wj);
#else
            h = 0.5*(particles->sml[i] + particles->sml[j]);
            kernel(&W, dWdx, &dWdr, dr, h);
#endif

            shepard_correction += particles->mass[j]/particles->rho[j]*W;
        }
        // TODO: particles or particles_rhs: shepard_correction
        //particles->shepard_correction[i] = shepard_correction;
        //printf("%g\n", shepard_correction);
    }
}
#endif


#if LINEAR_CONSISTENCY //TENSORIAL_CORRECTION
    // this adds first order consistency but needs one more loop over all neighbours
__global__ void tensorialCorrection(SPH_kernel kernel, Particles *particles, int *interactions, int numParticles)
{
    register int i, inc, j, k, m;
    register int d, dd;
    int rv = 0;
    inc = blockDim.x * gridDim.x;
    register real r, dr[DIM], h, dWdr, tmp, f1, f2;
    real W, dWdx[DIM];
    real Wj, dWdxj[DIM];
    real wend_f, wend_sml, q, distance;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        register real corrmatrix[DIM*DIM];
        register real matrix[DIM*DIM];
        for (d = 0; d < DIM*DIM; d++) {
            corrmatrix[d] = 0;
            matrix[d] = 0;
        }
        //if (EOS_TYPE_IGNORE == matEOS[p_rhs.materialId[i]] || p_rhs.materialId[i] == EOS_TYPE_IGNORE) {
        //       continue;
        //}

        k = particles->noi[i];

        // loop over all interaction partner
        for (m = 0; m < k; m++) {
            j = interactions[i*MAX_NUM_INTERACTIONS+m];
            //if (EOS_TYPE_IGNORE == matEOS[p_rhs.materialId[j]] || p_rhs.materialId[j] == EOS_TYPE_IGNORE) {
            //    continue;
            //}
            dr[0] = particles->x[i] - particles->x[j];
#if DIM > 1
            dr[1] = particles->y[i] - particles->y[j];
#if DIM == 3
            dr[2] = particles->z[i] - particles->z[j];

            r = sqrt(dr[0]*dr[0]+dr[1]*dr[1]+dr[2]*dr[2]);
#elif DIM == 2
            r = sqrt(dr[0]*dr[0]+dr[1]*dr[1]);
#endif
#endif

#if AVERAGE_KERNELS
            kernel(&W, dWdx, &dWdr, dr, particles->sml[i]);
            kernel(&Wj, dWdxj, &dWdr, dr, particles->sml[j]);
# if SHEPARD_CORRECTION
            W /= particles->shepard_correction[i];
            Wj /= particles->shepard_correction[j];
            for (d = 0; d < DIM; d++) {
                dWdx[d] /= particles->shepard_correction[i];
                dWdxj[d] /= particles->shepard_correction[j];
            }
            for (d = 0; d < DIM; d++) {
                dWdx[d] = 0.5 * (dWdx[d] + dWdxj[d]);
            }
            W = 0.5 * (W + Wj);
# endif


#else
            h = 0.5*(particles->sml[i] + particles->sml[j]);
            kernel(&W, dWdx, &dWdr, dr, h);
# if SHEPARD_CORRECTION
            W /= particles->shepard_correction[i];
            for (d = 0; d < DIM; d++) {
                dWdx[d] /= particles->shepard_correction[i];
            }
# endif
#endif // AVERAGE_KERNELS

            for (d = 0; d < DIM; d++) {
                for (dd = 0; dd < DIM; dd++) {
                    corrmatrix[d*DIM+dd] -= particles->mass[j]/particles->rho[j] * dr[d] * dWdx[dd];
                }
            }
        } // end loop over interaction partners

        rv = CudaUtils::invertMatrix(corrmatrix, matrix);
        // if something went wrong during inversion, use identity matrix
        if (rv < 0 || k < MIN_NUMBER_OF_INTERACTIONS_FOR_TENSORIAL_CORRECTION_TO_WORK) {
            #if DEBUG_LINALG
            if (threadIdx.x == 0) {
                printf("could not invert matrix: rv: %d and k: %d\n", rv, k);
                for (d = 0; d < DIM; d++) {
                    for (dd = 0; dd < DIM; dd++) {
                        printf("%e\t", corrmatrix[d*DIM+dd]);
                    }
                        printf("\n");
                }
            }
            #endif
            for (d = 0; d < DIM; d++) {
                for (dd = 0; dd < DIM; dd++) {
                    matrix[d*DIM+dd] = 0.0;
                    if (d == dd)
                        matrix[d*DIM+dd] = 1.0;
                }
            }
        }
        for (d = 0; d < DIM*DIM; d++) {
            // TODO: particles or particles_rhs: tensorialCorrectionMatrix
            //particles->tensorialCorrectionMatrix[i*DIM*DIM+d] = matrix[d];

        }
    }
}
#endif
}

