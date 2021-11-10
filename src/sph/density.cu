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
            real W;
            real Wj;
            real dx[DIM];
            real dWdx[DIM];
            real dWdr;
            real rho;
            real sml;
            real tolerance;
#if SML_CORRECTION
    real dhdrho, sml_omega,sml_omega_sum, r;
    real f, df, h_new, h_init, rho_h;
    //the proportionality constant (h_fact = 4.0) defines the average number of neighbours: [2D] noi = pi * h_fact^2, [3D] noi = 4/3 * pi * h_fact^3
    real h_fact = 4.0;
#endif // SML_CORRECTION

            inc = blockDim.x * gridDim.x;
            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
                //if (EOS_TYPE_IGNORE == matEOS[p_rhs.materialId[i]] || p_rhs.materialId[i] == EOS_TYPE_IGNORE) {
                //    continue;
                //}
                tolerance = 0.0;
                int cnt = 0;

#if SML_CORRECTION
        h_init = particles->sml[i];
        h_new = 0.0;
        // if Bisection method is used
        // double a = 0.0, b = 0.0, c = 0.0;
	    // int bis_cnt = 0;
        // int bisection = 0;
#endif // SML_CORRECTION

                do {
#if SML_CORRECTION
            sml_omega_sum = 0.0;
#endif // SML_CORRECTION
                    sml = particles->sml[i];

                    // self density is m_i W_ii
                    for (d = 0; d < DIM; d++) {
                        dx[d] = 0;
                    }

                    kernel(&W, dWdx, &dWdr, dx, sml);
//#if SHEPARD_CORRECTION
//            W /= p_rhs.shepard_correction[i];
//#endif
                    rho = particles->mass[i] * W;
                    if (rho == 0.0) {
                        printf("%i: rho is %f W: %e m: %e x = (%e, %e, %e) sml = %e\n", i, rho, W, particles->mass[i],
                               particles->x[i], particles->y[i], particles->z[i], particles->sml[i]);
                        //assert(0);
                    }
                    // sph sum for particle i
                    for (j = 0; j < particles->noi[i]; j++) {
                        ip = interactions[i * MAX_NUM_INTERACTIONS + j];
                        //if (EOS_TYPE_IGNORE == matEOS[p_rhs.materialId[ip]] || p_rhs.materialId[ip] == EOS_TYPE_IGNORE) {
                        //    continue;
                        //}
#if (VARIABLE_SML || INTEGRATE_SML) // || DEAL_WITH_TOO_MANY_INTERACTIONS)
                sml = 0.5 * (particles->sml[i] + particles->sml[ip]);
#endif

                        dx[0] = particles->x[i] - particles->x[ip];
#if DIM > 1
                        dx[1] = particles->y[i] - particles->y[ip];
#if DIM > 2
                        dx[2] = particles->z[i] - particles->z[ip];
#endif
#endif

#if SML_CORRECTION
                r = 0;
                for (d = 0; d < DIM; d++) {
                    r += dx[d]*dx[d];
                }
                r = cuda::math::sqrt(r);
#endif // SML_CORRECTION

//#if AVERAGE_KERNELS
//                kernel(&W, dWdx, &dWdr, dx, p.h[i]);
//                Wj = 0;
//                kernel(&Wj, dWdx, &dWdr, dx, p.h[j]);
//# if SHEPARD_CORRECTION
//                W /= p_rhs.shepard_correction[i];
//                Wj /= p_rhs.shepard_correction[j];
//# endif
//                W = 0.5 * (W + Wj);
//#else
                        kernel(&W, dWdx, &dWdr, dx, sml);
//# if SHEPARD_CORRECTION
//                W /= p_rhs.shepard_correction[i];
//# endif
//                // contribution of interaction
//#endif // AVERAGE_KERNELS

#if SML_CORRECTION
                sml_omega_sum += particles->mass[ip] * (-1) * (DIM * W/sml + (r / sml) * dWdr);
#endif // SML_CORRECTION
                        rho += particles->mass[ip] * W;
                    }
#if SML_CORRECTION
            rho_h = particles->mass[i] * pow(real(h_fact / particles->sml[i]), DIM);
            dhdrho = -particles->sml[i] / (DIM * rho);
            sml_omega = 1 - dhdrho * sml_omega_sum;

            // Newton-Raphson method tolerance e-3 (Phantom)
            f = rho_h - rho;
            df = -DIM * rho / particles->sml[i] * sml_omega;
            h_new = particles->sml[i] - f / df;

            // arbitrary set limit for sml change
            if (h_new > 1.2 * particles->sml[i]) {
                h_new = 1.2 * particles->sml[i];
            } else if (h_new < 0.8 * particles->sml[i]) {
                h_new = 0.8 * particles->sml[i];
            }

           	tolerance = abs(h_new - particles->sml[i]) / h_init;
            if (tolerance > 1e-3) {
                if (h_new < 0){
	       	        printf("SML_CORRECTION: NEGATIVE SML!");
                }
                particles->sml[i] = h_new;
                particles->sml_omega[i] = sml_omega;
                real r;
                real x_radius = 0.5 * (*tree->maxX - (*tree->minX));
#if DIM > 1
                real y_radius = 0.5 * (*tree->maxY - (*tree->minY));
#if DIM == 3
                real z_radius = 0.5 * (*tree->maxZ - (*tree->minZ));
#endif
#endif

#if DIM == 1
                r = x_radius;
#elif DIM == 2
                r = cuda::math::max(x_radius, y_radius);
#else
                real r_temp = cuda::math::max(x_radius, y_radius);
                r = cuda::math::max(r_temp, z_radius); //TODO: (0.5 * r) or (1.0 * r)
#endif
                ::SPH::Kernel::redoNeighborSearch(tree, particles, i,
                        interactions, r, numParticles, 0);
                cnt++;
            }
#endif // SML_CORRECTION

                } while (tolerance > 1e-3 && cnt < 10);
                // write to global memory
                particles->rho[i] = rho;
                if (particles->rho[i] == 0.) {
                    printf("density rho[%i] = %f\n", i, particles->rho[i]);
                }
                //if (particles->rho[i] > 0.) {
                //    printf("density: rho[%i] = %f\n", i, particles->rho[i]);
                //}
            }
        }

        real Launch::calculateDensity(::SPH::SPH_kernel kernel, Tree *tree, Particles *particles, int *interactions, int numParticles) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SPH::Kernel::calculateDensity, kernel, tree, particles, interactions, numParticles);
        }

    }
}
