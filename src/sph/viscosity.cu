#include "../../include/sph/viscosity.cuh"
#include "../include/cuda_utils/cuda_launcher.cuh"

#if NAVIER_STOKES
__global__ void SPH::Kernel::calculate_shear_stress_tensor(::SPH::SPH_kernel kernel, Material *materials, Particles *particles, int *interactions, int numRealParticles) {
    int i, inc;
    inc = blockDim.x * gridDim.x;
    //Particle Loop
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numRealParticles; i += inc) {
//#if SHAKURA_SUNYAEV_ALPHA
//        double R = sqrt(p.x[i]*p.x[i] + p.y[i]*p.y[i]);
//	    p_rhs.eta[i] = matalpha_shakura[p_rhs.materialId[i]] * p.cs[i] * p.rho[i] * scale_height * R ;
//#elif CONSTANT_KINEMATIC_VISCOSITY
        // TODO: matnu
        //particles->eta[i] = matnu[particles->materialId[i]] * particles->rho[i];
//#else
        printf("not implemented\n");
        assert(0);
//#endif
    }
}
#endif // NAVIER_STOKES

#if NAVIER_STOKES
__global__ void SPH::Kernel::calculate_kinematic_viscosity(::SPH::SPH_kernel kernel, Material *materials, Particles *particles, int *interactions, int numRealParticles) {

    int i, inc;
    int e, f, g;
    int j, k;

    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numRealParticles; i += inc) {

        real dv[DIM];
        real dr[DIM];
        real r;
        real sml;
        real dWdr, dWdrj, W, Wj;
        real dWdx[DIM], dWdxj[DIM];

        for (k = 0; k < DIM * DIM; k++) {
            particles->Tshear[i*DIM*DIM+k] = 0.0;
        }

        for (k = 0; k < particles->noi[i]; k++) {

            j = interactions[i * MAX_NUM_INTERACTIONS + k];

            dv[0] = particles->vx[i] - particles->vx[j];
#if DIM > 1
            dv[1] = particles->vy[i] - particles->vy[j];
#if DIM == 3
            dv[2] = particles->vz[i] - particles->vz[j];
#endif
#endif

            dr[0] = particles->x[i] - particles->x[j];
#if DIM > 1
            dr[1] = particles->y[i] - particles->y[j];
#if DIM > 2
            dr[2] = particles->z[i] - particles->z[j];
#endif
#endif

            r = 0;
            for (e = 0; e < DIM; e++) {
                r += dr[e] * dr[e];
                dWdx[e] = 0.0;
            }
            W = 0.0;
            dWdr = 0.0;
            r = sqrt(r);

            sml = particles->sml[i];

#if (VARIABLE_SML || INTEGRATE_SML) // || DEAL_WITH_TOO_MANY_INTERACTIONS)
            sml = 0.5*(particles->sml[i] + particles->sml[j]);
#endif

            kernel(&W, dWdx, &dWdr, dr, sml);


            double trace = 0;
            for (e = 0; e < DIM; e++) {
# if (SPH_EQU_VERSION == 1)
                trace +=  particles->mass[j]/particles->rho[i] * (-dv[e])*dWdx[e] ;
# elif (SPH_EQU_VERSION == 2)
                trace +=  particles->mass[j]/particles->rho[j] * (-dv[e])*dWdx[e] ;
#endif
            }

            for (e = 0; e < DIM; e++) {
                for (f = 0; f < DIM; f++) {
# if (SPH_EQU_VERSION == 1)
                    particles->Tshear[i*DIM*DIM+e*DIM+f] += particles->mass[j]/particles->rho[i] * (-dv[e]*dWdx[f] - dv[f]*dWdx[e]);
# elif (SPH_EQU_VERSION == 2)
                    particles->Tshear[i*DIM*DIM+e*DIM+f] += particles->mass[j]/particles->rho[j] * (-dv[e]*dWdx[f] - dv[f]*dWdx[e]);
#endif
                    // traceless
                    if (e == f) {
# if (SPH_EQU_VERSION == 1)
                        particles->Tshear[i*DIM*DIM+e*DIM+f] -= 2./3 * trace;
# elif (SPH_EQU_VERSION == 2)
                        particles->Tshear[i*DIM*DIM+e*DIM+f] -= 2./3 * trace;
#endif
                    }
                }
            }
        }
    }
}
#endif // NAVIER STOKES

