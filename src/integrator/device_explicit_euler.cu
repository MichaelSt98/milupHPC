#include "../../include/integrator/device_explicit_euler.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

__global__ void ExplicitEulerNS::Kernel::update(Particles *particles, integer n, real dt) {

    integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    integer stride = blockDim.x * gridDim.x;
    integer offset = 0;

    while (bodyIndex + offset < n) {

        // calculating/updating the velocities
        particles->vx[bodyIndex + offset] += dt * (particles->ax[bodyIndex + offset] + particles->g_ax[bodyIndex + offset]);
#if DIM > 1
        particles->vy[bodyIndex + offset] += dt * (particles->ay[bodyIndex + offset] + particles->g_ay[bodyIndex + offset]);
#if DIM == 3
        particles->vz[bodyIndex + offset] += dt * (particles->az[bodyIndex + offset] + particles->g_az[bodyIndex + offset]);
#endif
#endif

        //if ((bodyIndex + offset) % 1000 == 0) {
        //    printf("vx[%i] += dt * (%f + %f) = %f\n", bodyIndex + offset, particles->ax[bodyIndex + offset],
        //           particles->g_ax[bodyIndex + offset], dt * (particles->ax[bodyIndex + offset] + particles->g_ax[bodyIndex + offset]));
        //}

        // calculating/updating the positions
        particles->x[bodyIndex + offset] += dt * particles->vx[bodyIndex + offset];
#if DIM > 1
        particles->y[bodyIndex + offset] += dt * particles->vy[bodyIndex + offset];
#if DIM == 3
        particles->z[bodyIndex + offset] += dt * particles->vz[bodyIndex + offset];
#endif
#endif

#if INTEGRATE_DENSITY
        particles->rho[bodyIndex + offset] = particles->rho[bodyIndex + offset] + dt * particles->drhodt[bodyIndex + offset];
        //particles->drhodt[i] = 0.5 * (predictor->drhodt[i] + particles->drhodt[i]);
#endif
#if INTEGRATE_ENERGY
        particles->e[bodyIndex + offset] += dt * particles->dedt[bodyIndex + offset];
        if (particles->e[bodyIndex + offset] < 1e-50) {
            particles->e[bodyIndex + offset] = 1e-50;
        }
        //printf("e = %e + (%e * %e)\n", particles->e[bodyIndex + offset], dt, particles->dedt[bodyIndex + offset]);
        //particles->dedt[i] = 0.5 * (predictor->dedt[i] + particles->dedt[i]);
#endif
#if INTEGRATE_SML
        particles->sml[bodyIndex + offset] = particles->sml[bodyIndex + offset] + dt * particles->dsmldt[bodyIndex + offset];
        //printf("dsmldt: sml += %e * (dsmldt[%i] = %e)\n", dt, bodyIndex + offset, particles->dsmldt[bodyIndex + offset]);
#endif

        offset += stride;
    }
}

real ExplicitEulerNS::Kernel::Launch::update(Particles *particles, integer n, real dt) {

    ExecutionPolicy executionPolicy;
    return cuda::launch(true, executionPolicy, ::ExplicitEulerNS::Kernel::update, particles, n, dt);

}
