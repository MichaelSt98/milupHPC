#include "../../include/integrator/device_leapfrog.cuh"
#if TARGET_GPU
#include "../../include/cuda_utils/cuda_launcher.cuh"

__global__ void LeapfrogNS::Kernel::updateX(Particles *particles, integer n, real dt) {

    integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    integer stride = blockDim.x * gridDim.x;
    integer offset = 0;

    while (bodyIndex + offset < n) {


        //if ((bodyIndex + offset) == 10) {
        //    printf("check: x = %e += %e * (%e + 0.5 * %e * %e) = %e\n", particles->x[bodyIndex + offset], dt, particles->vx[bodyIndex + offset],
        //           dt, particles->g_ax[bodyIndex + offset], dt * (particles->vx[bodyIndex + offset] + 0.5 * dt * particles->g_ax[bodyIndex + offset]));
        //}

        // update x
        particles->x[bodyIndex + offset] += dt * (particles->vx[bodyIndex + offset] + 0.5 * dt * particles->g_ax[bodyIndex + offset]);
        particles->g_ax_old[bodyIndex + offset] = particles->g_ax[bodyIndex + offset];
#if DIM > 1
        particles->y[bodyIndex + offset] += dt * (particles->vy[bodyIndex + offset] + 0.5 * dt * particles->g_ay[bodyIndex + offset]);
        particles->g_ay_old[bodyIndex + offset] = particles->g_ay[bodyIndex + offset];
#if DIM == 3
        particles->z[bodyIndex + offset] += dt * (particles->vz[bodyIndex + offset] + 0.5 * dt * particles->g_az[bodyIndex + offset]);
        particles->g_az_old[bodyIndex + offset] = particles->g_az[bodyIndex + offset];
#endif
#endif

        offset += stride;
    }
}

__global__ void LeapfrogNS::Kernel::updateV(Particles *particles, integer n, real dt) {

    integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    integer stride = blockDim.x * gridDim.x;
    integer offset = 0;

    while (bodyIndex + offset < n) {

        //if ((bodyIndex + offset) == 10) {
        //    printf("check: vx = %e += 0.5 * %e * (%e + %e) = %e\n", particles->vx[bodyIndex + offset], dt, particles->g_ax[bodyIndex + offset], particles->g_ax_old[bodyIndex + offset],
        //           0.5 * dt * (particles->g_ax[bodyIndex + offset] + particles->g_ax_old[bodyIndex + offset]));
        //}

        // update v
        particles->vx[bodyIndex + offset] += 0.5 * dt * (particles->g_ax[bodyIndex + offset] + particles->g_ax_old[bodyIndex + offset]);
#if DIM > 1
        particles->vy[bodyIndex + offset] += 0.5 * dt * (particles->g_ay[bodyIndex + offset] + particles->g_ay_old[bodyIndex + offset]);
#if DIM == 3
        particles->vz[bodyIndex + offset] += 0.5 * dt * (particles->g_az[bodyIndex + offset] + particles->g_az_old[bodyIndex + offset]);
#endif
#endif

        offset += stride;
    }
}

real LeapfrogNS::Kernel::Launch::updateX(Particles *particles, integer n, real dt) {

    ExecutionPolicy executionPolicy;
    return cuda::launch(true, executionPolicy, ::LeapfrogNS::Kernel::updateX, particles, n, dt);

}

real LeapfrogNS::Kernel::Launch::updateV(Particles *particles, integer n, real dt) {

    ExecutionPolicy executionPolicy;
    return cuda::launch(true, executionPolicy, ::LeapfrogNS::Kernel::updateV, particles, n, dt);

}

#endif // TARGET_GPU
