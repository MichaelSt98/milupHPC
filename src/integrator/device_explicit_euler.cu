#include "../../include/integrator/device_explicit_euler.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

__global__ void ExplicitEulerNS::Kernel::update(Particles *particles, integer n, real dt) {

    integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    integer stride = blockDim.x * gridDim.x;
    integer offset = 0;

    while (bodyIndex + offset < n) {

        // calculating/updating the velocities
        particles->vx[bodyIndex + offset] += dt * (particles->ax[bodyIndex + offset] + particles->g_ax[bodyIndex + offset]);
        //if ((bodyIndex + offset) % 1000 == 0) {
        //    printf("vx[%i] += dt * (%f + %f) = %f\n", bodyIndex + offset, particles->ax[bodyIndex + offset],
        //           particles->g_ax[bodyIndex + offset], particles->vx[bodyIndex + offset]);
        //}
#if DIM > 1
        particles->vy[bodyIndex + offset] += dt * (particles->ay[bodyIndex + offset] + particles->g_ay[bodyIndex + offset]);
#if DIM == 3
        particles->vz[bodyIndex + offset] += dt * (particles->az[bodyIndex + offset] + particles->g_az[bodyIndex + offset]);
#endif
#endif

        // calculating/updating the positions
        particles->x[bodyIndex + offset] += dt * particles->vx[bodyIndex + offset];
#if DIM > 1
        particles->y[bodyIndex + offset] += dt * particles->vy[bodyIndex + offset];
#if DIM == 3
        particles->z[bodyIndex + offset] += dt * particles->vz[bodyIndex + offset];
#endif
#endif

        // debug
        //if (bodyIndex + offset == n - 1 || bodyIndex + offset == 0) {
        // //if ((bodyIndex + offset) % 100 == 0) {
        //    printf("update: %i (%f, %f, %f) x += (%f, %f, %f)\n", bodyIndex + offset, particles->x[bodyIndex + offset],
        //           particles->y[bodyIndex + offset], particles->z[bodyIndex + offset], d * dt * particles->vx[bodyIndex + offset],
        //           d * dt * particles->vy[bodyIndex + offset], d * dt * particles->vz[bodyIndex + offset]);
        //    printf("update: %i (%f, %f, %f) %f (%f, %f, %f) (%f, %f, %f) %f\n", bodyIndex + offset,
        //           particles->x[bodyIndex + offset],
        //           particles->y[bodyIndex + offset],
        //           particles->z[bodyIndex + offset],
        //           particles->mass[bodyIndex + offset],
        //           particles->vx[bodyIndex + offset],
        //           particles->vy[bodyIndex + offset],
        //           particles->vz[bodyIndex + offset],
        //           particles->ax[bodyIndex + offset],
        //           particles->ay[bodyIndex + offset],
        //           particles->az[bodyIndex + offset],
        //           particles->ax[bodyIndex + offset] * particles->ax[bodyIndex + offset] +
        //           particles->ay[bodyIndex + offset] * particles->ay[bodyIndex + offset] +
        //           particles->az[bodyIndex + offset] * particles->az[bodyIndex + offset]);
        //}
        //if (abs(particles->x[bodyIndex + offset]) < 3 && abs(particles->y[bodyIndex + offset]) < 3 &&
        //        abs(particles->z[bodyIndex + offset]) < 3) {
        //    printf("centered: index = %i (%f, %f, %f) %f\n", bodyIndex + offset,
        //           particles->x[bodyIndex + offset],
        //           particles->y[bodyIndex + offset],
        //           particles->z[bodyIndex + offset],
        //           particles->mass[bodyIndex + offset]);
        //    if (particles->mass[bodyIndex + offset] < 1) {
        //        //assert(0);
        //    }
        //}
        //if (abs(particles->ax[bodyIndex + offset]) < 10 && abs(particles->ay[bodyIndex + offset]) < 10 &&
        //    abs(particles->az[bodyIndex + offset]) < 10) {
        //if (true) {
        //    printf("ACCELERATION tiny! centered: index = %i (%f, %f, %f) %f (%f, %f, %f) (%f, %f, %f)\n", bodyIndex + offset,
        //           particles->x[bodyIndex + offset],
        //           particles->y[bodyIndex + offset],
        //           particles->z[bodyIndex + offset],
        //           particles->mass[bodyIndex + offset],
        //           particles->vx[bodyIndex + offset],
        //           particles->vy[bodyIndex + offset],
        //           particles->vz[bodyIndex + offset],
        //           particles->ax[bodyIndex + offset],
        //           particles->ay[bodyIndex + offset],
        //           particles->az[bodyIndex + offset]);
        //    if (particles->mass[bodyIndex + offset] < 1) {
        //        assert(0);
        //    }
        //}
        // end: debug

        offset += stride;
    }
}

real ExplicitEulerNS::Kernel::Launch::update(Particles *particles, integer n, real dt) {

    ExecutionPolicy executionPolicy;
    return cuda::launch(true, executionPolicy, ::ExplicitEulerNS::Kernel::update, particles, n, dt);

}
