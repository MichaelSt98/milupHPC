#include "../../include/integrator/device_godunov.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

__global__ void GodunovNS::Kernel::update(Particles *particles, integer n, real dt) {

    integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    integer stride = blockDim.x * gridDim.x;
    integer offset = 0;

    while (bodyIndex + offset < n) {
        // TODO: actual update
        offset += stride;
    }
}

real GodunovNS::Kernel::Launch::update(Particles *particles, integer n, real dt) {

    ExecutionPolicy executionPolicy;
    return cuda::launch(true, executionPolicy, ::GodunovNS::Kernel::update, particles, n, dt);

}
