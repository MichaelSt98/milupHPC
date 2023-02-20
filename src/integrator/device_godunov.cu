#include "../../include/integrator/device_godunov.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

__global__ void GodunovNS::Kernel::update(Particles *particles, integer n, real dt) {
}

real GodunovNS::Kernel::Launch::update(Particles *particles, integer n, real dt) {

    ExecutionPolicy executionPolicy;
    return cuda::launch(true, executionPolicy, ::GodunovNS::Kernel::update, particles, n, dt);

}
