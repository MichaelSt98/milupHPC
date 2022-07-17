#include "../include/device_rhs.cuh"
#include "../include/cuda_utils/cuda_launcher.cuh"

#if TARGET_GPU
namespace Kernel {

    __global__ void resetArrays(Tree *tree, Particles *particles, integer *mutex, integer n, integer m) {

        integer bodyIndex = threadIdx.x + blockDim.x*blockIdx.x;
        integer stride = blockDim.x*gridDim.x;
        integer offset = 0;

        while ((bodyIndex + offset) < m) {
            tree->reset(bodyIndex + offset, n);

            if ((bodyIndex + offset) >= n) {
                particles->reset(bodyIndex + offset);
            }

            offset += stride;
        }

        if (bodyIndex == 0) {
            *mutex = 0;
            *tree->index = n;
            *tree->minX = 0;
            *tree->maxX = 0;
#if DIM > 1
            *tree->minY = 0;
            *tree->maxY = 0;
#if DIM == 3
            *tree->minZ = 0;
            *tree->maxZ = 0;
#endif
#endif
            tree->toDeleteLeaf[0] = -1;
            tree->toDeleteLeaf[1] = -1;
            tree->toDeleteNode[0] = -1;
            tree->toDeleteNode[1] = -1;

        }
    }

    real Launch::resetArrays(Tree *tree, Particles *particles, integer *mutex, integer n, integer m, bool time) {
        ExecutionPolicy executionPolicy;
        return cuda::launch(time, executionPolicy, ::Kernel::resetArrays, tree, particles, mutex,  n, m);
    }

}
#endif

