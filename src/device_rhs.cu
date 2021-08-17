//
// Created by Michael Staneker on 15.08.21.
//

#include "../include/device_rhs.cuh"
#include "../include/cuda_utils/cuda_launcher.cuh"

namespace device {

    __global__ void resetArraysKernel(Tree *tree, Particles *particles, integer *mutex, integer n, integer m) {

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

        }
    }

    void launchResetArraysKernel(Tree *tree, Particles *particles, integer *mutex, integer n, integer m) {
        ExecutionPolicy executionPolicy;
        cuda::launch(false, executionPolicy, resetArraysKernel, tree, particles, mutex,  n, m);
    }

}

