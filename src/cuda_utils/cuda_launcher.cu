#include "../../include/cuda_utils/cuda_launcher.cuh"

#if TARGET_GPU

ExecutionPolicy::ExecutionPolicy() : gridSize(256), blockSize(256), sharedMemBytes(0) {};

ExecutionPolicy::ExecutionPolicy(dim3 _gridSize, dim3 _blockSize, size_t _sharedMemBytes)
        : gridSize(_gridSize), blockSize(_blockSize), sharedMemBytes(_sharedMemBytes) {};

ExecutionPolicy::ExecutionPolicy(dim3 _gridSize, dim3 _blockSize)
        : gridSize(_gridSize), blockSize(_blockSize), sharedMemBytes(0) {};

#endif // TARGET_GPU