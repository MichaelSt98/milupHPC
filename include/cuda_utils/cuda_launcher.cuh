//
// Created by Michael Staneker on 05.08.21.
//

#ifndef MILUPHPC_CUDALAUNCHER_CUH
#define MILUPHPC_CUDALAUNCHER_CUH

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>


class ExecutionPolicy {

public:
    const dim3 gridSize;
    const dim3 blockSize;
    const size_t sharedMemBytes;

    ExecutionPolicy();
    ExecutionPolicy(dim3 _gridSize, dim3 _blockSize, size_t _sharedMemBytes);
    ExecutionPolicy(dim3 _gridSize, dim3 _blockSize);
};

template <typename... Arguments>
float cudaLaunch(bool timeKernel, const ExecutionPolicy &policy,
                 void (*f)(Arguments...),
                 Arguments... args)
{
    float elapsedTime = 0.f;
    ExecutionPolicy p = policy;
    //checkCuda(configureGrid(p, f));
    if (timeKernel) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        f<<<p.gridSize, p.blockSize, p.sharedMemBytes>>>(args...);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        f<<<p.gridSize, p.blockSize, p.sharedMemBytes>>>(args...);
    }

    return elapsedTime;
}

template <typename... Arguments>
float cudaLaunch(bool timeKernel, void(*f)(Arguments... args), Arguments... args)
{
    cudaLaunch(ExecutionPolicy(), f, args...);
}


#endif //MILUPHPC_CUDALAUNCHER_CUH
