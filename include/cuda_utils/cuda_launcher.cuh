#ifndef MILUPHPC_CUDALAUNCHER_CUH
#define MILUPHPC_CUDALAUNCHER_CUH

#include "../parameter.h"
#include "cuda_utilities.cuh"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>


class ExecutionPolicy {

public:
    const dim3 gridSize;
    const dim3 blockSize;
    const size_t sharedMemBytes;
    //const blockSizeInt = ;

    ExecutionPolicy();
    ExecutionPolicy(dim3 _gridSize, dim3 _blockSize, size_t _sharedMemBytes);
    ExecutionPolicy(dim3 _gridSize, dim3 _blockSize);
};

namespace cuda {
    template<typename... Arguments>
    real launch(bool timeKernel, const ExecutionPolicy &policy,
                    void (*f)(Arguments...),
                    Arguments... args) {
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
        } else {
            f<<<p.gridSize, p.blockSize, p.sharedMemBytes>>>(args...);
        }

        gpuErrorcheck( cudaPeekAtLastError() );
        gpuErrorcheck( cudaDeviceSynchronize() );

        return elapsedTime;
    }

    template<typename... Arguments>
    real launch(bool timeKernel, void(*f)(Arguments... args), Arguments... args) {
        cudaLaunch(ExecutionPolicy(), f, args...);
    }

}

//#elseif

#endif //MILUPHPC_CUDALAUNCHER_CUH
