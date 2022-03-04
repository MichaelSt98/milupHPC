/**
 * @file cuda_runtime.h
 * @brief CUDA runtime functionalities and wrappers.
 *
 * Wrapping CUDA runtime functions for better C++ integration.
 *
 * @author Michael Staneker
 * @bug no known bugs
 */
#ifndef MILUPHPC_CUDA_RUNTIME_H
#define MILUPHPC_CUDA_RUNTIME_H

#include "cuda_utilities.cuh"
#include "../parameter.h"

namespace cuda {

    /**
     * @brief Copy between host and device and vice-versa.
     *
     * Wrapping CUDA runtime function `cudaMemcpy()`.
     *
     * @tparam T Variable type to be copied
     * @param h_var Host variable
     * @param d_var Device variable
     * @param count Amount of variable to be copied
     * @param copyTo Copy from device to host or from host to device
     */
    template <typename T>
    void copy(T *h_var, T *d_var, std::size_t count = 1, To::Target copyTo = To::device) {

        switch (copyTo) {
            case To::device: {
                gpuErrorcheck(cudaMemcpy(d_var, h_var, count * sizeof(T), cudaMemcpyHostToDevice));
            } break;
            case To::host: {
                gpuErrorcheck(cudaMemcpy(h_var, d_var, count * sizeof(T), cudaMemcpyDeviceToHost));
            } break;
            default: {
                printf("cuda::copy Target not available!\n");
            }
        }
    }

    /**
     * @brief Set device memory to a specific value.
     *
     * Wrapping CUDA runtime function `cudaMemset()`.
     *
     * @tparam T Variable type to be set
     * @param d_var Device variable or memory to be set
     * @param val Value the device memory to be set
     * @param count Amount of variables or subsequent memory location to be set
     */
    template <typename T>
    void set(T *d_var, T val, std::size_t count = 1) {
        gpuErrorcheck(cudaMemset(d_var, val, count * sizeof(T)));
    }

    /**
     * @brief Allocate device memory.
     *
     * Wrapping CUDA runtime function `cudaMalloc()`.
     *
     * @tparam T Variable type to be allocated
     * @param d_var Device variable pointing to device memory allocated
     * @param count Amount of variables or subsequent memory location to be allocated
     */
    template <typename T>
    void malloc(T *&d_var, std::size_t count) {
        gpuErrorcheck(cudaMalloc((void**)&d_var, count * sizeof(T)));
    }

    /**
     * @brief Free device memory.
     *
     * @tparam T Variable type to be freed
     * @param d_var Device variable to be freed
     */
    template <typename T>
    void free(T *d_var) {
        gpuErrorcheck(cudaFree(d_var));
    }

}


#endif //MILUPHPC_CUDA_RUNTIME_H
