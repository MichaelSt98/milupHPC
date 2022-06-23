/**
 * @file kernel_handler.h
 * @brief Handling the SPH smoothing kernels.
 *
 * @author Michael Staneker
 * @bug no known bugs
 */
#ifndef MILUPHPC_KERNEL_HANDLER_H
#define MILUPHPC_KERNEL_HANDLER_H

#include "kernel.cuh"
#include "../parameter.h"

namespace SPH {

    /**
     * @brief SPH smoothing kernel handler.
     */
    class KernelHandler {

    public:

        //SPHKernel *kernel;

        /// SPH smoothing kernel typedef/kind of function pointer
        SPH_kernel kernel;

        /**
         * @brief Default constructor.
         */
        KernelHandler();

        /**
         * @brief Constructor choosing the SPH smoothing kernel.
         *
         * @param smoothingKernel SPH smoothing kernel selection
         */
        KernelHandler(Smoothing::Kernel smoothingKernel);

        /**
         * @brief Destructor.
         */
        ~KernelHandler();

    };
}


#endif //MILUPHPC_KERNEL_HANDLER_H
