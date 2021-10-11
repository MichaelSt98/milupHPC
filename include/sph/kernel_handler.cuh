#ifndef MILUPHPC_KERNEL_HANDLER_H
#define MILUPHPC_KERNEL_HANDLER_H

#include "kernel.cuh"
#include "../parameter.h"

namespace SPH {

    class KernelHandler {

    public:

        //SPHKernel *kernel;
        SPH_kernel kernel;

        KernelHandler();
        KernelHandler(Smoothing::Kernel smoothingKernel);
        ~KernelHandler();

    };
}


#endif //MILUPHPC_KERNEL_HANDLER_H
