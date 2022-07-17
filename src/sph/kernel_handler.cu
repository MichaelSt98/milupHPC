#include "../../include/sph/kernel_handler.cuh"
#if TARGET_GPU

extern __device__ SPH::SPH_kernel spiky_p;
extern __device__ SPH::SPH_kernel cubicSpline_p;
extern __device__ SPH::SPH_kernel wendlandc2_p;
extern __device__ SPH::SPH_kernel wendlandc4_p;
extern __device__ SPH::SPH_kernel wendlandc6_p;

SPH::KernelHandler::KernelHandler() {

}

SPH::KernelHandler::KernelHandler(Smoothing::Kernel smoothingKernel) {

    switch (smoothingKernel) {
        case Smoothing::spiky: {
            cudaMemcpyFromSymbol(&kernel, spiky_p, sizeof(SPH_kernel));
        } break;
        case Smoothing::cubic_spline: {
            cudaMemcpyFromSymbol(&kernel, cubicSpline_p, sizeof(SPH_kernel));
        } break;
        case Smoothing::wendlandc2: {
            cudaMemcpyFromSymbol(&kernel, wendlandc2_p, sizeof(SPH_kernel));
        } break;
        case Smoothing::wendlandc4: {
            cudaMemcpyFromSymbol(&kernel, wendlandc4_p, sizeof(SPH_kernel));
        } break;
        case Smoothing::wendlandc6: {
            cudaMemcpyFromSymbol(&kernel, wendlandc6_p, sizeof(SPH_kernel));
        } break;
        default:
            printf("Not available!\n");
    }


}

SPH::KernelHandler::~KernelHandler() {

}

#endif