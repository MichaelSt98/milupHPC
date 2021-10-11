#include "../../include/materials/material.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

CUDA_CALLABLE_MEMBER Material::Material() {

}

CUDA_CALLABLE_MEMBER Material::~Material() {

}

CUDA_CALLABLE_MEMBER void Material::info() {
    printf("Material: ID                      = %i\n", ID);
    printf("Material: interactions            = %i\n", interactions);
    printf("Material: alpha                   = %f\n", artificialViscosity.alpha);
    printf("Material: eos: type               = %i\n", eos.type);
    printf("Material: eos: polytropic_K       = %f\n", eos.polytropic_K);
    printf("Material: eos: polytropic_gamma   = %f\n", eos.polytropic_gamma);
}

namespace MaterialNS {
    namespace Kernel {
        __global__ void info(Material *material) {
            material->info();
        }

        void Launch::info(Material *material) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::MaterialNS::Kernel::info, material);
        }
    }
}


CUDA_CALLABLE_MEMBER ArtificialViscosity::ArtificialViscosity() : alpha(0.0), beta(0.0) {

}
CUDA_CALLABLE_MEMBER ArtificialViscosity::ArtificialViscosity(real alpha, real beta) : alpha(alpha), beta(beta) {

}

CUDA_CALLABLE_MEMBER EqOfSt::EqOfSt() : type(0), polytropic_K(0.), polytropic_gamma(0.) {

}

CUDA_CALLABLE_MEMBER EqOfSt::EqOfSt(int type, real polytropic_K, real polytropic_gamma) : type(type),
                            polytropic_K(polytropic_K), polytropic_gamma(polytropic_gamma) {

}

