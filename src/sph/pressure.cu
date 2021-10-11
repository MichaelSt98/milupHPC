#include "../../include/sph/pressure.cuh"
#include "../include/cuda_utils/cuda_launcher.cuh"


namespace EOS {
    __device__ void polytropicGas(Material *materials, Particles *particles, int index) {
        particles->p[index] = materials[particles->materialId[index]].eos.polytropic_K *
                pow(particles->rho[index], materials[particles->materialId[index]].eos.polytropic_gamma);
        //if (true /*particles->p[index] > 0.*/) {
        //    printf("pressure: p[%i] = %f, rho[%i] = %f, polyTropic_K = %f, polytropic_gamma = %f\n", index,
        //           particles->p[index], index, particles->rho[index], materials[particles->materialId[index]].eos.polytropic_K,
        //           materials[particles->materialId[index]].eos.polytropic_K);
        //}
    }

    __device__ void isothermalGas(Material *materials, Particles *particles, int index) {
        particles->p[index] = 41255.407 * particles->rho[index];
    }

    __device__ void locallyIsothermalGas(Material *materials, Particles *particles, int index) {
        particles->p[index] = particles->cs[index] * particles->cs[index] * particles->rho[index];
    }
}

namespace SPH {
    namespace Kernel {
        __global__ void calculatePressure(Material *materials, Particles *particles, int numParticles) {

            register int i, inc;
            register double eta, e, rho, mu, p1, p2;
            int i_rho, i_e;
            double pressure;

            inc = blockDim.x * gridDim.x;
            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {

                pressure = 0.0;

                switch (materials[particles->materialId[i]].ID) {
                    case EquationOfStates::EOS_TYPE_POLYTROPIC_GAS: {
                        ::EOS::polytropicGas(materials, particles, i);
                    }
                        break;
                    case EquationOfStates::EOS_TYPE_ISOTHERMAL_GAS: {
                        ::EOS::isothermalGas(materials, particles, i);
                    }
                        break;
                    case EquationOfStates::EOS_TYPE_LOCALLY_ISOTHERMAL_GAS: {
                        ::EOS::locallyIsothermalGas(materials, particles, i);
                    }
                        break;
                    default:
                        printf("not implemented!\n");
                }

            }
        }

        real Launch::calculatePressure(Material *materials, Particles *particles, int numParticles) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SPH::Kernel::calculatePressure, materials,
                                particles, numParticles);
        }
    }
}

