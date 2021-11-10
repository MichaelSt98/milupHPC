#include "../../include/sph/soundspeed.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

__global__ void SPH::Kernel::initializeSoundSpeed(Particles *particles, Material *materials, int numParticles) {

    register int i, inc, matId;
    inc = blockDim.x * gridDim.x;

    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        matId = particles->materialId[i];

        switch (materials[matId].eos.type) {
            case EquationOfStates::EOS_TYPE_POLYTROPIC_GAS: {
                particles->cs[i] = 0.0; // for gas this will be calculated each step by kernel calculateSoundSpeed
            } break;
            case EquationOfStates::EOS_TYPE_ISOTHERMAL_GAS: {
                particles->cs[i] = 203.0; // this is pure molecular hydrogen at 10 K
#if !SI_UNITS
                particles->cs[i] /= 2.998e8; // speed of light
#endif
            } break;
            case EquationOfStates::EOS_TYPE_LOCALLY_ISOTHERMAL_GAS: {
                //TODO: initial sound speed for EOS_TYPE_LOCALLY_ISOTHERMAL_GAS?
            } break;
            default:
                printf("not implemented!\n");
        }
    }

}

__global__ void SPH::Kernel::calculateSoundSpeed(Particles *particles, Material *materials, int numParticles) {

    register int i, inc, matId;
    int d;
    int j;
    double m_com;
    register double cs, rho, pressure, eta, omega0, z, cs_sq,  cs_c_sq, cs_e_sq, Gamma_e, mu, y; //Gamma_c;
    int i_rho, i_e;

    inc = blockDim.x * gridDim.x;

    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {

        matId = particles->materialId[i];

        switch (materials[matId].eos.type) {
            case EquationOfStates::EOS_TYPE_POLYTROPIC_GAS: {
                particles->cs[i] = cuda::math::sqrt(materials[matId].eos.polytropic_K *
                                        pow(particles->rho[i], materials[matId].eos.polytropic_gamma-1.0));
            } break;
            //case EquationOfStates::EOS_TYPE_ISOTHERMAL_GAS: {
            //    // do nothing since constant
            //} break;
            case EquationOfStates::EOS_TYPE_LOCALLY_ISOTHERMAL_GAS: {
                real distance = 0.0;
                distance = particles->x[i] * particles->x[i];
#if DIM > 1
                distance += particles->y[i] * particles->y[i];
#if DIM == 3
                distance += particles->z[i] * particles->z[i];
#endif
#endif
                distance = cuda::math::sqrt(distance);
                m_com = 0;
                //TODO: how to calculate cs for EOS_TYPE_ISOTHERMAL_GAS
                //for (j = 0; j < numPointmasses; j++) {
                //    m_com += pointmass.m[j];
                //}
                //double vkep = cuda::math::sqrt(gravConst * m_com/distance);
                //p.cs[i] = vkep * scale_height;
                particles->cs[i] = 0;
            } break;
            //default:
                //printf("not implemented!\n");
        }
    }

}

real SPH::Kernel::Launch::initializeSoundSpeed(Particles *particles, Material *materials, int numParticles) {
    ExecutionPolicy executionPolicy;
    return cuda::launch(true, executionPolicy, ::SPH::Kernel::initializeSoundSpeed, particles, materials, numParticles);
}

real SPH::Kernel::Launch::calculateSoundSpeed(Particles *particles, Material *materials, int numParticles) {
    ExecutionPolicy executionPolicy;
    return cuda::launch(true, executionPolicy, ::SPH::Kernel::calculateSoundSpeed, particles, materials, numParticles);
}

