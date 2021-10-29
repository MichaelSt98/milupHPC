#ifndef MILUPHPC_DEVICE_PREDICTOR_CORRECTOR_EULER_CUH
#define MILUPHPC_DEVICE_PREDICTOR_CORRECTOR_EULER_CUH

#include "../particles.cuh"

namespace PredictorCorrectorEulerNS {

    namespace Kernel {

        __global__ void corrector(Particles *particles, IntegratedParticles *predictor, real dt, int numParticles);
        __global__ void predictor(Particles *particles, IntegratedParticles *predictor, real dt, int numParticles);

        __global__ void setTimeStep();

        __global__ void pressureChangeCheck();

        namespace Launch {

            real corrector(Particles *particles, IntegratedParticles *predictor, real dt, int numParticles);
            real predictor(Particles *particles, IntegratedParticles *predictor, real dt, int numParticles);

            real setTimeStep();

            real pressureChangeCheck();

        }

    }
}

#endif //MILUPHPC_DEVICE_PREDICTOR_CORRECTOR_EULER_CUH
