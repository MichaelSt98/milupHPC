#ifndef MILUPHPC_DEVICE_EXPLICIT_EULER_CUH
#define MILUPHPC_DEVICE_EXPLICIT_EULER_CUH

#endif //MILUPHPC_DEVICE_EXPLICIT_EULER_CUH

#include "../particles.cuh"
#include <assert.h>

namespace ExplicitEulerNS {

    namespace Kernel {

        __global__ void update(Particles *particles, integer n, real dt);

        namespace Launch {
            real update(Particles *particles, integer n, real dt);
        }
    }

}