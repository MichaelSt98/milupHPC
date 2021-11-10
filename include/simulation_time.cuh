#ifndef MILUPHPC_SIMULATIONTIME_CUH
#define MILUPHPC_SIMULATIONTIME_CUH

#include "parameter.h"
#include "cuda_utils/cuda_utilities.cuh"

class SimulationTime {
public:

    real *dt;
    real *startTime;
    real *endTime;
    real *currentTime;
    real *dt_max;

    CUDA_CALLABLE_MEMBER SimulationTime();
    CUDA_CALLABLE_MEMBER SimulationTime(real *startTime, real *endTime, real *dt);
    CUDA_CALLABLE_MEMBER ~SimulationTime();

    CUDA_CALLABLE_MEMBER void set(real *dt, real *startTime, real *endTime, real *currentTime, real *dt_max);

};

namespace SimulationTimeNS {
    namespace Kernel {
        __global__ void set(SimulationTime *simulationTime, real *dt, real *startTime, real *endTime,
                            real *currentTime, real *dt_max);

        namespace Launch {
            void set(SimulationTime *simulationTime, real *dt, real *startTime, real *endTime, real *currentTime,
                     real *dt_max);
        }
    }
}


#endif //MILUPHPC_SIMULATIONTIME_CUH
