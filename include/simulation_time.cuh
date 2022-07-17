/**
 * @file simulation_time.cuh
 * @brief Simulation time related variables and functions instantiable on device and host.
 *
 * @author Michael Staneker
 * @bug no known bugs
 */
#ifndef MILUPHPC_SIMULATIONTIME_CUH
#define MILUPHPC_SIMULATIONTIME_CUH

#include "parameter.h"
#include "cuda_utils/cuda_utilities.cuh"

class SimulationTime {
public:

    real *dt;
    real *startTime;
    real *subEndTime;
    real *endTime;
    real *currentTime;
    real *dt_max;

    CUDA_CALLABLE_MEMBER SimulationTime();
    CUDA_CALLABLE_MEMBER SimulationTime(real *startTime, real *endTime, real *dt);
    CUDA_CALLABLE_MEMBER ~SimulationTime();

    CUDA_CALLABLE_MEMBER void set(real *dt, real *startTime, real *subEndTime, real *endTime,
                                  real *currentTime, real *dt_max);

};

#if TARGET_GPU
namespace SimulationTimeNS {
    namespace Kernel {
        __global__ void set(SimulationTime *simulationTime, real *dt, real *startTime, real *subEndTime, real *endTime,
                            real *currentTime, real *dt_max);

        namespace Launch {
            void set(SimulationTime *simulationTime, real *dt, real *startTime, real *subEndTime, real *endTime,
                     real *currentTime, real *dt_max);
        }
    }
}
#endif // TARGET_GPU

#endif //MILUPHPC_SIMULATIONTIME_CUH
