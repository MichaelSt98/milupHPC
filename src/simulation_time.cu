#include "../include/simulation_time.cuh"
#include "../include/cuda_utils/cuda_launcher.cuh"


CUDA_CALLABLE_MEMBER SimulationTime::SimulationTime() {

}
CUDA_CALLABLE_MEMBER SimulationTime::SimulationTime(real *startTime, real *endTime, real *dt) :
                                startTime(startTime), endTime(endTime), dt(dt) {

}

CUDA_CALLABLE_MEMBER SimulationTime::~SimulationTime() {

}

CUDA_CALLABLE_MEMBER void SimulationTime::set(real *dt, real *startTime, real *subEndTime, real *endTime,
                                              real *currentTime, real *dt_max) {
    this->dt = dt;
    this->startTime = startTime;
    this->subEndTime = subEndTime;
    this->endTime = endTime;
    this->currentTime = currentTime;
    this->dt_max = dt_max;
}

#if TARGET_GPU
namespace SimulationTimeNS {
    namespace Kernel {
        __global__ void set(SimulationTime *simulationTime, real *dt, real *startTime, real *subEndTime, real *endTime,
                            real *currentTime, real *dt_max) {
            simulationTime->set(dt, startTime, subEndTime, endTime, currentTime, dt_max);
        }

        namespace Launch {
            void set(SimulationTime *simulationTime, real *dt, real *startTime, real *subEndTime, real *endTime,
                     real *currentTime, real *dt_max) {
                ExecutionPolicy executionPolicy(1, 1);
                cuda::launch(false, executionPolicy, ::SimulationTimeNS::Kernel::set, simulationTime, dt, startTime,
                             subEndTime, endTime, currentTime, dt_max);
            }
        }
    }
}
#endif // TARGET_GPU