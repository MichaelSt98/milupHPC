#ifndef MILUPHPC_SIMULATION_TIME_HANDLER_H
#define MILUPHPC_SIMULATION_TIME_HANDLER_H

#include "simulation_time.cuh"
#include "cuda_utils/cuda_runtime.h"
#include <boost/mpi.hpp>

class SimulationTimeHandler {

public:

    real *h_dt;
    real *h_startTime;
    real *h_endTime;
    real *h_currentTime;
    real *h_dt_max;

    SimulationTime *h_simulationTime;

    real *d_dt;
    real *d_startTime;
    real *d_endTime;
    real *d_currentTime;
    real *d_dt_max;

    SimulationTime *d_simulationTime;

    SimulationTimeHandler(real dt, real endTime, real dt_max);
    ~SimulationTimeHandler();

    void copy(To::Target target);

    void globalize(Execution::Location exLoc);

};


#endif //MILUPHPC_SIMULATION_TIME_HANDLER_H
