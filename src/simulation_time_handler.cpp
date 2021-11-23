#include "../include/simulation_time_handler.h"

SimulationTimeHandler::SimulationTimeHandler(real dt, real endTime, real dt_max) {
    h_dt = new real;
    h_startTime = new real;
    h_endTime = new real;
    h_currentTime = new real;
    h_dt_max = new real;

    *h_dt = dt;
    *h_startTime = 0;
    *h_endTime = endTime;
    *h_currentTime = 0;
    *h_dt_max = dt_max;

    h_simulationTime = new SimulationTime;
    h_simulationTime->set(h_dt, h_startTime, h_endTime, h_currentTime, h_dt_max);

    cuda::malloc(d_dt, 1);
    cuda::malloc(d_startTime, 1);
    cuda::malloc(d_endTime, 1);
    cuda::malloc(d_currentTime, 1);
    cuda::malloc(d_dt_max, 1);

    cuda::malloc(d_simulationTime, 1);
    SimulationTimeNS::Kernel::Launch::set(d_simulationTime, d_dt, d_startTime, d_endTime, d_currentTime, d_dt_max);

    copy(To::device);
}

SimulationTimeHandler::~SimulationTimeHandler() {
    delete h_dt;
    delete h_startTime;
    delete h_endTime;
    delete h_currentTime;
    delete h_dt_max;

    delete h_simulationTime;

    cuda::free(d_dt);
    cuda::free(d_startTime);
    cuda::free(d_endTime);
    cuda::free(d_currentTime);
    cuda::free(d_dt_max);

    cuda::free(d_simulationTime);
}

void SimulationTimeHandler::copy(To::Target target) {

    cuda::copy(h_dt, d_dt, 1, target);
    cuda::copy(h_startTime, d_startTime, 1, target);
    cuda::copy(h_endTime, d_endTime, 1, target);
    cuda::copy(h_currentTime, d_currentTime, 1, target);
    cuda::copy(h_dt_max, d_dt_max, 1, target);

}

void SimulationTimeHandler::globalize(Execution::Location exLoc) {

    boost::mpi::communicator comm;
    switch (exLoc) {

        case Execution::host: {
            all_reduce(comm, boost::mpi::inplace_t<real*>(h_dt), 1, boost::mpi::minimum<real>());
        } break;
        case Execution::device: {
            all_reduce(comm, boost::mpi::inplace_t<real*>(d_dt), 1, boost::mpi::minimum<real>());
        } break;

    }

}
