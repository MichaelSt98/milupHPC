#include "../../include/integrator/godunov.h"

Godunov::Godunov(SimulationParameters simulationParameters) : Miluphpc(simulationParameters) {
    Logger(DEBUG) << "Godunov()";
    // TODO: init Leapfrog to be able to compute time-centered gravity source terms
}

Godunov::~Godunov() {
    Logger(DEBUG) << "~Godunov()";
}

void Godunov::integrate(int step){

    Timer timer;
    real time = 0.;

    real timeElapsed;

    Timer timerRhs;

    Logger(INFO) << "Godunov::integrate()... currentTime: " << *simulationTimeHandler->h_currentTime
                 << " | subEndTime: " << *simulationTimeHandler->h_subEndTime
                 << " | endTime: " << *simulationTimeHandler->h_endTime;

    while (*simulationTimeHandler->h_currentTime < *simulationTimeHandler->h_subEndTime) {

        profiler.setStep(subStep);
        subStep++;

        Logger(INFO) << "Godunov::integrate while...";

        timer.reset();
        if (simulationParameters.removeParticles) {
            time = removeParticles();
        }
        timeElapsed = timer.elapsed();
        profiler.value2file(ProfilerIds::Time::removeParticles, timeElapsed);
        Logger(TIME) << "removing particles: " << timeElapsed << " ms";

        Logger(INFO) << "rhs::loadBalancing()";
        if (simulationParameters.loadBalancing && step != 0 && step % simulationParameters.loadBalancingInterval == 0) {
            dynamicLoadBalancing();
        }

        timerRhs.reset();

        // TODO: set timestep

        // -------------------------------------------------------------------------------------------------------------
        time += rhs(step, true, true);
        // -------------------------------------------------------------------------------------------------------------
        timeElapsed = timerRhs.elapsed();
        Logger(TIME) << "rhsElapsed: " << timeElapsed;
        //Logger(TIME) << "rhs: " << time << " ms";
        profiler.value2file(ProfilerIds::Time::rhs, time);
        profiler.value2file(ProfilerIds::Time::rhsElapsed, timeElapsed);

        // TODO: what else is needed here?

        Logger(INFO) << "Timestep update with Godunov scheme";
        time += GodunovNS::Kernel::Launch::update(particleHandler->d_particles, numParticlesLocal,
                                                 *simulationTimeHandler->h_dt);

        profiler.value2file(ProfilerIds::Time::integrate, time);

        //Logger(INFO) << "timestep: " << (real) simulationParameters.timestep;

        *simulationTimeHandler->h_currentTime += *simulationTimeHandler->h_dt;
        simulationTimeHandler->copy(To::device);

        Logger(TRACE) << "finished sub step - simulation time: " << *simulationTimeHandler->h_currentTime
                      << " (STEP: " << step << " | subStep: " << subStep
                      << " | time = " << *simulationTimeHandler->h_currentTime << "/"
                      << *simulationTimeHandler->h_subEndTime << "/"
                      << *simulationTimeHandler->h_endTime << ")";

        subDomainKeyTreeHandler->copy(To::host, true, false);
        profiler.vector2file(ProfilerIds::ranges, subDomainKeyTreeHandler->h_range);

        boost::mpi::communicator comm;
        sumParticles = numParticlesLocal;
        all_reduce(comm, boost::mpi::inplace_t<integer*>(&sumParticles), 1, std::plus<integer>());

        profiler.value2file(ProfilerIds::numParticles, sumParticles);
        profiler.value2file(ProfilerIds::numParticlesLocal, numParticlesLocal);

        timeElapsed = timer.elapsed();
        Logger(TIME) << "integration step elapsed: " << timeElapsed << " ms";

    }

}