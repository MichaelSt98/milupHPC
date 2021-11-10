#include "../../include/integrator/explicit_euler.h"

/*ExplicitEuler::ExplicitEuler(SimulationParameters simulationParameters, integer numParticles,
                             integer numNodes) : Miluphpc(simulationParameters, numParticles, numNodes) {
    //integratedParticles = new IntegratedParticles[1];
    printf("ExplicitEuler()\n");
}*/

ExplicitEuler::ExplicitEuler(SimulationParameters simulationParameters) : Miluphpc(simulationParameters) {
    printf("ExplicitEuler()\n");
}

ExplicitEuler::~ExplicitEuler() {
    printf("~ExplicitEuler()\n");
}

void ExplicitEuler::integrate(int step) {

    Timer timer;

    removeParticles();

    Logger(INFO) << "rhs::loadBalancing()";
    timer.reset();
    if (simulationParameters.loadBalancing && step != 0 && step % simulationParameters.loadBalancingInterval == 0) {
        dynamicLoadBalancing();
    }
    real elapsed = timer.elapsed();
    //totalTime += elapsed;
    Logger(TIME) << "rhs::loadBalancing(): " << elapsed << " ms";
    profiler.value2file(ProfilerIds::Time::loadBalancing, elapsed);

    //Logger(INFO) << "checking for nans before update() ..";
    //ParticlesNS::Kernel::Launch::check4nans(particleHandler->d_particles, numParticlesLocal);

    printf("ExplicitEuler::integrate()\n");
    timer.reset();
    real time;
    time = rhs(step, true);
    real time_elapsed = timer.elapsed();
    Logger(TIME) << "rhs: " << time << " ms";
    Logger(TIME) << "rhs elapsed: " << time_elapsed  << " ms";

    ExplicitEulerNS::Kernel::Launch::update(particleHandler->d_particles, numParticlesLocal,
                                    (real)simulationParameters.timestep);

#if INTEGRATE_SML
    cuda::set(particleHandler->d_dsmldt, (real)0, numParticles);
#endif

    //Logger(INFO) << "checking for nans after update()...";
    //ParticlesNS::Kernel::Launch::check4nans(particleHandler->d_particles, numParticlesLocal);

    //H5Profiler &profiler = H5Profiler::getInstance("log/performance.h5");
    profiler.value2file(ProfilerIds::Time::rhs, time);
    profiler.value2file(ProfilerIds::Time::rhsElapsed, time_elapsed);

    subDomainKeyTreeHandler->copy(To::host, true, false);
    profiler.vector2file(ProfilerIds::ranges, subDomainKeyTreeHandler->h_range);

    boost::mpi::communicator comm;
    sumParticles = numParticlesLocal;
    all_reduce(comm, boost::mpi::inplace_t<integer*>(&sumParticles), 1, std::plus<integer>());

    profiler.value2file(ProfilerIds::numParticles, sumParticles);
    profiler.value2file(ProfilerIds::numParticlesLocal, numParticlesLocal);


}