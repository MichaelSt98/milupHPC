#include "../../include/integrator/leapfrog.h"

/*Leapfrog::Leapfrog(SimulationParameters simulationParameters, integer numParticles,
                             integer numNodes) : Miluphpc(simulationParameters, numParticles, numNodes) {
    //integratedParticles = new IntegratedParticles[1];
    printf("Leapfrog()\n");
}*/

Leapfrog::Leapfrog(SimulationParameters simulationParameters) : Miluphpc(simulationParameters) {

    particleHandler->initLeapfrog();
    Logger(DEBUG) << "Leapfrog()";
}

Leapfrog::~Leapfrog() {

    particleHandler->freeLeapfrog();
    Logger(DEBUG) << "~Leapfrog()";
}

void Leapfrog::integrate(int step) {

    Timer timer;
    real time = 0.;

    real timeElapsed;

    Logger(INFO) << "Leapfrog::integrate()... currentTime: " << *simulationTimeHandler->h_currentTime
        << " | subEndTime: " << *simulationTimeHandler->h_subEndTime
        << " | endTime: " << *simulationTimeHandler->h_endTime;

    while (*simulationTimeHandler->h_currentTime < *simulationTimeHandler->h_subEndTime) {

        //profiler.setStep(subStep);

        if (step == 0 && subStep == 0) {
            Logger(TRACE) << "First rhs()";
            rhs(step, true, true);
        }

        subStep++;

        Logger(INFO) << "Leapfrog::integrate while...";

        timer.reset();
        if (simulationParameters.removeParticles) {
            time = removeParticles();
        }
        timeElapsed = timer.elapsed();
        profiler.value2file(ProfilerIds::Time::removeParticles, timeElapsed);
        Logger(TIME) << "removing particles: " << timeElapsed << " ms";

        Logger(INFO) << "rhs::loadBalancing()";
        timer.reset();
        if (simulationParameters.loadBalancing && step != 0 && step % simulationParameters.loadBalancingInterval == 0) {
            dynamicLoadBalancing(simulationParameters.loadBalancingBins);
        }
        real elapsed = timer.elapsed();
        //totalTime += elapsed;
        Logger(TIME) << "rhs::loadBalancing(): " << elapsed << " ms";
        profiler.value2file(ProfilerIds::Time::loadBalancing, elapsed);

        //Logger(INFO) << "checking for nans before update() ..";
        //ParticlesNS::Kernel::Launch::check4nans(particleHandler->d_particles, numParticlesLocal);

#if TARGET_GPU
        time = LeapfrogNS::Kernel::Launch::updateX(particleHandler->d_particles, numParticlesLocal,
                                                  *simulationTimeHandler->h_dt); //(real) simulationParameters.timestep);

#else
        // TODO: LeapfrogNS::updateX()
#endif // TARGET_GPU
        timer.reset();
        //real time;
        // -------------------------------------------------------------------------------------------------------------
        time = rhs(step, true, true);
        // -------------------------------------------------------------------------------------------------------------
        timeElapsed = timer.elapsed();
        Logger(TIME) << "rhs: " << time << " ms";
        Logger(TIME) << "rhsElapsed: " << timeElapsed << " ms";
        profiler.value2file(ProfilerIds::Time::rhs, time);
        profiler.value2file(ProfilerIds::Time::rhsElapsed, timeElapsed);

#if TARGET_GPU
        time = LeapfrogNS::Kernel::Launch::updateV(particleHandler->d_particles, numParticlesLocal,
                                                *simulationTimeHandler->h_dt); //(real) simulationParameters.timestep);
#else
        // TODO: LeapfrogNS::updateV()
#endif // TARGET_GPU

        profiler.value2file(ProfilerIds::Time::integrate, time);

        //Logger(INFO) << "timestep: " << (real) simulationParameters.timestep;

        *simulationTimeHandler->h_currentTime += *simulationTimeHandler->h_dt;
#if TARGET_GPU
        simulationTimeHandler->copy(To::device);
#endif

#if TARGET_GPU
#if INTEGRATE_SML
        cuda::set(particleHandler->d_dsmldt, (real)0, numParticles);
#endif
#endif

        Logger(TRACE) << "finished sub step - simulation time: " << *simulationTimeHandler->h_currentTime
                     << " (STEP: " << step << " | subStep: " << subStep
                     << " | time = " << *simulationTimeHandler->h_currentTime << "/"
                     << *simulationTimeHandler->h_subEndTime << "/"
                     << *simulationTimeHandler->h_endTime << ")";

#if TARGET_GPU
        subDomainKeyTreeHandler->copy(To::host, true, false);
#endif
        profiler.vector2file(ProfilerIds::ranges, subDomainKeyTreeHandler->h_range);

        boost::mpi::communicator comm;
        sumParticles = numParticlesLocal;
        all_reduce(comm, boost::mpi::inplace_t<integer*>(&sumParticles), 1, std::plus<integer>());

        profiler.value2file(ProfilerIds::numParticles, sumParticles);
        profiler.value2file(ProfilerIds::numParticlesLocal, numParticlesLocal);

    }

    //Logger(INFO) << "checking for nans after update()...";
    //ParticlesNS::Kernel::Launch::check4nans(particleHandler->d_particles, numParticlesLocal);



}