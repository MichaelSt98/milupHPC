#include "../../include/integrator/predictor_corrector_euler.h"

PredictorCorrectorEuler::PredictorCorrectorEuler(SimulationParameters simulationParameters) : Miluphpc(simulationParameters) {
    printf("PredictorCorrectorEuler()\n");
    integratedParticles = new IntegratedParticleHandler(numParticles, numNodes);


    cudaGetDevice(&device);
    printf("Device: %i\n", device);
    cudaGetDeviceProperties(&prop, device);
    //printf("prop.multiProcessorCount: %i\n", prop.multiProcessorCount);
    cuda::malloc(d_blockCount, 1);
    cuda::set(d_blockCount, 0, 1);

    cuda::malloc(d_block_forces, prop.multiProcessorCount);
    cuda::malloc(d_block_courant, prop.multiProcessorCount);
    cuda::malloc(d_block_artVisc, prop.multiProcessorCount);
    cuda::malloc(d_block_e, prop.multiProcessorCount);
    cuda::malloc(d_block_rho, prop.multiProcessorCount);
    cuda::malloc(d_block_vmax, prop.multiProcessorCount);

    cuda::malloc(d_blockShared, 1);

    PredictorCorrectorEulerNS::BlockSharedNS::Launch::set(d_blockShared, d_block_forces, d_block_courant,
                                                  d_block_courant);
    PredictorCorrectorEulerNS::BlockSharedNS::Launch::setE(d_blockShared, d_block_e);
    PredictorCorrectorEulerNS::BlockSharedNS::Launch::setRho(d_blockShared, d_block_rho);
    PredictorCorrectorEulerNS::BlockSharedNS::Launch::setVmax(d_blockShared, d_block_vmax);

}

PredictorCorrectorEuler::~PredictorCorrectorEuler() {
    printf("~PredictorCorrectorEuler()\n");

    delete [] integratedParticles;

    cuda::free(d_block_forces);
    cuda::free(d_block_courant);
    cuda::free(d_block_artVisc);
    cuda::free(d_block_e);
    cuda::free(d_block_rho);
    cuda::free(d_block_vmax);

    cuda::free(d_blockShared);
}

void PredictorCorrectorEuler::integrate(int step) {

    printf("PredictorCorrector::integrate()\n");

    Timer timer;
    real time = 0.;

    real timeElapsed;

    Timer timerRhs;

    while (*simulationTimeHandler->h_currentTime < *simulationTimeHandler->h_subEndTime) {

        profiler.setStep(subStep);
        subStep++;

        timer.reset();

#if PERIODIC_BOUNDARIES
        //TODO: write time to profiler
        time = moveParticlesPeriodic();
#else
        if (simulationParameters.removeParticles) {
            time = removeParticles();
        }
        timeElapsed = timer.elapsed();
        profiler.value2file(ProfilerIds::Time::removeParticles, timeElapsed);
        Logger(TIME) << "removing particles: " << timeElapsed << " ms";
#endif
        Logger(INFO) << "rhs::loadBalancing()";
        if (simulationParameters.loadBalancing && step != 0 && step % simulationParameters.loadBalancingInterval == 0) {
            dynamicLoadBalancing();
        }

        timerRhs.reset();
        // -------------------------------------------------------------------------------------------------------------
        time += rhs(step, true, true);
        // -------------------------------------------------------------------------------------------------------------
        timeElapsed = timerRhs.elapsed();
        Logger(TIME) << "rhsElapsed: " << timeElapsed;
        //Logger(TIME) << "rhs: " << time << " ms";
        profiler.value2file(ProfilerIds::Time::rhs, time);
        profiler.value2file(ProfilerIds::Time::rhsElapsed, timeElapsed);

        // ------------------------------------------------------------------------------------------------------------
        //simulationTimeHandler->copy(To::host);
        //Logger(INFO) << "h_dt = " << *simulationTimeHandler->h_dt;
        //Logger(INFO) << "h_startTime = " << *simulationTimeHandler->h_startTime;
        //Logger(INFO) << "h_subEndTime = " << *simulationTimeHandler->h_subEndTime;
        //Logger(INFO) << "h_endTime = " << *simulationTimeHandler->h_endTime;
        //Logger(INFO) << "h_currentTime = " << *simulationTimeHandler->h_currentTime;
        //Logger(INFO) << "h_dt_max = " << *simulationTimeHandler->h_dt_max;

        Logger(INFO) << "setTimeStep: search radius: " << h_searchRadius;
        PredictorCorrectorEulerNS::Kernel::Launch::setTimeStep(prop.multiProcessorCount,
                                                               simulationTimeHandler->d_simulationTime,
                                                               materialHandler->d_materials,
                                                               particleHandler->d_particles,
                                                               d_blockShared, d_blockCount, h_searchRadius,
                                                               numParticlesLocal);

        simulationTimeHandler->globalizeTimeStep(Execution::device);
        simulationTimeHandler->copy(To::host);
        Logger(INFO) << "h_dt = " << *simulationTimeHandler->h_dt << "  | h_dt_max = "
                     << *simulationTimeHandler->h_dt_max;;
        Logger(INFO) << "h_startTime = " << *simulationTimeHandler->h_startTime;
        Logger(INFO) << "h_subEndTime = " << *simulationTimeHandler->h_subEndTime;
        Logger(INFO) << "h_endTime = " << *simulationTimeHandler->h_endTime;
        Logger(INFO) << "h_currentTime = " << *simulationTimeHandler->h_currentTime;
        //Logger(INFO) << "h_dt_max = " << *simulationTimeHandler->h_dt_max;
        // ------------------------------------------------------------------------------------------------------------
        Logger(INFO) << "PREDICTOR!";
        time += PredictorCorrectorEulerNS::Kernel::Launch::predictor(particleHandler->d_particles,
                                                                     integratedParticles[0].d_integratedParticles,
                                                                     *simulationTimeHandler->h_dt, //(real) simulationParameters.timestep,
                                                                     numParticlesLocal);

        Logger(INFO) << "setPointer()...";
        particleHandler->setPointer(&integratedParticles[0]);

        timerRhs.reset();
        // -------------------------------------------------------------------------------------------------------------
        time += rhs(step, false, false);
        // -------------------------------------------------------------------------------------------------------------
        timeElapsed = timerRhs.elapsed();
        Logger(TIME) << "rhsElapsed: " << timeElapsed;
        //Logger(TIME) << "rhs: " << time << " ms";
        profiler.value2file(ProfilerIds::Time::rhs, time);
        profiler.value2file(ProfilerIds::Time::rhsElapsed, timeElapsed);

        Logger(INFO) << "resetPointer()...";
        particleHandler->resetPointer();

        Logger(INFO) << "CORRECTOR!";
        time += PredictorCorrectorEulerNS::Kernel::Launch::corrector(particleHandler->d_particles,
                                                                     integratedParticles[0].d_integratedParticles,
                                                                     *simulationTimeHandler->h_dt, //(real) simulationParameters.timestep,
                                                                     numParticlesLocal);


        *simulationTimeHandler->h_currentTime += *simulationTimeHandler->h_dt;
        simulationTimeHandler->copy(To::device);

        Logger(TRACE) << "finished sub step - simulation time: " << *simulationTimeHandler->h_currentTime
                      << " (STEP: " << step << " | subStep: " << subStep
                      << " | time = " << *simulationTimeHandler->h_currentTime << "/"
                      << *simulationTimeHandler->h_subEndTime << "/"
                      << *simulationTimeHandler->h_endTime << ")";

        //H5Profiler &profiler = H5Profiler::getInstance("log/performance.h5");


        subDomainKeyTreeHandler->copy(To::host, true, false);
        profiler.vector2file(ProfilerIds::ranges, subDomainKeyTreeHandler->h_range);

        boost::mpi::communicator comm;
        sumParticles = numParticlesLocal;
        all_reduce(comm, boost::mpi::inplace_t<integer*>(&sumParticles), 1, std::plus<integer>());

        profiler.value2file(ProfilerIds::numParticles, sumParticles);
        profiler.value2file(ProfilerIds::numParticlesLocal, numParticlesLocal);

    }

    timeElapsed = timer.elapsed();
    Logger(TIME) << "integration step elapsed: " << timeElapsed << " ms";

}
