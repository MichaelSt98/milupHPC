#include "../../include/integrator/predictor_corrector_euler.h"

PredictorCorrectorEuler::PredictorCorrectorEuler(SimulationParameters simulationParameters) : Miluphpc(simulationParameters) {
    printf("PredictorCorrectorEuler()\n");
    integratedParticles = new IntegratedParticleHandler(numParticles, numNodes);
}

PredictorCorrectorEuler::~PredictorCorrectorEuler() {
    delete [] integratedParticles;
    printf("~PredictorCorrectorEuler()\n");
}

void PredictorCorrectorEuler::integrate(int step) {

    printf("PredictorCorrector::integrate()\n");

    Timer timer;
    real time = 0.;

    time += rhs(step, true);
    Logger(INFO) << "PREDICTOR!";
    time += PredictorCorrectorEulerNS::Kernel::Launch::predictor(particleHandler->d_particles,
                                                                 integratedParticles[0].d_integratedParticles,
                                                                 (real)simulationParameters.timestep, numParticlesLocal);

    particleHandler->setPointer(&integratedParticles[0]);

    time += rhs(step, false);
    Logger(INFO) << "CORRECTOR!";
    time += PredictorCorrectorEulerNS::Kernel::Launch::corrector(particleHandler->d_particles,
                                                                 integratedParticles[0].d_integratedParticles,
                                                                 (real)simulationParameters.timestep, numParticlesLocal);

     particleHandler->resetPointer();

    Logger(TIME) << "rhs: " << time << " ms";

    real time_elapsed = timer.elapsed();
    Logger(TIME) << "rhs elapsed: " << time_elapsed  << " ms";

    //Gravity::Kernel::Launch::update(particleHandler->d_particles, numParticlesLocal,
    //                                (real)simulationParameters.timestep, (real)simulationParameters.dampening);

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

