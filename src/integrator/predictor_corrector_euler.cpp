#include "../../include/integrator/predictor_corrector_euler.h"

PredictorCorrectorEuler::PredictorCorrectorEuler(SimulationParameters simulationParameters) : Miluphpc(simulationParameters) {
    printf("PredictorCorrectorEuler()\n");
    integratedParticles = new IntegratedParticleHandler(numParticles, numNodes);


    cudaGetDevice(device);
    cudaGetDeviceProperties(prop, *device);
    cuda::malloc(d_blockCount, 1);
    cuda::set(d_blockCount, 0, 1);

    cuda::malloc(d_block_forces, prop->multiProcessorCount);
    cuda::malloc(d_block_courant, prop->multiProcessorCount);
    cuda::malloc(d_block_artVisc, prop->multiProcessorCount);
    cuda::malloc(d_block_e, prop->multiProcessorCount);
    cuda::malloc(d_block_rho, prop->multiProcessorCount);

    cuda::malloc(d_blockShared, 1);

    PredictorCorrectorEulerNS::BlockSharedNS::set(d_blockShared, d_block_forces, d_block_courant,
                                                  d_block_courant);
    PredictorCorrectorEulerNS::BlockSharedNS::setE(d_blockShared, d_block_e);
    PredictorCorrectorEulerNS::BlockSharedNS::setRho(d_blockShared, d_block_rho);

}

PredictorCorrectorEuler::~PredictorCorrectorEuler() {
    delete [] integratedParticles;
    printf("~PredictorCorrectorEuler()\n");

    cuda::free(d_block_forces);
    cuda::free(d_block_courant);
    cuda::free(d_block_artVisc);
    cuda::free(d_block_e);
    cuda::free(d_block_rho);

    cuda::free(d_blockShared);
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

