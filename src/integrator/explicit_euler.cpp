#include "../../include/integrator/explicit_euler.h"

/*ExplicitEuler::ExplicitEuler(SimulationParameters simulationParameters, integer numParticles,
                             integer numNodes) : Miluphpc(simulationParameters, numParticles, numNodes) {
    //integratedParticles = new IntegratedParticles[1];
    printf("ExplicitEuler()\n");
}*/

ExplicitEuler::ExplicitEuler(SimulationParameters simulationParameters) : Miluphpc(simulationParameters) {
    printf("ExplicitEuler()\n");
    // just testing // TODO: remove!!!
    //integratedParticles = new IntegratedParticleHandler(numParticles, numNodes);
    //particleHandler->setPointer(&integratedParticles[0]);
    // end: testing
}

ExplicitEuler::~ExplicitEuler() {
    printf("~ExplicitEuler()\n");
}

void ExplicitEuler::integrate(int step) {

    Timer timer;
    real time = 0.;

    real time_elapsed;

    Logger(INFO) << "ExplicitEuler::integrate()... currentTime: " << *simulationTimeHandler->h_currentTime
        << " | endTime: " << *simulationTimeHandler->h_endTime;

    while (*simulationTimeHandler->h_currentTime < *simulationTimeHandler->h_endTime) {

        Logger(INFO) << "ExplicitEuler::integrate while...";

        //removeParticles();

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
        //real time;
        time = rhs(step, true, true);
        time_elapsed = timer.elapsed();
        Logger(TIME) << "rhs: " << time << " ms";
        Logger(TIME) << "rhsElapsed: " << time_elapsed << " ms";

        ExplicitEulerNS::Kernel::Launch::update(particleHandler->d_particles, numParticlesLocal,
                                                *simulationTimeHandler->h_dt); //(real) simulationParameters.timestep);

        //Logger(INFO) << "timestep: " << (real) simulationParameters.timestep;

        *simulationTimeHandler->h_currentTime += *simulationTimeHandler->h_dt;
        simulationTimeHandler->copy(To::device);

        /* // testing
        real x_diam = std::abs(*treeHandler->h_maxX) + std::abs(*treeHandler->h_minX);
    #if DIM > 1
        real y_diam = std::abs(*treeHandler->h_maxY) + std::abs(*treeHandler->h_minY);
    #if DIM == 3
        real z_diam = std::abs(*treeHandler->h_maxZ) + std::abs(*treeHandler->h_minZ);
    #endif
    #endif

        real maxDiam;
    #if DIM == 1
        maxDiam = x_diam;
    #elif DIM == 2
        maxDiam = std::max(x_diam, y_diam);
    #else
        real tempDiam = std::max(x_diam, y_diam);
        maxDiam = std::max(tempDiam, z_diam);
    #endif

        int bins = 1e3;
        int *h_particlesWithin = new int[bins];
        int *d_particlesWithin;
        real *d_radial_rho;
        real *h_radial_rho;
        cuda::malloc(d_particlesWithin, bins);
        cuda::malloc(d_radial_rho, bins);
        cuda::set(d_particlesWithin, 0, bins);
        cuda::set(d_radial_rho, (real)0, bins);
        real deltaRadial = maxDiam/bins;

        Processing::Kernel::Launch::particlesWithinRadii(particleHandler->d_particles, d_particlesWithin, deltaRadial, numParticlesLocal);
        boost::mpi::communicator comm;
        all_reduce(comm, boost::mpi::inplace_t<int*>(d_particlesWithin), bins, std::plus<int>());
        cuda::copy(h_particlesWithin, d_particlesWithin, bins, To::host);

        int sum = 0;
        for (int i=0; i<bins; i++) {
            if (i %100 == 0) {
                Logger(INFO) << "particles within " << i << " : " << h_particlesWithin[i];
            }
            sum += h_particlesWithin[i];
        }
        Logger(INFO) << "sum = " << sum;

        Processing::Kernel::Launch::cartesianToRadial(particleHandler->d_particles, d_particlesWithin, particleHandler->d_rho, d_radial_rho,
                                                      deltaRadial, numParticlesLocal);
        all_reduce(comm, boost::mpi::inplace_t<real*>(d_radial_rho), bins, std::plus<real>());
        cuda::copy(h_radial_rho, d_radial_rho, bins, To::host);

        Logger(INFO) << "Finished calculation";

        if (subDomainKeyTreeHandler->h_rank == 0) {
            HighFive::File h5file("log/radial.h5",
                                  HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate); //,
                                  //HighFive::MPIOFileDriver(MPI_COMM_WORLD, MPI_INFO_NULL));

            HighFive::DataSet h5_rho = h5file.createDataSet<real>("/rho", HighFive::DataSpace(bins));
            std::vector <real> vec_rho;

            for (int i = 0; i < bins; i++) {
                vec_rho.push_back(h_radial_rho[i]);
            }

            h5_rho.write(vec_rho);
        }


        cuda::free(d_particlesWithin);
        cuda::free(d_radial_rho);
        delete [] h_particlesWithin; */
        // end: testing

#if INTEGRATE_SML
        cuda::set(particleHandler->d_dsmldt, (real)0, numParticles);
#endif

        Logger(INFO) << "simulation time: " << *simulationTimeHandler->h_currentTime << "( STEP: " << step << ", endTime = " << *simulationTimeHandler->h_endTime << ")";
    }

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