#include "../include/miluphpc.h"

Miluphpc::Miluphpc(SimulationParameters simulationParameters) {

    this->simulationParameters = simulationParameters;
    subStep = 0;

    // get number of particles from provided input file
    numParticlesFromFile(simulationParameters.inputFile);
    // estimate amount of memory for simulation // TODO: generalize/optimize numNodes
    numNodes = 2 * numParticles + 20000;

    boost::mpi::communicator comm;
    sumParticles = numParticlesLocal;
    all_reduce(comm, boost::mpi::inplace_t<integer*>(&sumParticles), 1, std::plus<integer>());

    Logger(DEBUG) << "Initialized: numParticlesLocal: " << numParticlesLocal;
    Logger(DEBUG) << "Initialized: numParticles:      " << numParticles;
    Logger(DEBUG) << "Initialized: numNodes:          " << numNodes;
    Logger(DEBUG) << "Initialized: sumParticles:      " << sumParticles;

    // memory allocations (via instantiation of handlers)
#if TARGET_GPU
    cuda::malloc(d_mutex, 1);
#endif
    //helperHandler = new HelperHandler(numNodes);
    //buffer = new HelperHandler(numNodes);
    subDomainKeyTreeHandler = new SubDomainKeyTreeHandler();
    Logger(DEBUG) << "Initialized: numProcesses:      " << subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses;
    buffer = new HelperHandler(subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses, numParticlesLocal, numParticles,
                               sumParticles, numNodes);
    particleHandler = new ParticleHandler(numParticles, numNodes);
    treeHandler = new TreeHandler(numParticles, numNodes);
    domainListHandler = new DomainListHandler(simulationParameters.domainListSize);
    lowestDomainListHandler = new DomainListHandler(simulationParameters.domainListSize);
#if SPH_SIM
    materialHandler = new MaterialHandler(simulationParameters.materialConfigFile.c_str()); //"config/material.cfg"
#if DEBUGGING
    for (int i=0; i<materialHandler->numMaterials; i++) {
        materialHandler->h_materials[i].info();
    }
#endif
    materialHandler->copy(To::device);
#endif
    simulationTimeHandler = new SimulationTimeHandler(simulationParameters.timeStep,
                                                      simulationParameters.timeEnd,
                                                      simulationParameters.maxTimeStep);
#if TARGET_GPU
    simulationTimeHandler->copy(To::device);
#endif
    if (subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses == 1) {
        curveType = Curve::lebesgue;
    }
    else {
        //curveType = Curve::lebesgue; //curveType = Curve::hilbert;
        if (simulationParameters.sfcSelection >= 0 && simulationParameters.sfcSelection <= 2) {
            curveType = Curve::Type(simulationParameters.sfcSelection);
            if (simulationParameters.sfcSelection == 0) { Logger(DEBUG) << "Selected sfc: Lebesgue"; }
            else if (simulationParameters.sfcSelection == 1) { Logger(DEBUG) << "Selected sfc: Hilbert"; }
            else { Logger(DEBUG) << "Selected sfc: not valid!"; }
        }
        else {
            curveType = Curve::Type(0); // selecting lebesgue sfc as default
            Logger(DEBUG) << "Selected sfc: Lebesgue (Default)";
        }
    }

    // TODO: buffer arrays/pointers handling/optimization (and corresponding freeing)
    //cuda::malloc(d_particles2SendIndices, numNodes); // numParticles
    //cuda::malloc(d_pseudoParticles2SendIndices, numNodes);
    //cuda::malloc(d_pseudoParticles2SendLevels, numNodes);
    //cuda::malloc(d_pseudoParticles2ReceiveLevels, numNodes);
    //cuda::malloc(d_particles2SendCount, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses);
    //cuda::malloc(d_pseudoParticles2SendCount, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses);
    //cuda::malloc(d_particles2removeBuffer, numParticles);
    //cuda::malloc(d_particles2removeVal, 1);
    //cuda::malloc(d_idIntegerBuffer, numParticles);
    //cuda::malloc(d_idIntegerCopyBuffer, numParticles);

#if SPH_SIM
    try {
        kernelHandler = SPH::KernelHandler(Smoothing::Kernel(simulationParameters.smoothingKernelSelection));
    }
    catch(...) {
        Logger(WARN) << "Kernel not available! Selecting cubic spline [1] as default!";
        kernelHandler = SPH::KernelHandler(Smoothing::Kernel::cubic_spline);
    }
#else
    // TODO: instance should not be needed at all if SPH_SIM 0
#if TARGET_GPU
    kernelHandler = SPH::KernelHandler(Smoothing::Kernel(0));
#endif
#endif

    // read particle data, load balancing, ...
    prepareSimulation();

    // check memory allocations
    getMemoryInfo();

}

Miluphpc::~Miluphpc() {

    //delete helperHandler;
    delete buffer;
    delete particleHandler;
    delete subDomainKeyTreeHandler;
    delete treeHandler;
#if SPH_SIM
    delete materialHandler;
#endif
    delete simulationTimeHandler;

#if TARGET_GPU
    cuda::free(d_mutex);
#endif // TARGET_GPU
    //cuda::free(d_particles2SendIndices);
    //cuda::free(d_pseudoParticles2SendIndices);
    //cuda::free(d_pseudoParticles2SendLevels);
    //cuda::free(d_pseudoParticles2ReceiveLevels);
    //cuda::free(d_particles2SendCount);
    //cuda::free(d_pseudoParticles2SendCount);
    //cuda::free(d_particles2removeBuffer);
    //cuda::free(d_particles2removeVal);
    //cuda::free(d_idIntegerBuffer);
    //cuda::free(d_idIntegerCopyBuffer);

}

void Miluphpc::numParticlesFromFile(const std::string& filename) {

    boost::mpi::communicator comm;
    HighFive::File file(filename.c_str(), HighFive::File::ReadOnly);
    std::vector<real> m;
    std::vector<std::vector<real>> x;

    // read datasets from file
    HighFive::DataSet mass = file.getDataSet("/m");
    HighFive::DataSet pos = file.getDataSet("/x");

    mass.read(m);
    pos.read(x);

    numParticles = (int)(m.size() * simulationParameters.particleMemoryContingent);

    integer ppp = m.size()/comm.size();
    integer ppp_remnant = m.size() % comm.size();

#if DEBUGGING
    Logger(INFO) << "ppp = " << ppp;
    Logger(INFO) << "ppp remnant = " << ppp_remnant;
#endif

    if (ppp_remnant == 0) {
        numParticlesLocal = ppp;
    }
    else {
        if (comm.rank() < (comm.size()-1)) {
            numParticlesLocal = ppp;
        }
        else {
            numParticlesLocal = ppp + ppp_remnant;
        }

    }
}

void Miluphpc::distributionFromFile(const std::string& filename) {

    HighFive::File file(filename.c_str(), HighFive::File::ReadOnly);

    // containers to be filled
    std::vector<real> m, u;
    std::vector<std::vector<real>> x, v;
    std::vector<integer> materialId;

    // read datasets from file
    HighFive::DataSet mass = file.getDataSet("/m");
    HighFive::DataSet pos = file.getDataSet("/x");
    HighFive::DataSet vel = file.getDataSet("/v");

#if SPH_SIM
    HighFive::DataSet matId = file.getDataSet("/materialId");
#if INTEGRATE_ENERGY
    HighFive::DataSet h5_u = file.getDataSet("/u");
#endif
#endif

    // read data
    mass.read(m);
    pos.read(x);
    vel.read(v);
#if SPH_SIM
    matId.read(materialId);
    //TODO: when do I want to read in the (internal) energy?
#if INTEGRATE_ENERGY
    h5_u.read(u);
#endif
#endif

    // TODO: read sml if xy?

    integer ppp = m.size()/subDomainKeyTreeHandler->h_numProcesses;
    integer ppp_remnant = m.size() % subDomainKeyTreeHandler->h_numProcesses;

    int startIndex = subDomainKeyTreeHandler->h_subDomainKeyTree->rank * ppp;
    int endIndex = (subDomainKeyTreeHandler->h_rank + 1) * ppp;
    if (subDomainKeyTreeHandler->h_rank == (subDomainKeyTreeHandler->h_numProcesses - 1)) {
        endIndex += ppp_remnant;
    }

    for (int j = startIndex; j < endIndex; j++) {
        int i = j - subDomainKeyTreeHandler->h_rank * ppp;

        particleHandler->h_particles->uid[i] = j;
        particleHandler->h_particles->mass[i] = m[j];

        particleHandler->h_particles->x[i] = x[j][0];
        particleHandler->h_particles->vx[i] = v[j][0];
#if DIM > 1
        particleHandler->h_particles->y[i] = x[j][1];
        particleHandler->h_particles->vy[i] = v[j][1];
#if DIM == 3
        particleHandler->h_particles->z[i] = x[j][2];
        particleHandler->h_particles->vz[i] = v[j][2];
#endif
#endif


#if SPH_SIM
#if INTEGRATE_ENERGY
        particleHandler->h_particles->e[i] = u[j];
#endif
        particleHandler->h_particles->materialId[i] = materialId[j];
        //particleHandler->h_particles->sml[i] = simulationParameters.sml;
        particleHandler->h_particles->sml[i] = materialHandler->h_materials[materialId[j]].sml;
#endif

    }

    //for (int particleIndex=0; particleIndex < 10; particleIndex++) {
    //    Logger(TRACE) << "x = (" << particleHandler->h_particles->x[particleIndex] << ", " << particleHandler->h_particles->y[particleIndex] << ", " << particleHandler->h_particles->z[particleIndex] << ")";
    //}
}

//TODO: block/warp/stack size for computeBoundingBox and computeForces
void Miluphpc::prepareSimulation() {

    Logger(DEBUG) << "Preparing simulation ...";

    Logger(DEBUG) << "Initialize/Read particle distribution ...";
    distributionFromFile(simulationParameters.inputFile);

#if SPH_SIM
    SPH::Kernel::Launch::initializeSoundSpeed(particleHandler->d_particles, materialHandler->d_materials, numParticlesLocal);
#endif

#if TARGET_GPU
    particleHandler->copyDistribution(To::device, true, true);
#endif

#if SPH_SIM
    //TODO: problem: (e.g.) cs should not be copied....
    //particleHandler->copySPH(To::device);
    //cuda::copy(particleHandler->h_rho, particleHandler->d_rho, numParticles, To::device);
    //cuda::copy(particleHandler->h_p, particleHandler->d_p, numParticles, To::device);
#if TARGET_GPU
    cuda::copy(particleHandler->h_e, particleHandler->d_e, numParticles, To::device);
    cuda::copy(particleHandler->h_sml, particleHandler->d_sml, numParticles, To::device);
#endif
    //cuda::copy(particleHandler->h_noi, particleHandler->d_noi, numParticles, To::device);
    //cuda::copy(particleHandler->h_cs, particleHandler->d_cs, numParticles, To::device);
#endif

#if TARGET_GPU
    if (simulationParameters.removeParticles) {
        removeParticles();
    }
#endif

    Logger(DEBUG) << "Compute bounding box ...";
#if TARGET_GPU
    TreeNS::Kernel::Launch::computeBoundingBox(treeHandler->d_tree, particleHandler->d_particles, d_mutex,
                                               numParticlesLocal, 256, false);

#if DEBUGGING
#if DIM == 3
    treeHandler->copy(To::host);
    Logger(DEBUG) << "x: " << std::abs(*treeHandler->h_maxX) << ", " << std::abs(*treeHandler->h_minX);
    Logger(DEBUG) << "y: " << std::abs(*treeHandler->h_maxY) << ", " << std::abs(*treeHandler->h_minY);
    Logger(DEBUG) << "z: " << std::abs(*treeHandler->h_maxZ) << ", " << std::abs(*treeHandler->h_minZ);
#endif
#endif

    treeHandler->globalizeBoundingBox(Execution::device);
    treeHandler->copy(To::host);

#endif // TARGET_GPU

    if (simulationParameters.loadBalancing) {
        fixedLoadBalancing();
        dynamicLoadBalancing();
    }
    else {
        fixedLoadBalancing();
    }

    subDomainKeyTreeHandler->copy(To::device);

    boost::mpi::communicator comm;
    sumParticles = numParticlesLocal;
    all_reduce(comm, boost::mpi::inplace_t<integer*>(&sumParticles), 1, std::plus<integer>());

}

real Miluphpc::rhs(int step, bool selfGravity, bool assignParticlesToProcess) {

    // TESTING
    //Logger(INFO) << "reduction: max:";
    //HelperNS::reduceAndGlobalize(particleHandler->d_sml, helperHandler->d_realVal, numParticlesLocal, Reduction::max);
    //Logger(INFO) << "reduction: min:";
    //HelperNS::reduceAndGlobalize(particleHandler->d_sml, helperHandler->d_realVal, numParticlesLocal, Reduction::min);
    //Logger(INFO) << "reduction: sum:";
    //HelperNS::reduceAndGlobalize(particleHandler->d_sml, helperHandler->d_realVal, numParticlesLocal, Reduction::sum);
    // end: TESTING

    Logger(INFO) << "Miluphpc::rhs()";

    getMemoryInfo();

    Timer timer;
    real time = 0.;
    real elapsed;
    real *profilerTime = &elapsed; //&time;
    real totalTime = 0;


    Logger(INFO) << "rhs::reset()";
    timer.reset();
    // -----------------------------------------------------------------------------------------------------------------
    time = reset();
    // -----------------------------------------------------------------------------------------------------------------
    elapsed = timer.elapsed();
    totalTime += time;
    Logger(TIME) << "rhs::reset(): " << elapsed << " ms"; //time << " ms";
    profiler.value2file(ProfilerIds::Time::reset, *profilerTime);


    //Logger(INFO) << "checking for nans before bounding box...";
    //ParticlesNS::Kernel::Launch::check4nans(particleHandler->d_particles, numParticlesLocal);

    Logger(INFO) << "rhs::boundingBox()";
    timer.reset();
    // -----------------------------------------------------------------------------------------------------------------
    time = boundingBox();
    // -----------------------------------------------------------------------------------------------------------------
    elapsed = timer.elapsed();
    totalTime += time;
    Logger(TIME) << "rhs::boundingBox(): " << time << " ms";
    profiler.value2file(ProfilerIds::Time::boundingBox, *profilerTime);


    treeHandler->h_tree->buildTree(particleHandler->h_particles, numParticlesLocal, numParticles);



    //for (int i=0; i<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses+1; ++i) {
    //    Logger(TRACE) << "range[" << i << "] = " << subDomainKeyTreeHandler->h_subDomainKeyTree->range[i];
    //}

    if (subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses > 1) {
        subDomainKeyTreeHandler->h_subDomainKeyTree->buildDomainTree(treeHandler->h_tree, particleHandler->h_particles,
                                                                     domainListHandler->h_domainList, numParticles,
                                                                     curveType);
    }

    pseudoParticles();


    real massCheck = 0.;
    for (int i=0; i<POW_DIM; ++i) {
        //Logger(TRACE) << "child[" << i << "] = " << treeHandler->h_tree->child[i] << "  mass: " << particleHandler->h_particles->mass[treeHandler->h_tree->child[i]];
        massCheck += particleHandler->h_particles->mass[treeHandler->h_tree->child[i]];
    }
    Logger(TRACE) << "massCheck: " << massCheck;


    //Logger(TRACE) << "domainList: " << *domainListHandler->h_domainList->domainListIndex << " | lowestDomainList: " << *lowestDomainListHandler->h_domainList->domainListIndex;

    //for (int i=0; i<*lowestDomainListHandler->h_domainList->domainListIndex; i++) {
    //    Logger(TRACE) << "lowestDomainList[" << i << "] = " << lowestDomainListHandler->h_domainList->domainListKeys[i]
    //                  << "  level = " << lowestDomainListHandler->h_domainList->domainListLevels[i] << "  mass = " << particleHandler->h_particles->mass[lowestDomainListHandler->h_domainList->domainListIndices[i]];
    //}

    //for (int i=0; i<*domainListHandler->h_domainList->domainListIndex; i++) {
    //    Logger(TRACE) << "domainList[" << i << "] = " << domainListHandler->h_domainList->domainListKeys[i]
    //    << "  mass = " << particleHandler->h_particles->mass[domainListHandler->h_domainList->domainListIndices[i]];
    //}


    /*

#if DEBUGGING
    Logger(INFO) << "checking for nans before assigning particles...";
    ParticlesNS::Kernel::Launch::check4nans(particleHandler->d_particles, numParticlesLocal);
#endif

//#if DEBUGGING
    Logger(INFO) << "before: numParticlesLocal: " << numParticlesLocal;
    Logger(INFO) << "before: numParticles:      " << numParticles;
    Logger(INFO) << "before: numNodes:          " << numNodes;
//#endif

    if (assignParticlesToProcess) {
        timer.reset();
        if (subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses > 1) {
            Logger(INFO) << "rhs::assignParticles()";
            // ---------------------------------------------------------------------------------------------------------
            time = assignParticles();
            // ---------------------------------------------------------------------------------------------------------
        }
        elapsed = timer.elapsed();
        totalTime += time;
        Logger(TIME) << "rhs::assignParticles(): " << elapsed << " ms"; //time << " ms";
        profiler.value2file(ProfilerIds::Time::assignParticles, *profilerTime);
    }
    //else {
    //    profiler.value2file(ProfilerIds::Time::assignParticles, 0.);
    //}

    //Logger(INFO) << "checking for nans after assigning particles...";
    //ParticlesNS::Kernel::Launch::check4nans(particleHandler->d_particles, numParticlesLocal);

#if DEBUGGING
    Logger(INFO) << "after: numParticlesLocal: " << numParticlesLocal;
    Logger(INFO) << "after: numParticles:      " << numParticles;
    Logger(INFO) << "after: numNodes:          " << numNodes;
#endif

    boost::mpi::communicator comm;
    sumParticles = numParticlesLocal;
    all_reduce(comm, boost::mpi::inplace_t<integer*>(&sumParticles), 1, std::plus<integer>());
    Logger(DEBUG) << "numParticlesLocal = " << numParticlesLocal;

    subDomainKeyTreeHandler->copy(To::host, true, false);
    for (int i=0; i<=subDomainKeyTreeHandler->h_numProcesses; i++) {
        Logger(DEBUG) << "rangeValues[" << i << "] = " << subDomainKeyTreeHandler->h_subDomainKeyTree->range[i];
    }

    Logger(INFO) << "rhs::tree()";
    timer.reset();
    // -----------------------------------------------------------------------------------------------------------------
    time = tree();
    // -----------------------------------------------------------------------------------------------------------------
    elapsed = timer.elapsed();
    totalTime += time;
    Logger(TIME) << "rhs::tree(): " << elapsed << " ms"; //time << " ms";
    profiler.value2file(ProfilerIds::Time::tree, *profilerTime);

#if GRAVITY_SIM
    if (selfGravity) {
        Logger(INFO) << "rhs::pseudoParticles()";
        timer.reset();
        // -------------------------------------------------------------------------------------------------------------
        time = pseudoParticles();
        // -------------------------------------------------------------------------------------------------------------
        elapsed = timer.elapsed();
        totalTime += time;
        Logger(TIME) << "rhs::pseudoParticles(): " << elapsed << " ms"; //time << " ms";
        profiler.value2file(ProfilerIds::Time::pseudoParticle, *profilerTime);
    }
    else {
#if TARGET_GPU
        // -------------------------------------------------------------------------------------------------------------
        DomainListNS::Kernel::Launch::lowestDomainList(subDomainKeyTreeHandler->d_subDomainKeyTree, treeHandler->d_tree,
                                                       particleHandler->d_particles, domainListHandler->d_domainList,
                                                       lowestDomainListHandler->d_domainList, numParticles, numNodes);
        // -------------------------------------------------------------------------------------------------------------
#endif // TARGET_GPU
    }
#else
    // -----------------------------------------------------------------------------------------------------------------
    DomainListNS::Kernel::Launch::lowestDomainList(subDomainKeyTreeHandler->d_subDomainKeyTree, treeHandler->d_tree,
                                                   particleHandler->d_particles, domainListHandler->d_domainList,
                                                   lowestDomainListHandler->d_domainList, numParticles, numNodes);
    // -----------------------------------------------------------------------------------------------------------------
#endif

#if GRAVITY_SIM
    if (selfGravity) {
        timer.reset();
        Logger(INFO) << "rhs::gravity()";
        // -------------------------------------------------------------------------------------------------------------
        time = gravity();
        // -------------------------------------------------------------------------------------------------------------
        elapsed = timer.elapsed();
        totalTime += time;
        Logger(TIME) << "rhs::gravity(): " << time << " ms";
        profiler.value2file(ProfilerIds::Time::gravity, *profilerTime);
    }
    //else {
    //    int *dummy = new int[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    //    for (int i_dummy=0; i_dummy<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; ++i_dummy) {
    //        dummy[i_dummy] = 0;
    //    }
    //    profiler.vector2file(ProfilerIds::SendLengths::gravityParticles, dummy);
    //    profiler.vector2file(ProfilerIds::SendLengths::gravityPseudoParticles, dummy);
    //    profiler.vector2file(ProfilerIds::ReceiveLengths::gravityParticles, dummy);
    //    profiler.vector2file(ProfilerIds::ReceiveLengths::gravityPseudoParticles, dummy);
    //    delete [] dummy;
    //
    //    profiler.value2file(ProfilerIds::Time::Gravity::compTheta, 0.);
    //    profiler.value2file(ProfilerIds::Time::Gravity::symbolicForce, 0.);
    //    profiler.value2file(ProfilerIds::Time::Gravity::sending, 0.);
    //    profiler.value2file(ProfilerIds::Time::Gravity::insertReceivedPseudoParticles, 0.);
    //    profiler.value2file(ProfilerIds::Time::Gravity::insertReceivedParticles, 0.);
    //    profiler.value2file(ProfilerIds::Time::Gravity::force, 0.);
    //    profiler.value2file(ProfilerIds::Time::Gravity::repairTree, 0.);
    //    profiler.value2file(ProfilerIds::Time::gravity, 0.);
    //}
#endif

#if SPH_SIM
    Logger(INFO) << "rhs: sph()";
    timer.reset();
    // -----------------------------------------------------------------------------------------------------------------
    time = sph();
    // -----------------------------------------------------------------------------------------------------------------
    elapsed = timer.elapsed();
    Logger(TIME) << "rhs::sph(): " << elapsed << " ms"; //time << " ms";
    profiler.value2file(ProfilerIds::Time::sph, *profilerTime);
    totalTime += time;
#endif
    */
    return totalTime;
}

void Miluphpc::afterIntegrationStep() {
    // things which need/can be done after (full) integration step

#if TARGET_GPU
    // angular momentum calculation
    if (simulationParameters.calculateAngularMomentum) {
        angularMomentum();
    }
    // energy calculation
    if (simulationParameters.calculateEnergy) {
        energy();
    }
#else
    // TODO: CPU afterIntegrationStep()
#endif
}

#if TARGET_GPU
real Miluphpc::angularMomentum() {
    real time;

    boost::mpi::communicator comm;

    const unsigned int blockSizeReduction = 256;
    real *d_outputData;
    //cuda::malloc(d_outputData, blockSizeReduction * DIM);
    d_outputData = buffer->d_realBuffer;
    cuda::set(d_outputData, (real)0., blockSizeReduction * DIM);
    time = Physics::Kernel::Launch::calculateAngularMomentumBlockwise<blockSizeReduction>(particleHandler->d_particles,
                                                                                          d_outputData,
                                                                                          numParticlesLocal);
    real *d_intermediateAngularMomentum;
    //cuda::malloc(d_intermediateAngularMomentum, DIM);
    d_intermediateAngularMomentum = buffer->d_realBuffer1; //&d_outputData[blockSizeReduction * DIM];
    cuda::set(d_intermediateAngularMomentum, (real)0., DIM);
    time += Physics::Kernel::Launch::sumAngularMomentum<blockSizeReduction>(d_outputData,
                                                                            d_intermediateAngularMomentum);

    real *h_intermediateResult = new real[DIM];
    //cuda::copy(h_intermediateResult, d_intermediateAngularMomentum, DIM, To::host);
    //Logger(INFO) << "angular momentum before MPI: (" << h_intermediateResult[0] << ", "
    // << h_intermediateResult[1] << ", " << h_intermediateResult[2] << ")";

    all_reduce(comm, boost::mpi::inplace_t<real*>(d_intermediateAngularMomentum), DIM, std::plus<real>());
    cuda::copy(h_intermediateResult, d_intermediateAngularMomentum, DIM, To::host);

    //Logger(INFO) << "angular momentum: (" << h_intermediateResult[0] << ", " << h_intermediateResult[1] << ", " << h_intermediateResult[2] << ")";

    real angularMomentum;
#if DIM == 1
    angularMomentum = abs(h_intermediateResult[0]);
#elif DIM == 2
    angularMomentum = sqrt(h_intermediateResult[0] * h_intermediateResult[0] + h_intermediateResult[1] *
            h_intermediateResult[1]);
#else
    angularMomentum = sqrt(h_intermediateResult[0] * h_intermediateResult[0] +
                            h_intermediateResult[1] * h_intermediateResult[1] +
                            h_intermediateResult[2] * h_intermediateResult[2]);
#endif

    totalAngularMomentum = angularMomentum;
    Logger(DEBUG) << "angular momentum: " << totalAngularMomentum;

    delete [] h_intermediateResult;
    //cuda::free(d_outputData);
    //cuda::free(d_intermediateAngularMomentum);

    Logger(TIME) << "angular momentum: " << time << " ms";

    return time;
}
#endif // TARGET_GPU

#if TARGET_GPU
real Miluphpc::energy() {

    real time = 0;

    time = Physics::Kernel::Launch::kineticEnergy(particleHandler->d_particles, numParticlesLocal);

    boost::mpi::communicator comm;

    const unsigned int blockSizeReduction = 256;
    real *d_outputData;
    //cuda::malloc(d_outputData, blockSizeReduction);
    d_outputData = buffer->d_realBuffer;
    cuda::set(d_outputData, (real)0., blockSizeReduction);
    time += CudaUtils::Kernel::Launch::reduceBlockwise<real, blockSizeReduction>(particleHandler->d_u, d_outputData,
                                                                                 numParticlesLocal);
    real *d_intermediateResult;
    //cuda::malloc(d_intermediateResult, 1);
    d_intermediateResult = buffer->d_realVal;
    cuda::set(d_intermediateResult, (real)0., 1);
    time += CudaUtils::Kernel::Launch::blockReduction<real, blockSizeReduction>(d_outputData, d_intermediateResult);

    real h_intermediateResult;
    //cuda::copy(&h_intermediateResult, d_intermediateResult, 1, To::host);
    //Logger(INFO) << "local energy: " << h_intermediateResult;

    all_reduce(comm, boost::mpi::inplace_t<real*>(d_intermediateResult), 1, std::plus<real>());

    cuda::copy(&h_intermediateResult, d_intermediateResult, 1, To::host);
    totalEnergy = h_intermediateResult;

    //cuda::free(d_outputData);
    //cuda::free(d_intermediateResult);

    Logger(DEBUG) << "energy: " << totalEnergy;
    Logger(TIME) << "energy: " << time << " ms";

    //cuda::copy(particleHandler->h_u, particleHandler->d_u, numParticlesLocal, To::host);
    //cuda::copy(particleHandler->h_mass, particleHandler->d_mass, numParticlesLocal, To::host);
    //for (int i=0; i<numParticlesLocal; i++) {
    //    if (i % 100 == 0 || particleHandler->h_mass[i] > 0.0001) {
    //        Logger(INFO) << "u[" << i << "] = " << particleHandler->h_u[i] << "( mass = "
    //        << particleHandler->h_mass[i] << ")";
    //    }
    //}

    return time;
}
#endif // TARGET_GPU

real Miluphpc::reset() {
    real time;

    *treeHandler->h_tree->index = numParticles;

    std::fill(treeHandler->h_tree->child, treeHandler->h_tree->child + POW_DIM * numNodes, -1);
    std::fill(&particleHandler->h_particles->x[numParticles], &particleHandler->h_particles->x[numNodes], 0.);
    std::fill(&particleHandler->h_particles->y[numParticles], &particleHandler->h_particles->y[numNodes], 0.);
    std::fill(&particleHandler->h_particles->z[numParticles], &particleHandler->h_particles->z[numNodes], 0.);
    std::fill(&particleHandler->h_particles->mass[numParticles], &particleHandler->h_particles->mass[numNodes], 0.);
    std::fill(particleHandler->h_particles->nodeType, &particleHandler->h_particles->nodeType[numNodes], -1);

    domainListHandler->reset();
    lowestDomainListHandler->reset();

#if TARGET_GPU
    // START: resetting arrays, variables, buffers, ...
    Logger(DEBUG) << "resetting (device) arrays ...";
    // -----------------------------------------------------------------------------------------------------------------
    time = Kernel::Launch::resetArrays(treeHandler->d_tree, particleHandler->d_particles, d_mutex, numParticles,
                                       numNodes, true);
    // -----------------------------------------------------------------------------------------------------------------


#if SPH_SIM
    cuda::set(particleHandler->d_u, (real)0., numParticlesLocal);
#endif

    cuda::set(particleHandler->d_ax, (real)0., numParticles);
#if DIM > 1
    cuda::set(particleHandler->d_ay, (real)0., numParticles);
#if DIM == 3
    cuda::set(particleHandler->d_az, (real)0., numParticles);
#endif
#endif

    // TODO: just some testing

    //cuda::set(treeHandler->d_child, -1, POW_DIM * numNodes);
    //cuda::set(&particleHandler->d_mass[numParticlesLocal], (real)0., numNodes - numParticlesLocal);
    //cuda::set(&particleHandler->d_x[numParticlesLocal], (real)0., numNodes - numParticlesLocal);
    //#if DIM > 1
    //cuda::set(&particleHandler->d_y[numParticlesLocal], (real)0., numNodes - numParticlesLocal);
    //#if DIM == 3
    //cuda::set(&particleHandler->d_z[numParticlesLocal], (real)0., numNodes - numParticlesLocal);
    //#endif
    //#endif

    // end: testing

    cuda::set(particleHandler->d_nodeType, 0, numNodes);
#else
    // TODO: CPU reset particle entries
#endif // TARGET_GPU

    //cuda::set(&particleHandler->d_x[numParticles], (real)0., numNodes-numParticles);
    //cuda::set(&particleHandler->d_y[numParticles], (real)0., numNodes-numParticles);
    //cuda::set(&particleHandler->d_z[numParticles], (real)0., numNodes-numParticles);
    //cuda::set(&particleHandler->d_mass[numParticles], (real)0., numNodes-numParticles);

    //helperHandler->reset();
    buffer->reset();
    domainListHandler->reset();
    lowestDomainListHandler->reset();
    subDomainKeyTreeHandler->reset();

#if TARGET_GPU
#if SPH_SIM
    cuda::set(particleHandler->d_noi, 0, numParticles);
    cuda::set(particleHandler->d_nnl, -1, MAX_NUM_INTERACTIONS * numParticles);
#endif
#else
    // TODO: CPU set particle entries (noi, nnl)
#endif

    //Logger(TIME) << "resetArrays: " << time << " ms";
    // END: resetting arrays, variables, buffers, ...

    return time;
}

real Miluphpc::boundingBox() {

    //ParticlesNS::Kernel::Launch::check4nans(particleHandler->d_particles, numParticlesLocal);

    real time;
#if TARGET_GPU
    Logger(DEBUG) << "computing bounding box ...";
    // ---------------------------------------------------------
    time = TreeNS::Kernel::Launch::computeBoundingBox(treeHandler->d_tree, particleHandler->d_particles, d_mutex,
                                                      numParticlesLocal, 256, true);
    // ---------------------------------------------------------

    // this is/was not working properly (why?)
    //*treeHandler->h_minX = HelperNS::reduceAndGlobalize(particleHandler->d_x, treeHandler->d_minX, numParticlesLocal, Reduction::min);
    //*treeHandler->h_maxX = HelperNS::reduceAndGlobalize(particleHandler->d_x, treeHandler->d_maxX, numParticlesLocal, Reduction::max);
    //*treeHandler->h_minY = HelperNS::reduceAndGlobalize(particleHandler->d_y, treeHandler->d_minY, numParticlesLocal, Reduction::min);
    //*treeHandler->h_maxY = HelperNS::reduceAndGlobalize(particleHandler->d_y, treeHandler->d_maxY, numParticlesLocal, Reduction::max);
    //*treeHandler->h_minZ = HelperNS::reduceAndGlobalize(particleHandler->d_z, treeHandler->d_minZ, numParticlesLocal, Reduction::min);
    //*treeHandler->h_maxZ = HelperNS::reduceAndGlobalize(particleHandler->d_z, treeHandler->d_maxZ, numParticlesLocal, Reduction::max);
    //treeHandler->copy(To::device);

    //debug

    /*
    *treeHandler->copy(To::host);
    *treeHandler->h_minX *= 1.1;
    *treeHandler->h_maxX *= 1.1;
    *treeHandler->h_minY *= 1.1;
    *treeHandler->h_maxY *= 1.1;
    *treeHandler->h_minZ *= 1.1;
    *treeHandler->h_maxZ *= 1.1;
    treeHandler->copy(To::device);
    */

    //Logger(INFO) << "x: max = " << *treeHandler->h_maxX << ", min = " << *treeHandler->h_minX;
    //Logger(INFO) << "y: max = " << *treeHandler->h_maxY << ", min = " << *treeHandler->h_minY;
    //Logger(INFO) << "z: max = " << *treeHandler->h_maxZ << ", min = " << *treeHandler->h_minZ;
    //end: debug

    treeHandler->globalizeBoundingBox(Execution::device);
    treeHandler->copy(To::host);

#if DIM == 1
    Logger(DEBUG) << "Bounding box: x = (" << *treeHandler->h_minX << ", " << *treeHandler->h_maxX << ")";
#elif DIM == 2
    Logger(DEBUG) << "Bounding box: x = (" << *treeHandler->h_minX << ", " << *treeHandler->h_maxX << ")" << "y = ("
                 << *treeHandler->h_minY << ", " << *treeHandler->h_maxY << ")";
#else
    Logger(DEBUG) << "Bounding box: x = (" << std::setprecision(9) << *treeHandler->h_minX << ", "
                    << *treeHandler->h_maxX << ")" << "y = (" << *treeHandler->h_minY << ", "
                    << *treeHandler->h_maxY << ")" << "z = " << *treeHandler->h_minZ << ", "
                    << *treeHandler->h_maxZ << ")";
#endif

    Logger(TIME) << "computeBoundingBox: " << time << " ms";
#else
    time = 0.; // TODO: CPU boundingBox()


    std::pair<real*, real*> minmax = std::minmax_element(particleHandler->h_particles->x,
                                                         particleHandler->h_particles->x + numParticlesLocal);
    *treeHandler->h_minX = *(minmax.first);
    *treeHandler->h_maxX = *(minmax.second);
#if DIM > 1
    minmax = std::minmax_element(particleHandler->h_particles->y,
                                 particleHandler->h_particles->y + numParticlesLocal);
    *treeHandler->h_minY = *(minmax.first);
    *treeHandler->h_maxY = *(minmax.second);
#if DIM == 3
    minmax = std::minmax_element(particleHandler->h_particles->z,
                                 particleHandler->h_particles->z + numParticlesLocal);
    *treeHandler->h_minZ = *(minmax.first);
    *treeHandler->h_maxZ = *(minmax.second);
#endif
#endif

    //Logger(TRACE) << "Bounding box: x = (" << std::setprecision(9) << *treeHandler->h_minX << ", "
    //              << *treeHandler->h_maxX << ")" << "y = (" << *treeHandler->h_minY << ", "
    //              << *treeHandler->h_maxY << ")" << "z = " << *treeHandler->h_minZ << ", "
    //              << *treeHandler->h_maxZ << ")";

#endif // TARGET_GPU
    return time;
}

real Miluphpc::assignParticles() {

    real time;
#if TARGET_GPU
    // -----------------------------------------------------------------------------------------------------------------
    time = SubDomainKeyTreeNS::Kernel::Launch::particlesPerProcess(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                                   treeHandler->d_tree, particleHandler->d_particles,
                                                                   numParticlesLocal, numNodes, curveType);
    // -----------------------------------------------------------------------------------------------------------------

    int *d_particlesProcess = buffer->d_integerBuffer; //helperHandler->d_integerBuffer;
    int *d_particlesProcessSorted = buffer->d_integerBuffer1;
    real *d_tempEntry = buffer->d_realBuffer; //helperHandler->d_realBuffer;
    real *d_copyBuffer = buffer->d_realBuffer1;
    idInteger *d_idIntTempEntry = buffer->d_idIntegerBuffer; //d_idIntegerBuffer;
    idInteger *d_idIntCopyBuffer = buffer->d_idIntegerBuffer1; //d_idIntegerCopyBuffer;

    // arrange velocities and accelerations as well (which is normally not necessary)
    bool arrangeAll = false;

    // TODO: principally sorting the keys would only be necessary once
    //  - is it possible to implement a own version of radix sort only sorting the keys ones?
    // ---------------------------------------------------------
    time += SubDomainKeyTreeNS::Kernel::Launch::markParticlesProcess(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                                     treeHandler->d_tree, particleHandler->d_particles,
                                                                     numParticlesLocal, numNodes,
                                                                     d_particlesProcess, curveType);
    // ---------------------------------------------------------
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_x, d_tempEntry);
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_vx, d_tempEntry);
    if (arrangeAll) {
        time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_ax,
                                       d_tempEntry);
        time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_g_ax,
                                       d_tempEntry);
    }
#if DIM > 1
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_y, d_tempEntry);
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_vy, d_tempEntry);
    if (arrangeAll) {
        time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_ay,
                                       d_tempEntry);
        time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_g_ay,
                                       d_tempEntry);
    }
#if DIM == 3
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_z, d_tempEntry);
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_vz, d_tempEntry);
    if (arrangeAll) {
        time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_az,
                                       d_tempEntry);
        time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_g_az,
                                       d_tempEntry);
    }
#endif
#endif

    if (particleHandler->leapfrog) {
        time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted,
                                       particleHandler->d_ax_old, d_tempEntry);
        time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted,
                                       particleHandler->d_g_ax_old, d_tempEntry);
#if DIM > 1
        time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted,
                                       particleHandler->d_ay_old, d_tempEntry);
        time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted,
                                       particleHandler->d_g_ay_old, d_tempEntry);
#if DIM == 3
        time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted,
                                       particleHandler->d_az_old, d_tempEntry);
        time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted,
                                       particleHandler->d_g_az_old, d_tempEntry);
#endif
#endif
    }

    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted,
                                   particleHandler->d_mass, d_tempEntry);
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted,
                                   particleHandler->d_uid, d_idIntTempEntry);

#if SPH_SIM
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_sml, d_tempEntry);
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_e, d_tempEntry);
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_rho, d_tempEntry);
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_cs, d_tempEntry);
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_p, d_tempEntry);
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted,
                                   particleHandler->d_materialId, d_idIntTempEntry);
#endif

    subDomainKeyTreeHandler->copy(To::host, true, true);

    Timer timer;
    integer *sendLengths;
    sendLengths = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    sendLengths[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] = 0;
    integer *receiveLengths;
    receiveLengths = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    receiveLengths[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] = 0;

    for (int proc=0; proc < subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            sendLengths[proc] = subDomainKeyTreeHandler->h_procParticleCounter[proc];
        }
    }
    mpi::messageLengths(subDomainKeyTreeHandler->h_subDomainKeyTree, sendLengths, receiveLengths);

    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_x, d_tempEntry, d_copyBuffer);
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_vx, d_tempEntry, d_copyBuffer);
    if (arrangeAll) {
        sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_ax, d_tempEntry, d_copyBuffer);
        sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_g_ax, d_tempEntry, d_copyBuffer);
    }
#if DIM > 1
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_y, d_tempEntry, d_copyBuffer);
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_vy, d_tempEntry, d_copyBuffer);
    if (arrangeAll) {
        sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_ay, d_tempEntry, d_copyBuffer);
        sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_g_ay, d_tempEntry, d_copyBuffer);
    }
#if DIM == 3
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_z, d_tempEntry, d_copyBuffer);
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_vz, d_tempEntry, d_copyBuffer);
    if (arrangeAll) {
        sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_az, d_tempEntry, d_copyBuffer);
        sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_g_az, d_tempEntry, d_copyBuffer);
    }
#endif
#endif

    if (particleHandler->leapfrog) {
        sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_ax_old, d_tempEntry, d_copyBuffer);
        sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_g_ax_old, d_tempEntry, d_copyBuffer);
#if DIM > 1
        sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_ay_old, d_tempEntry, d_copyBuffer);
        sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_g_ay_old, d_tempEntry, d_copyBuffer);
#if DIM == 3
        sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_az_old, d_tempEntry, d_copyBuffer);
        sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_g_az_old, d_tempEntry, d_copyBuffer);
#endif
#endif
    }

    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_uid, d_idIntTempEntry, d_idIntCopyBuffer);

#if SPH_SIM
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_sml, d_tempEntry, d_copyBuffer);
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_e, d_tempEntry, d_copyBuffer);
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_rho, d_tempEntry, d_copyBuffer);
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_cs, d_tempEntry, d_copyBuffer);
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_p, d_tempEntry, d_copyBuffer);
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_materialId, d_idIntTempEntry, d_idIntCopyBuffer);
#endif
    numParticlesLocal = sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_mass,
                                           d_tempEntry, d_copyBuffer);

    if (numParticlesLocal > numParticles) {
        MPI_Finalize();
        Logger(ERROR) << "numParticlesLocal = " << numParticlesLocal << " > "
                           << " numParticles = " << numParticles;
        // TODO: implement possibility to restart simulation
        Logger(ERROR) << "Restart simulation with more memory! exiting ...";
        exit(1);
    }

    delete [] sendLengths;
    delete [] receiveLengths;

    time += timer.elapsed();

    int resetLength = numParticles-numParticlesLocal;
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_x[numParticlesLocal],
                                                 (real)0, resetLength);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_vx[numParticlesLocal],
                                                 (real)0, resetLength);
    if (arrangeAll) {
        time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_ax[numParticlesLocal],
                                                     (real) 0, resetLength);
        time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_g_ax[numParticlesLocal],
                                                     (real) 0, resetLength);
    }
#if DIM > 1
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_y[numParticlesLocal],
                                                 (real)0, resetLength);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_vy[numParticlesLocal],
                                                 (real)0, resetLength);
    if (arrangeAll) {
        time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_ay[numParticlesLocal],
                                                     (real) 0, resetLength);
        time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_g_ay[numParticlesLocal],
                                                     (real) 0, resetLength);
    }
#if DIM == 3
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_z[numParticlesLocal],
                                                 (real)0, resetLength);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_vz[numParticlesLocal],
                                                 (real)0, resetLength);
    if (arrangeAll) {
        time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_az[numParticlesLocal],
                                                     (real) 0, resetLength);
        time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_g_az[numParticlesLocal],
                                                     (real) 0, resetLength);
    }
#endif
#endif

    if (particleHandler->leapfrog) {
        time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_ax_old[numParticlesLocal],
                                                     (real) 0, resetLength);
        time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_g_ax_old[numParticlesLocal],
                                                     (real) 0, resetLength);
#if DIM > 1
        time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_ay_old[numParticlesLocal],
                                                     (real) 0, resetLength);
        time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_g_ay_old[numParticlesLocal],
                                                     (real) 0, resetLength);
#if DIM == 3
        time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_az_old[numParticlesLocal],
                                                     (real) 0, resetLength);
        time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_g_az_old[numParticlesLocal],
                                                     (real) 0, resetLength);
#endif
#endif
    }

    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_mass[numParticlesLocal],
                                                 (real)0, resetLength);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_uid[numParticlesLocal],
                                                 (idInteger)0, resetLength);

#if SPH_SIM
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_sml[numParticlesLocal], (real)0, resetLength);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_e[numParticlesLocal], (real)0, resetLength);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_rho[numParticlesLocal], (real)0, resetLength);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_cs[numParticlesLocal], (real)0, resetLength);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_p[numParticlesLocal], (real)0, resetLength);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_materialId[numParticlesLocal],
                                                 (integer)0, resetLength);
#endif

#else
    time = 0.; // TODO: CPU assignParticles()
#endif // TARGET_GPU
    return time;
}

template <typename U, typename T>
real Miluphpc::arrangeParticleEntries(U *sortArray, U *sortedArray, T *entry, T *temp) {
    real time;
#if TARGET_GPU
    time = HelperNS::sortArray(entry, temp, sortArray, sortedArray, numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(entry, temp, numParticlesLocal);
#else
    time = 0.; // TODO: CPU arrangeParticleEntries()
#endif
    return time;
}


real Miluphpc::tree() {

#if TARGET_GPU
    real time = parallel_tree();
#else
    real time = 0.;

#endif
    return time;
}

real Miluphpc::cpu_tree() {

    // TODO: cpu_tree();
    return 0.;
}

#if TARGET_GPU
real Miluphpc::parallel_tree() {
    real time;
    real totalTime = 0.;

    // START: creating domain list
    Logger(DEBUG) << "building domain list ...";
    // -----------------------------------------------------------------------------------------------------------------
    if (subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses > 1) {
        time = DomainListNS::Kernel::Launch::createDomainList(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                              domainListHandler->d_domainList, MAX_LEVEL,
                                                              curveType);
    }
    // -----------------------------------------------------------------------------------------------------------------
    totalTime += time;
    Logger(DEBUG) << "createDomainList: " << time << " ms";
    profiler.value2file(ProfilerIds::Time::Tree::createDomain, time);

    integer domainListLength;
    cuda::copy(&domainListLength, domainListHandler->d_domainListIndex, 1, To::host);
    Logger(DEBUG) << "domainListLength = " << domainListLength;
    // END: creating domain list

    // START: tree construction (including common coarse tree)
#if DEBUGGING
    integer treeIndexBeforeBuildingTree;
    cuda::copy(&treeIndexBeforeBuildingTree, treeHandler->d_index, 1, To::host);
    Logger(DEBUG) << "treeIndexBeforeBuildingTree: " << treeIndexBeforeBuildingTree;
#endif

    Logger(DEBUG) << "building tree ...";
    // ---------------------------------------------------------
    time = TreeNS::Kernel::Launch::buildTree(treeHandler->d_tree, particleHandler->d_particles, numParticlesLocal,
                                             numParticles, true);
    // ---------------------------------------------------------
    totalTime += time;
    Logger(TIME) << "buildTree: " << time << " ms";
    profiler.value2file(ProfilerIds::Time::Tree::tree, time);

    integer treeIndex;
    cuda::copy(&treeIndex, treeHandler->d_index, 1, To::host);

    Logger(INFO) << "treeIndex: " << treeIndex << " vs. numNodes: " << numNodes << " = "
                      << (double)treeIndex / (double)numNodes * 100. << " %";

#if DEBUGGING
    Logger(INFO) << "numParticlesLocal: " << numParticlesLocal;
    Logger(INFO) << "numParticles: " << numParticles;
    Logger(INFO) << "numNodes: " << numNodes;
    Logger(INFO) << "treeIndex: " << treeIndex;
    integer numParticlesSum = numParticlesLocal;
    boost::mpi::communicator comm;
    all_reduce(comm, boost::mpi::inplace_t<integer*>(&numParticlesSum), 1, std::plus<integer>());
    Logger(INFO) << "numParticlesSum: " << numParticlesSum;
    //ParticlesNS::Kernel::Launch::info(particleHandler->d_particles, numParticlesLocal, numParticles, treeIndex);
#endif

    Logger(DEBUG) << "building domain tree ...";
    cuda::set(domainListHandler->d_domainListCounter, 0, 1);

    // serial version
    //time = SubDomainKeyTreeNS::Kernel::Launch::buildDomainTree(treeHandler->d_tree, particleHandler->d_particles,
    //                                                           domainListHandler->d_domainList, numParticlesLocal,
    //                                                           numNodes);

    time = 0;
    // parallel version
    // serial version (above) working for one process, "parallel" version not working for one process, thus
    //  if statement introduced
    if (subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses > 1) {
        for (int level = 0; level <= MAX_LEVEL; level++) {
            // ---------------------------------------------------------------------------------------------------------
            time += SubDomainKeyTreeNS::Kernel::Launch::buildDomainTree(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                                        treeHandler->d_tree,
                                                                        particleHandler->d_particles,
                                                                        domainListHandler->d_domainList,
                                                                        numParticlesLocal,
                                                                        numNodes, level);
            // ---------------------------------------------------------------------------------------------------------
        }
    }

#if DEBUGGING
    int domainListCounterAfterwards;
    cuda::copy(&domainListCounterAfterwards, domainListHandler->d_domainListCounter, 1, To::host);
    Logger(DEBUG) << "domain list counter afterwards : " << domainListCounterAfterwards;
#endif

    cuda::set(domainListHandler->d_domainListCounter, 0, 1);
    // END: tree construction (including common coarse tree)
    totalTime += time;
    Logger(TIME) << "build(Domain)Tree: " << time << " ms";
    profiler.value2file(ProfilerIds::Time::Tree::buildDomain, time);

    return totalTime;
}
#endif // TARGET_GPU


real Miluphpc::pseudoParticles() {

#if TARGET_GPU
    real time = parallel_pseudoParticles();
#else
    real time = 0.;
    cpu_pseudoParticles();

#endif // TARGET_GPU
    return time;
}

real Miluphpc::cpu_pseudoParticles() {
    // TODO: CPU pseudoParticles()

    TreeNS::compPseudoParticles(treeHandler->h_tree, particleHandler->h_particles, domainListHandler->h_domainList,
                                numParticles, 0);

    TreeNS::lowestDomainListNodes(treeHandler->h_tree, particleHandler->h_particles, domainListHandler->h_domainList,
                                  lowestDomainListHandler->h_domainList, numParticles);

    TreeNS::zeroDomainListNodes(treeHandler->h_tree, particleHandler->h_particles, domainListHandler->h_domainList);

    integer domainListIndex = *domainListHandler->h_domainList->domainListIndex;
    integer lowestDomainListIndex = *lowestDomainListHandler->h_domainList->domainListIndex;

    Logger(DEBUG) << "domainListIndex: " << domainListIndex << " | lowestDomainListIndex: " << lowestDomainListIndex;
    Logger(DEBUG) << "communicating/exchanging and updating domain list nodes ...";

    boost::mpi::communicator comm;

    //TODO: current approach reasonable?
    // or template functions and explicitly hand over buffer(s) (and not instance of buffer class)

    // x ---------------------------------------------------------------------------------------------------------------

    lowestDomainListHandler->h_domainList->domainListCounter = 0;

    real *h_realBuffer = new real[lowestDomainListIndex];

    // ---- x ---------------------
    for (int i=0; i<lowestDomainListIndex; ++i) {
        h_realBuffer[i] = particleHandler->h_particles->x[lowestDomainListHandler->h_domainList->domainListIndices[i]];
        //Logger(TRACE) << "index: " << lowestDomainListHandler->h_domainList->domainListIndices[i] << " | h_realBuffer[" << i << "] = " << h_realBuffer[i];
        //Logger(TRACE) << "before: x[" << i << "] = " << particleHandler->h_particles->x[lowestDomainListHandler->h_domainList->domainListIndices[i]];
    }

    all_reduce(comm, boost::mpi::inplace_t<real*>(&h_realBuffer[0]), lowestDomainListIndex, std::plus<real>());

    for (int i=0; i<lowestDomainListIndex; ++i) {
        particleHandler->h_particles->x[lowestDomainListHandler->h_domainList->domainListIndices[i]] = h_realBuffer[i];
        //Logger(TRACE) << "after: x[" << i << "] = " << particleHandler->h_particles->x[lowestDomainListHandler->h_domainList->domainListIndices[i]];
    }
    // ---- end: x ---------------------
#if DIM > 1
    // ---- y ---------------------
    for (int i=0; i<lowestDomainListIndex; ++i) {
        h_realBuffer[i] = particleHandler->h_particles->y[lowestDomainListHandler->h_domainList->domainListIndices[i]];
    }

    all_reduce(comm, boost::mpi::inplace_t<real*>(&h_realBuffer[0]), lowestDomainListIndex, std::plus<real>());

    for (int i=0; i<lowestDomainListIndex; ++i) {
        particleHandler->h_particles->y[lowestDomainListHandler->h_domainList->domainListIndices[i]] = h_realBuffer[i];
    }
    // ---- end: y ---------------------
#if DIM == 3
    // ---- z ---------------------
    for (int i=0; i<lowestDomainListIndex; ++i) {
        h_realBuffer[i] = particleHandler->h_particles->z[lowestDomainListHandler->h_domainList->domainListIndices[i]];
    }

    all_reduce(comm, boost::mpi::inplace_t<real*>(&h_realBuffer[0]), lowestDomainListIndex, std::plus<real>());

    for (int i=0; i<lowestDomainListIndex; ++i) {
        particleHandler->h_particles->z[lowestDomainListHandler->h_domainList->domainListIndices[i]] = h_realBuffer[i];
    }
    // ---- end: z ---------------------
#endif
#endif

    // ---- mass ---------------------
    for (int i=0; i<lowestDomainListIndex; ++i) {
        h_realBuffer[i] = particleHandler->h_particles->mass[lowestDomainListHandler->h_domainList->domainListIndices[i]];
    }

    all_reduce(comm, boost::mpi::inplace_t<real*>(&h_realBuffer[0]), lowestDomainListIndex, std::plus<real>());

    for (int i=0; i<lowestDomainListIndex; ++i) {
        particleHandler->h_particles->mass[lowestDomainListHandler->h_domainList->domainListIndices[i]] = h_realBuffer[i];
    }
    // ---- end: mass ---------------------

    for (int i=0; i<lowestDomainListIndex; ++i) {
        if (particleHandler->h_particles->mass[lowestDomainListHandler->h_domainList->domainListIndices[i]] > 0) {
            particleHandler->h_particles->x[lowestDomainListHandler->h_domainList->domainListIndices[i]] /= particleHandler->h_particles->mass[lowestDomainListHandler->h_domainList->domainListIndices[i]];
#if DIM > 1
            particleHandler->h_particles->y[lowestDomainListHandler->h_domainList->domainListIndices[i]] /= particleHandler->h_particles->mass[lowestDomainListHandler->h_domainList->domainListIndices[i]];
#if DIM == 3
            particleHandler->h_particles->z[lowestDomainListHandler->h_domainList->domainListIndices[i]] /= particleHandler->h_particles->mass[lowestDomainListHandler->h_domainList->domainListIndices[i]];
#endif
#endif
        }
    }

    //for (int i=0; i<lowestDomainListIndex; ++i) {
    //}

    for (int domainLevel = MAX_LEVEL; domainLevel>= 0; domainLevel--) {
        // -------------------------------------------------------------------------------------------------------------
        TreeNS::compDomainListPseudoParticlesPerLevel(treeHandler->h_tree, particleHandler->h_particles,
                                                      domainListHandler->h_domainList,
                                                      lowestDomainListHandler->h_domainList, domainLevel);
        // -------------------------------------------------------------------------------------------------------------
    }

    delete [] h_realBuffer;

    /*
    // -----------------------------------------------------------------------------------------------------------------

    Logger(DEBUG) << "finish computation of lowest domain list nodes ...";
    // -----------------------------------------------------------------------------------------------------------------
    time += SubDomainKeyTreeNS::Kernel::Launch::compLowestDomainListNodes(treeHandler->d_tree,
                                                                          particleHandler->d_particles,
                                                                          lowestDomainListHandler->d_domainList);
    // -----------------------------------------------------------------------------------------------------------------
    //end: for all entries!

    Logger(DEBUG) << "finish computation of (all) domain list nodes ...";
    // per level computation of domain list pseudo-particles to ensure the correct order (avoid race condition)
    for (int domainLevel = MAX_LEVEL; domainLevel>= 0; domainLevel--) {
        // -------------------------------------------------------------------------------------------------------------
        time += SubDomainKeyTreeNS::Kernel::Launch::compDomainListPseudoParticlesPerLevel(treeHandler->d_tree,
                                                                                          particleHandler->d_particles,
                                                                                          domainListHandler->d_domainList,
                                                                                          lowestDomainListHandler->d_domainList,
                                                                                          numParticles, domainLevel);
        // -------------------------------------------------------------------------------------------------------------
    }
     */

    return 0.;
}

#if TARGET_GPU
real Miluphpc::parallel_pseudoParticles() {

    real time = 0;
    Logger(DEBUG) << "lowestDomainList() ...";
    // -----------------------------------------------------------------------------------------------------------------
    time += DomainListNS::Kernel::Launch::lowestDomainList(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                           treeHandler->d_tree, particleHandler->d_particles,
                                                           domainListHandler->d_domainList,
                                                           lowestDomainListHandler->d_domainList,
                                                           numParticles, numNodes);
    // -----------------------------------------------------------------------------------------------------------------

    Logger(DEBUG) << "calculateCentersOfMass() ...";
    real timeCOM = 0;
    for (int level=MAX_LEVEL; level>0; --level) {
        // -------------------------------------------------------------------------------------------------------------
        timeCOM += TreeNS::Kernel::Launch::calculateCentersOfMass(treeHandler->d_tree, particleHandler->d_particles,
                                                                  numParticles, level, true);
        // -------------------------------------------------------------------------------------------------------------
    }

    //timeCOM = TreeNS::Kernel::Launch::centerOfMass(treeHandler->d_tree, particleHandler->d_particles,
    // numParticlesLocal, true);

    Logger(TIME) << "calculate COM: " << timeCOM << " ms";
    time += timeCOM;

    // -----------------------------------------------------------------------------------------------------------------
    SubDomainKeyTreeNS::Kernel::Launch::zeroDomainListNodes(particleHandler->d_particles,
                                                            domainListHandler->d_domainList,
                                                            lowestDomainListHandler->d_domainList);
    // -----------------------------------------------------------------------------------------------------------------

    // old version
    //time += SubDomainKeyTreeNS::Kernel::Launch::compLocalPseudoParticles(treeHandler->d_tree,
    // particleHandler->d_particles, domainListHandler->d_domainList, numParticles);
    //DomainListNS::Kernel::Launch::info(particleHandler->d_particles, domainListHandler->d_domainList);

    integer domainListIndex;
    integer lowestDomainListIndex;

    cuda::copy(&domainListIndex, domainListHandler->d_domainListIndex, 1, To::host);
    cuda::copy(&lowestDomainListIndex, lowestDomainListHandler->d_domainListIndex, 1, To::host);

    Logger(DEBUG) << "domainListIndex: " << domainListIndex << " | lowestDomainListIndex: "
                       << lowestDomainListIndex;
    Logger(DEBUG) << "communicating/exchanging and updating domain list nodes ...";

    boost::mpi::communicator comm;

    //TODO: current approach reasonable?
    // or template functions and explicitly hand over buffer(s) (and not instance of buffer class)

    // x ---------------------------------------------------------------------------------------------------------------

    cuda::set(lowestDomainListHandler->d_domainListCounter, 0);
    time += SubDomainKeyTreeNS::Kernel::Launch::prepareLowestDomainExchange(particleHandler->d_particles,
                                                                            lowestDomainListHandler->d_domainList,
                                                                            buffer->d_realBuffer, Entry::x);

    time += HelperNS::sortArray(buffer->d_realBuffer,
                                &buffer->d_realBuffer[simulationParameters.domainListSize],
                                lowestDomainListHandler->d_domainListKeys,
                                lowestDomainListHandler->d_sortedDomainListKeys,
                                domainListIndex);

    // share among processes
    //TODO: domainListIndex or lowestDomainListIndex?
    all_reduce(comm, boost::mpi::inplace_t<real*>(&buffer->d_realBuffer[simulationParameters.domainListSize]),
               domainListIndex, std::plus<real>());

    cuda::set(lowestDomainListHandler->d_domainListCounter, 0);

    time += SubDomainKeyTreeNS::Kernel::Launch::updateLowestDomainListNodes(particleHandler->d_particles,
                                                                            lowestDomainListHandler->d_domainList,
                                                                            &buffer->d_realBuffer[simulationParameters.domainListSize],
                                                                            Entry::x);

#if DIM > 1
    // y ---------------------------------------------------------------------------------------------------------------

    cuda::set(lowestDomainListHandler->d_domainListCounter, 0);
    time += SubDomainKeyTreeNS::Kernel::Launch::prepareLowestDomainExchange(particleHandler->d_particles,
                                                                            lowestDomainListHandler->d_domainList,
                                                                            buffer->d_realBuffer, Entry::y);

    time += HelperNS::sortArray(buffer->d_realBuffer,
                                &buffer->d_realBuffer[simulationParameters.domainListSize],
                                lowestDomainListHandler->d_domainListKeys,
                                lowestDomainListHandler->d_sortedDomainListKeys,
                                domainListIndex);

    all_reduce(comm, boost::mpi::inplace_t<real*>(&buffer->d_realBuffer[simulationParameters.domainListSize]),
               domainListIndex, std::plus<real>());

    cuda::set(lowestDomainListHandler->d_domainListCounter, 0);

    time += SubDomainKeyTreeNS::Kernel::Launch::updateLowestDomainListNodes(particleHandler->d_particles,
                                                                            lowestDomainListHandler->d_domainList,
                                                                            &buffer->d_realBuffer[simulationParameters.domainListSize],
                                                                            Entry::y);

#if DIM == 3
    // z ---------------------------------------------------------------------------------------------------------------

    cuda::set(lowestDomainListHandler->d_domainListCounter, 0);
    time += SubDomainKeyTreeNS::Kernel::Launch::prepareLowestDomainExchange(particleHandler->d_particles,
                                                                            lowestDomainListHandler->d_domainList,
                                                                            buffer->d_realBuffer, Entry::z);

    time += HelperNS::sortArray(buffer->d_realBuffer,
                                &buffer->d_realBuffer[simulationParameters.domainListSize],
                                lowestDomainListHandler->d_domainListKeys,
                                lowestDomainListHandler->d_sortedDomainListKeys,
                                domainListIndex);

    all_reduce(comm, boost::mpi::inplace_t<real*>(&buffer->d_realBuffer[simulationParameters.domainListSize]),
               domainListIndex, std::plus<real>());

    cuda::set(lowestDomainListHandler->d_domainListCounter, 0);

    time += SubDomainKeyTreeNS::Kernel::Launch::updateLowestDomainListNodes(particleHandler->d_particles,
                                                                            lowestDomainListHandler->d_domainList,
                                                                            &buffer->d_realBuffer[simulationParameters.domainListSize],
                                                                            Entry::z);

#endif
#endif

    // m ---------------------------------------------------------------------------------------------------------------

    cuda::set(lowestDomainListHandler->d_domainListCounter, 0);
    time += SubDomainKeyTreeNS::Kernel::Launch::prepareLowestDomainExchange(particleHandler->d_particles,
                                                                            lowestDomainListHandler->d_domainList,
                                                                            buffer->d_realBuffer, Entry::mass);

    time += HelperNS::sortArray(buffer->d_realBuffer,
                                &buffer->d_realBuffer[simulationParameters.domainListSize],
                                lowestDomainListHandler->d_domainListKeys,
                                lowestDomainListHandler->d_sortedDomainListKeys,
                                domainListIndex);

    all_reduce(comm, boost::mpi::inplace_t<real*>(&buffer->d_realBuffer[simulationParameters.domainListSize]),
               domainListIndex, std::plus<real>());

    cuda::set(lowestDomainListHandler->d_domainListCounter, 0);

    time += SubDomainKeyTreeNS::Kernel::Launch::updateLowestDomainListNodes(particleHandler->d_particles,
                                                                            lowestDomainListHandler->d_domainList,
                                                                            &buffer->d_realBuffer[simulationParameters.domainListSize],
                                                                            Entry::mass);

    // -----------------------------------------------------------------------------------------------------------------

    Logger(DEBUG) << "finish computation of lowest domain list nodes ...";
    // -----------------------------------------------------------------------------------------------------------------
    time += SubDomainKeyTreeNS::Kernel::Launch::compLowestDomainListNodes(treeHandler->d_tree,
                                                                          particleHandler->d_particles,
                                                                          lowestDomainListHandler->d_domainList);
    // -----------------------------------------------------------------------------------------------------------------
    //end: for all entries!

    Logger(DEBUG) << "finish computation of (all) domain list nodes ...";
    // per level computation of domain list pseudo-particles to ensure the correct order (avoid race condition)
    for (int domainLevel = MAX_LEVEL; domainLevel>= 0; domainLevel--) {
        // -------------------------------------------------------------------------------------------------------------
        time += SubDomainKeyTreeNS::Kernel::Launch::compDomainListPseudoParticlesPerLevel(treeHandler->d_tree,
                                                                                          particleHandler->d_particles,
                                                                                          domainListHandler->d_domainList,
                                                                                          lowestDomainListHandler->d_domainList,
                                                                                          numParticles, domainLevel);
        // -------------------------------------------------------------------------------------------------------------
    }

    //timeCOM = TreeNS::Kernel::Launch::centerOfMass(treeHandler->d_tree, particleHandler->d_particles, numParticlesLocal,
    //                                                       true);

    return time;
}
#endif // TARGET_GPU

real Miluphpc::gravity() {

#if TARGET_GPU
    real time = parallel_gravity();
#else
    real time = 0.;
#endif // TARGET_GPU

    return time;
}

real Miluphpc::cpu_gravity() {

    // TODO: CPU gravity()
    return 0.;
}

#if TARGET_GPU
real Miluphpc::parallel_gravity() {

    real time;
    real totalTime = 0;

    /*
#if UNIT_TESTING
    //if (subStep == 10) {
        int actualTreeIndex;
        cuda::copy(&actualTreeIndex, treeHandler->d_index, 1, To::host);
        Logger(TRACE) << "[" << subStep << "] Checking Masses ...";
        UnitTesting::Kernel::Launch::test_localTree(treeHandler->d_tree, particleHandler->d_particles, numParticles, actualTreeIndex); //numNodes);
    //}
#endif
     */

    totalTime += HelperNS::Kernel::Launch::resetArray(buffer->d_realBuffer, (real)0, numParticles);

    cuda::set(domainListHandler->d_domainListCounter, 0);

    Logger(DEBUG) << "compTheta()";
    // -----------------------------------------------------------------------------------------------------------------
    time = Gravity::Kernel::Launch::compTheta(subDomainKeyTreeHandler->d_subDomainKeyTree, treeHandler->d_tree,
                                              particleHandler->d_particles, lowestDomainListHandler->d_domainList, //domainListHandler->d_domainList,
                                              curveType);
    // -----------------------------------------------------------------------------------------------------------------
    totalTime += time;
    Logger(TIME) << "compTheta(): " << time << " ms";
    profiler.value2file(ProfilerIds::Time::Gravity::compTheta, time);

    integer relevantIndicesCounter;
    // changed from domainListHandler to lowestDomainListHandler
    cuda::copy(&relevantIndicesCounter, lowestDomainListHandler->d_domainListCounter, 1, To::host);
    //cuda::copy(&relevantIndicesCounter, domainListHandler->d_domainListCounter, 1, To::host);

    //Logger(DEBUG) << "relevantIndicesCounter: " << relevantIndicesCounter;

    integer *h_relevantDomainListProcess;
    h_relevantDomainListProcess = new integer[relevantIndicesCounter];
    //cuda::copy(h_relevantDomainListProcess, domainListHandler->d_relevantDomainListProcess,
    // relevantIndicesCounter, To::host);
    // changed from domainListHandler to lowestDomainListHandler
    cuda::copy(h_relevantDomainListProcess, lowestDomainListHandler->d_relevantDomainListProcess,
               relevantIndicesCounter, To::host);
    //cuda::copy(h_relevantDomainListProcess, domainListHandler->d_relevantDomainListProcess,
    //           relevantIndicesCounter, To::host);

    //for (int i=0; i<relevantIndicesCounter; i++) {
    //    Logger(DEBUG) << "relevantDomainListProcess[" << i << "] = " << h_relevantDomainListProcess[i];
    //}

    treeHandler->copy(To::host);
#if CUBIC_DOMAINS
    real diam = std::abs(*treeHandler->h_maxX) + std::abs(*treeHandler->h_minX);
    //Logger(INFO) << "diam: " << diam;
#else // !CUBIC DOMAINS
    real diam_x = std::abs(*treeHandler->h_maxX) + std::abs(*treeHandler->h_minX);
#if DIM > 1
    real diam_y = std::abs(*treeHandler->h_maxY) + std::abs(*treeHandler->h_minY);
#if DIM == 3
    real diam_z = std::abs(*treeHandler->h_maxZ) + std::abs(*treeHandler->h_minZ);
#endif
#endif
#if DIM == 1
    real diam = diam_x;
    //Logger(INFO) << "diam: " << diam << "  (x = " << diam_x << ")";
#elif DIM == 2
    real diam = std::max({diam_x, diam_y});
    //Logger(INFO) << "diam: " << diam << "  (x = " << diam_x << ", y = " << diam_y << ")";
#else
    real diam = std::max({diam_x, diam_y, diam_z});
    //Logger(INFO) << "diam: " << diam << "  (x = " << diam_x << ", y = " << diam_y << ", z = " << diam_z << ")";
#endif
#endif // CUBIC_DOMAINS

    // TODO: create buffer concept and (re)use buffer for gravity internode communicaton
    // changed from domainListHandler to lowestDomainListHandler
    cuda::set(lowestDomainListHandler->d_domainListCounter, 0);
    //cuda::set(domainListHandler->d_domainListCounter, 0);

    integer *d_markedSendIndices = buffer->d_integerBuffer;
    real *d_collectedEntries = buffer->d_realBuffer;

    integer *d_particles2SendIndices = buffer->d_integerBuffer1;
    cuda::set(d_particles2SendIndices, -1, numParticles);
    integer *d_pseudoParticles2SendIndices = buffer->d_integerBuffer2;
    cuda::set(d_pseudoParticles2SendIndices, -1, numParticles);
    integer *d_pseudoParticles2SendLevels = buffer->d_integerBuffer3;
    cuda::set(d_pseudoParticles2SendLevels, -1, numParticles);
    integer *d_pseudoParticles2ReceiveLevels = buffer->d_integerBuffer4;
    cuda::set(d_pseudoParticles2ReceiveLevels, -1, numParticles);

    integer *d_particles2SendCount = buffer->d_sendCount;
    cuda::set(d_particles2SendCount, 0, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses);
    integer *d_pseudoParticles2SendCount = buffer->d_sendCount1;
    cuda::set(d_pseudoParticles2SendCount, 0, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses);

    integer particlesOffset = 0;
    integer pseudoParticlesOffset = 0;

    integer particlesOffsetBuffer;
    integer pseudoParticlesOffsetBuffer;

    integer *h_particles2SendCount = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    integer *h_pseudoParticles2SendCount = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];

    int symbolicForceVersion = 4;

    time = 0;
    if (symbolicForceVersion == 0) {
        for (int proc = 0; proc < subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
            if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
                cuda::set(d_markedSendIndices, -1, numNodes);
                for (int level = 0; level < MAX_LEVEL; level++) {
                    // ---------------------------------------------------------
                    // changed from domainListHandler to lowestDomainListHandler
                    time += Gravity::Kernel::Launch::intermediateSymbolicForce(
                            subDomainKeyTreeHandler->d_subDomainKeyTree,
                            treeHandler->d_tree, particleHandler->d_particles, lowestDomainListHandler->d_domainList,
                            d_markedSendIndices, diam, simulationParameters.theta, numParticlesLocal, numParticles, 0,
                            level,
                            curveType);
                    //time += Gravity::Kernel::Launch::intermediateSymbolicForce(subDomainKeyTreeHandler->d_subDomainKeyTree,
                    //                                                           treeHandler->d_tree, particleHandler->d_particles, domainListHandler->d_domainList,
                    //                                                           d_markedSendIndices, diam, simulationParameters.theta, numParticlesLocal, numParticles,0, level,
                    //                                                           curveType);
                    // ---------------------------------------------------------
                    for (int relevantIndex = 0; relevantIndex < relevantIndicesCounter; relevantIndex++) {
                        if (h_relevantDomainListProcess[relevantIndex] == proc) {
                            //Logger(INFO) << "h_relevantDomainListProcess[" << relevantIndex << "] = "
                            //             << h_relevantDomainListProcess[relevantIndex];
                            // ---------------------------------------------------------
                            // changed from domainListHandler to lowestDomainListHandler
                            time += Gravity::Kernel::Launch::symbolicForce(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                                           treeHandler->d_tree,
                                                                           particleHandler->d_particles,
                                                                           lowestDomainListHandler->d_domainList,
                                                                           d_markedSendIndices, diam,
                                                                           simulationParameters.theta,
                                                                           numParticlesLocal, numParticles,
                                                                           relevantIndex, level, curveType);
                            //time += Gravity::Kernel::Launch::symbolicForce(subDomainKeyTreeHandler->d_subDomainKeyTree,
                            //                                               treeHandler->d_tree, particleHandler->d_particles,
                            //                                               domainListHandler->d_domainList,
                            //                                               d_markedSendIndices, diam,
                            //                                               simulationParameters.theta,
                            //                                               numParticlesLocal, numParticles,
                            //                                               relevantIndex, level, curveType);
                            // ---------------------------------------------------------
                        }
                    }
                }
                // changed from domainListHandler to lowestDomainListHandler
                time += Gravity::Kernel::Launch::intermediateSymbolicForce(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                                           treeHandler->d_tree,
                                                                           particleHandler->d_particles,
                                                                           lowestDomainListHandler->d_domainList,
                                                                           d_markedSendIndices, diam,
                                                                           simulationParameters.theta,
                                                                           numParticlesLocal, numParticles, 0, 0,
                                                                           curveType);
                //time += Gravity::Kernel::Launch::intermediateSymbolicForce(subDomainKeyTreeHandler->d_subDomainKeyTree,
                //                                                           treeHandler->d_tree, particleHandler->d_particles, domainListHandler->d_domainList,
                //                                                           d_markedSendIndices, diam, simulationParameters.theta, numParticlesLocal, numParticles,0, 0,
                //                                                           curveType);
                // ---------------------------------------------------------
                time += Gravity::Kernel::Launch::collectSendIndices(treeHandler->d_tree, particleHandler->d_particles,
                                                                    d_markedSendIndices,
                                                                    &d_particles2SendIndices[particlesOffset],
                                                                    &d_pseudoParticles2SendIndices[pseudoParticlesOffset],
                                                                    &d_pseudoParticles2SendLevels[pseudoParticlesOffset],
                                                                    &d_particles2SendCount[proc],
                                                                    &d_pseudoParticles2SendCount[proc],
                                                                    numParticles, numNodes, curveType);
                // ---------------------------------------------------------

                cuda::copy(&particlesOffsetBuffer, &d_particles2SendCount[proc], 1, To::host);
                cuda::copy(&pseudoParticlesOffsetBuffer, &d_pseudoParticles2SendCount[proc], 1, To::host);

                Logger(DEBUG) << "particles2SendCount[" << proc << "] = " << particlesOffsetBuffer;
                Logger(DEBUG) << "pseudoParticles2SendCount[" << proc << "] = " << pseudoParticlesOffsetBuffer;

                particlesOffset += particlesOffsetBuffer;
                pseudoParticlesOffset += pseudoParticlesOffsetBuffer;
            }
        }
    }
    else if (symbolicForceVersion == 1) {
        // symbolic force testing
        time = 0;
        for (int proc = 0; proc < subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
            if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
                cuda::set(d_markedSendIndices, -1, numNodes);
                for (int relevantIndex = 0; relevantIndex < relevantIndicesCounter; relevantIndex++) {
                    if (h_relevantDomainListProcess[relevantIndex] == proc) {
                        time += Gravity::Kernel::Launch::symbolicForce_test(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                                            treeHandler->d_tree,
                                                                            particleHandler->d_particles,
                                                                            lowestDomainListHandler->d_domainList,
                                                                            d_markedSendIndices, diam,
                                                                            simulationParameters.theta,
                                                                            numParticlesLocal, numParticles,
                                                                            relevantIndex, 0, curveType);
                    }
                }
                time += Gravity::Kernel::Launch::collectSendIndices(treeHandler->d_tree, particleHandler->d_particles,
                                                                    d_markedSendIndices,
                                                                    &d_particles2SendIndices[particlesOffset],
                                                                    &d_pseudoParticles2SendIndices[pseudoParticlesOffset],
                                                                    &d_pseudoParticles2SendLevels[pseudoParticlesOffset],
                                                                    &d_particles2SendCount[proc],
                                                                    &d_pseudoParticles2SendCount[proc],
                                                                    numParticles, numNodes, curveType);

                cuda::copy(&particlesOffsetBuffer, &d_particles2SendCount[proc], 1, To::host);
                cuda::copy(&pseudoParticlesOffsetBuffer, &d_pseudoParticles2SendCount[proc], 1, To::host);

                Logger(DEBUG) << "particles2SendCount[" << proc << "] = " << particlesOffsetBuffer;
                Logger(DEBUG) << "pseudoParticles2SendCount[" << proc << "] = " << pseudoParticlesOffsetBuffer;

                particlesOffset += particlesOffsetBuffer;
                pseudoParticlesOffset += pseudoParticlesOffsetBuffer;
            }
        }
        // end: symbolic force testing
    }
    else if (symbolicForceVersion == 2) {
        // symbolic force testing 2
        time = 0;
        for (int proc = 0; proc < subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
            if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
                cuda::set(d_markedSendIndices, -1, numNodes);

                time += Gravity::Kernel::Launch::symbolicForce_test2(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                                     treeHandler->d_tree, particleHandler->d_particles,
                                                                     lowestDomainListHandler->d_domainList,
                                                                     d_markedSendIndices, diam,
                                                                     simulationParameters.theta,
                                                                     numParticles, numParticles,
                                                                     proc, relevantIndicesCounter, curveType);

                time += Gravity::Kernel::Launch::collectSendIndices(treeHandler->d_tree, particleHandler->d_particles,
                                                                    d_markedSendIndices,
                                                                    &d_particles2SendIndices[particlesOffset],
                                                                    &d_pseudoParticles2SendIndices[pseudoParticlesOffset],
                                                                    &d_pseudoParticles2SendLevels[pseudoParticlesOffset],
                                                                    &d_particles2SendCount[proc],
                                                                    &d_pseudoParticles2SendCount[proc],
                                                                    numParticles, numNodes, curveType);

                cuda::copy(&particlesOffsetBuffer, &d_particles2SendCount[proc], 1, To::host);
                cuda::copy(&pseudoParticlesOffsetBuffer, &d_pseudoParticles2SendCount[proc], 1, To::host);

                Logger(DEBUG) << "particles2SendCount[" << proc << "] = " << particlesOffsetBuffer;
                Logger(DEBUG) << "pseudoParticles2SendCount[" << proc << "] = " << pseudoParticlesOffsetBuffer;

                particlesOffset += particlesOffsetBuffer;
                pseudoParticlesOffset += pseudoParticlesOffsetBuffer;
            }
        }
        // end: symbolic force testing 2
    }
    else if (symbolicForceVersion == 3) {
        // symbolic force testing 3
        time = 0;
        real time_symbolic = 0.;
        real time_collect = 0.;
        for (int proc = 0; proc < subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
            if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
                cuda::set(d_markedSendIndices, -1, numNodes);

                time_symbolic += Gravity::Kernel::Launch::symbolicForce_test3(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                                     treeHandler->d_tree, particleHandler->d_particles,
                                                                     lowestDomainListHandler->d_domainList,
                                                                     d_markedSendIndices, diam,
                                                                     simulationParameters.theta,
                                                                     numParticles, numParticles,
                                                                     proc, relevantIndicesCounter, curveType);
                time += time_symbolic;
                time_collect += Gravity::Kernel::Launch::collectSendIndices(treeHandler->d_tree, particleHandler->d_particles,
                                                                    d_markedSendIndices,
                                                                    &d_particles2SendIndices[particlesOffset],
                                                                    &d_pseudoParticles2SendIndices[pseudoParticlesOffset],
                                                                    &d_pseudoParticles2SendLevels[pseudoParticlesOffset],
                                                                    &d_particles2SendCount[proc],
                                                                    &d_pseudoParticles2SendCount[proc],
                                                                    numParticles, numNodes, curveType);
                time += time_collect;

                cuda::copy(&particlesOffsetBuffer, &d_particles2SendCount[proc], 1, To::host);
                cuda::copy(&pseudoParticlesOffsetBuffer, &d_pseudoParticles2SendCount[proc], 1, To::host);

                Logger(DEBUG) << "particles2SendCount[" << proc << "] = " << particlesOffsetBuffer;
                Logger(DEBUG) << "pseudoParticles2SendCount[" << proc << "] = " << pseudoParticlesOffsetBuffer;

                particlesOffset += particlesOffsetBuffer;
                pseudoParticlesOffset += pseudoParticlesOffsetBuffer;
            }
        }
        Logger(TRACE) << "time symbolic: " << time_symbolic << " vs. collect: " << time_collect << " ms ...";
        // end: symbolic force testing 2
    }
    else if (symbolicForceVersion == 4) {

        time = 0;
        real time_symbolic = 0.;
        real time_collect = 0.;

        integer treeIndex;
        cuda::copy(&treeIndex, treeHandler->d_index, 1, To::host);

        cuda::set(d_markedSendIndices, 0, numNodes);

        time_symbolic += Gravity::Kernel::Launch::symbolicForce_test4(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                                      treeHandler->d_tree, particleHandler->d_particles,
                                                                      lowestDomainListHandler->d_domainList,
                                                                      d_markedSendIndices, diam,
                                                                      simulationParameters.theta,
                                                                      numParticles, numParticles,
                                                                      0, relevantIndicesCounter, curveType);

        time += time_symbolic;

        Logger(INFO) << "finished symbolic force ...";

        for (int proc = 0; proc < subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
            if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {

                time_collect += Gravity::Kernel::Launch::collectSendIndices_test4(treeHandler->d_tree, particleHandler->d_particles,
                                                                            d_markedSendIndices,
                                                                            &d_particles2SendIndices[particlesOffset],
                                                                            &d_pseudoParticles2SendIndices[pseudoParticlesOffset],
                                                                            &d_pseudoParticles2SendLevels[pseudoParticlesOffset],
                                                                            &d_particles2SendCount[proc],
                                                                            &d_pseudoParticles2SendCount[proc],
                                                                            numParticlesLocal, numParticles, treeIndex,
                                                                            proc, curveType);
                time += time_collect;

                cuda::copy(&particlesOffsetBuffer, &d_particles2SendCount[proc], 1, To::host);
                cuda::copy(&pseudoParticlesOffsetBuffer, &d_pseudoParticles2SendCount[proc], 1, To::host);

                Logger(DEBUG) << "particles2SendCount[" << proc << "] = " << particlesOffsetBuffer;
                Logger(DEBUG) << "pseudoParticles2SendCount[" << proc << "] = " << pseudoParticlesOffsetBuffer;

                particlesOffset += particlesOffsetBuffer;
                pseudoParticlesOffset += pseudoParticlesOffsetBuffer;
            }
        }
        Logger(TRACE) << "time symbolic: " << time_symbolic << " vs. collect: " << time_collect << " ms ...";
    }
    else {
        MPI_Finalize();
        Logger(ERROR) << "symbolicForceVersion: " << symbolicForceVersion << " not available!";
        exit(1);
    }


//#if DEBUGGING
//    Gravity::Kernel::Launch::testSendIndices(subDomainKeyTreeHandler->d_subDomainKeyTree, treeHandler->d_tree,
//                                             particleHandler->d_particles, d_pseudoParticles2SendIndices,
//                                             d_markedSendIndices,
//                                             d_pseudoParticles2SendLevels, curveType, pseudoParticlesOffset);
//#endif

    totalTime += time;
    Logger(TIME) << "symbolicForce: " << time << " ms";
    profiler.value2file(ProfilerIds::Time::Gravity::symbolicForce, time);

    Timer timer;

    cuda::copy(h_particles2SendCount, d_particles2SendCount, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses,
               To::host);
    cuda::copy(h_pseudoParticles2SendCount, d_pseudoParticles2SendCount,
               subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses, To::host);

    integer *particleSendLengths;
    particleSendLengths = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    particleSendLengths[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] = 0;
    integer *particleReceiveLengths;
    particleReceiveLengths = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    particleReceiveLengths[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] = 0;

    integer *pseudoParticleSendLengths;
    pseudoParticleSendLengths = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    pseudoParticleSendLengths[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] = 0;
    integer *pseudoParticleReceiveLengths;
    pseudoParticleReceiveLengths = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    pseudoParticleReceiveLengths[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] = 0;

    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        //Logger(INFO) << "h_particles2SendCount[" << proc << "] = " << h_particles2SendCount[proc];
        //Logger(INFO) << "h_pseudoParticles2SendCount[" << proc << "] = " << h_pseudoParticles2SendCount[proc];
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            particleSendLengths[proc] = h_particles2SendCount[proc];
            pseudoParticleSendLengths[proc] = h_pseudoParticles2SendCount[proc];
            Logger(INFO) << "particleSendLengths[" << proc << "] = " << particleSendLengths[proc];
            Logger(INFO) << "pseudoParticleSendLengths[" << proc << "] = " << pseudoParticleSendLengths[proc];
        }
    }

    profiler.vector2file(ProfilerIds::SendLengths::gravityParticles, particleSendLengths);
    profiler.vector2file(ProfilerIds::SendLengths::gravityPseudoParticles, pseudoParticleSendLengths);

    mpi::messageLengths(subDomainKeyTreeHandler->h_subDomainKeyTree, particleSendLengths, particleReceiveLengths);

    integer particleTotalReceiveLength = 0;
    integer particleTotalSendLength = 0;
    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            particleTotalReceiveLength += particleReceiveLengths[proc];
            particleTotalSendLength += particleSendLengths[proc];
        }
    }

    Logger(INFO) << "gravity: particleTotalReceiveLength: " << particleTotalReceiveLength;
    Logger(INFO) << "gravity: particleTotalSendLength: " << particleTotalSendLength;

    Logger(INFO) << "temporary numParticles: " << numParticlesLocal + particleTotalReceiveLength;

    mpi::messageLengths(subDomainKeyTreeHandler->h_subDomainKeyTree, pseudoParticleSendLengths,
                        pseudoParticleReceiveLengths);

    profiler.vector2file(ProfilerIds::ReceiveLengths::gravityParticles, particleReceiveLengths);
    profiler.vector2file(ProfilerIds::ReceiveLengths::gravityPseudoParticles, pseudoParticleReceiveLengths);

    integer pseudoParticleTotalReceiveLength = 0;
    integer pseudoParticleTotalSendLength = 0;
    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            pseudoParticleTotalReceiveLength += pseudoParticleReceiveLengths[proc];
            pseudoParticleTotalSendLength += pseudoParticleSendLengths[proc];
            //Logger(INFO) << "particleReceiveLengths[" << proc << "] = " << particleReceiveLengths[proc];
            //Logger(INFO) << "pseudoParticleReceiveLengths[" << proc << "] = " << pseudoParticleReceiveLengths[proc];
        }
    }

#if DEBUGGING
    Gravity::Kernel::Launch::testSendIndices(subDomainKeyTreeHandler->d_subDomainKeyTree, treeHandler->d_tree,
                                             particleHandler->d_particles, d_pseudoParticles2SendIndices,
                                             d_markedSendIndices,
                                             d_pseudoParticles2SendLevels, curveType, pseudoParticleTotalSendLength);
#endif

    // debug
    //particleHandler->copyDistribution(To::host, false, false, true);
    //int *h_pseudoParticlesSendIndices = new int[pseudoParticleTotalSendLength];
    //cuda::copy(h_pseudoParticlesSendIndices, d_pseudoParticles2SendIndices, pseudoParticleTotalSendLength, To::host);
    //int debug_offset = 0;
    //for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
    //    if (proc != proc < subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
    //        for (int i = 0; i < pseudoParticleSendLengths[proc]; i++) {
    //            for (int j = 0; j < pseudoParticleSendLengths[proc]; j++) {
    //                if (i != j) {
    //                    if (h_pseudoParticlesSendIndices[i + debug_offset] == h_pseudoParticlesSendIndices[j + debug_offset] ||
    //                            (particleHandler->h_x[h_pseudoParticlesSendIndices[i + debug_offset]] == particleHandler->h_x[h_pseudoParticlesSendIndices[j + debug_offset]] &&
    //                            particleHandler->h_y[h_pseudoParticlesSendIndices[i + debug_offset]] == particleHandler->h_y[h_pseudoParticlesSendIndices[j + debug_offset]])) {
    //                        Logger(INFO) << "found duplicate regarding proc " << proc << ": index: i: " << i << " = "
    //                                     << h_pseudoParticlesSendIndices[i + debug_offset]
    //                                     << ", j: " << j << " = " << h_pseudoParticlesSendIndices[j + debug_offset] <<
    //                                     " (" <<  particleHandler->h_x[h_pseudoParticlesSendIndices[i + debug_offset]]
    //                                     << ", " << particleHandler->h_x[h_pseudoParticlesSendIndices[j + debug_offset]] << ")";
    //                    }
    //                }
    //            }
    //        }
    //        debug_offset += pseudoParticleSendLengths[proc];
    //    }
    //}
    //delete [] h_pseudoParticlesSendIndices;
    // end: debug

    Logger(INFO) << "gravity: pseudoParticleTotalReceiveLength: " << pseudoParticleTotalReceiveLength;
    Logger(INFO) << "gravity: pseudoParticleTotalSendLength: " << pseudoParticleTotalSendLength;

    integer treeIndex;
    cuda::copy(&treeIndex, treeHandler->d_index, 1, To::host);

    // TODO: this is not working properly
    if (false/*simulationParameters.particlesSent2H5 && subStep == 500*/) {

        // WRITING PARTICLES TO SEND TO H5 FILE

        // writing particles
        int *h_particles2SendIndices = new int[particleTotalSendLength];
        cuda::copy(h_particles2SendIndices, d_particles2SendIndices, particleTotalSendLength, To::host);
        particles2file(std::string{"Gravity2SendParticles"}, h_particles2SendIndices, particleTotalSendLength);
        delete [] h_particles2SendIndices;
        // end: writing particles

        // writing pseudo-particles
        int *h_pseudoParticles2SendIndices = new int[pseudoParticleTotalSendLength];
        cuda::copy(h_pseudoParticles2SendIndices, d_pseudoParticles2SendIndices, pseudoParticleTotalSendLength, To::host);
        particles2file(std::string{"Gravity2SendPseudoParticles"}, h_pseudoParticles2SendIndices, pseudoParticleTotalSendLength);
        delete [] h_pseudoParticles2SendIndices;
        // end: writing pseudo-particles

        // writing both: particles and pseudo-particles
        int *h_sendIndices = new int[particleTotalSendLength + pseudoParticleTotalSendLength];
        cuda::copy(&h_sendIndices[0], d_particles2SendIndices, particleTotalSendLength, To::host);
        cuda::copy(&h_sendIndices[particleTotalSendLength], d_pseudoParticles2SendIndices, pseudoParticleTotalSendLength, To::host);
        particles2file(std::string{"Gravity2SendBoth"}, h_sendIndices, particleTotalSendLength + pseudoParticleTotalSendLength);
        delete [] h_sendIndices;
        // end: writing both: particles and pseudo-particles

        // END: WRITING PARTICLES TO SEND TO H5 FILE
    }

    //#if DEBUGGING
    //TreeNS::Kernel::Launch::info(treeHandler->d_tree, particleHandler->d_particles, treeIndex,
    //                                 treeIndex + pseudoParticleTotalReceiveLength);
    //TreeNS::Kernel::Launch::info(treeHandler->d_tree, particleHandler->d_particles, numParticlesLocal,
    //                                 numParticlesLocal + particleTotalReceiveLength);
    //#endif

    // x-entry pseudo-particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_pseudoParticles2SendIndices, particleHandler->d_x, d_collectedEntries,
                                             pseudoParticleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_x[treeIndex], pseudoParticleSendLengths,
                  pseudoParticleReceiveLengths);
    // x-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_x, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_x[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);

    //Particle velocity and acceleration:
    // vx-entry particle exchange
    //CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_vx, d_collectedEntries,
    //                                         particleTotalSendLength);
    //sendParticles(d_collectedEntries, &particleHandler->d_vx[numParticlesLocal], particleSendLengths,
    //              particleReceiveLengths);
    // ax-entry particle exchange
    //CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_ax, d_collectedEntries,
    //                                         particleTotalSendLength);
    //sendParticles(d_collectedEntries, &particleHandler->d_ax[numParticlesLocal], particleSendLengths,
    //              particleReceiveLengths);
#if DIM > 1
    // y-entry pseudo-particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_pseudoParticles2SendIndices, particleHandler->d_y, d_collectedEntries,
                                             pseudoParticleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_y[treeIndex], pseudoParticleSendLengths,
                  pseudoParticleReceiveLengths);
    // y-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_y, d_collectedEntries,
                                             particleTotalSendLength);

    sendParticles(d_collectedEntries, &particleHandler->d_y[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);

    //Particle velocity and acceleration:
    // vy-entry particle exchange
    //CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_vy, d_collectedEntries,
    //                                         particleTotalSendLength);
    //sendParticles(d_collectedEntries, &particleHandler->d_vy[numParticlesLocal], particleSendLengths,
    //              particleReceiveLengths);
    // ay-entry particle exchange
    //CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_ay, d_collectedEntries,
    //                                         particleTotalSendLength);
    //sendParticles(d_collectedEntries, &particleHandler->d_ay[numParticlesLocal], particleSendLengths,
    //              particleReceiveLengths);
#if DIM == 3
    // z-entry pseudo-particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_pseudoParticles2SendIndices, particleHandler->d_z, d_collectedEntries,
                                             pseudoParticleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_z[treeIndex], pseudoParticleSendLengths,
                  pseudoParticleReceiveLengths);
    // z-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_z, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_z[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);

    //Particle velocity and acceleration:
    // vz-entry particle exchange
    //CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_vz, d_collectedEntries,
    //                                         particleTotalSendLength);
    //sendParticles(d_collectedEntries, &particleHandler->d_vz[numParticlesLocal], particleSendLengths,
    //              particleReceiveLengths);
    // az-entry particle exchange
    //CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_az, d_collectedEntries,
    //                                         particleTotalSendLength);
    //sendParticles(d_collectedEntries, &particleHandler->d_az[numParticlesLocal], particleSendLengths,
    //              particleReceiveLengths);

#endif
#endif

    // mass-entry pseudo-particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_pseudoParticles2SendIndices, particleHandler->d_mass, d_collectedEntries,
                                             pseudoParticleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_mass[treeIndex], pseudoParticleSendLengths,
                  pseudoParticleReceiveLengths);
    // mass-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_mass, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_mass[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);

    // PSEUDO-PARTICLE level exchange
    sendParticles(d_pseudoParticles2SendLevels, d_pseudoParticles2ReceiveLevels, pseudoParticleSendLengths,
                  pseudoParticleReceiveLengths);


    time = timer.elapsed();
    totalTime += time;
    Logger(TIME) << "parallel_force(): sending particles: " << time << " ms";
    profiler.value2file(ProfilerIds::Time::Gravity::sending, time);

    //#if DEBUGGING
    //Logger(INFO) << "exchanged particle entry: x";
    //if (subDomainKeyTreeHandler->h_subDomainKeyTree->rank == 0) {
    //TreeNS::Kernel::Launch::info(treeHandler->d_tree, particleHandler->d_particles, treeIndex,
    //                             treeIndex + pseudoParticleTotalReceiveLength);
    //TreeNS::Kernel::Launch::info(treeHandler->d_tree, particleHandler->d_particles, numParticlesLocal,
    //                             numParticlesLocal + particleTotalReceiveLength);
    //}
    //#endif

    treeHandler->h_toDeleteLeaf[0] = numParticlesLocal;
    treeHandler->h_toDeleteLeaf[1] = numParticlesLocal + particleTotalReceiveLength;
    cuda::copy(treeHandler->h_toDeleteLeaf, treeHandler->d_toDeleteLeaf, 2, To::device);

    if (treeHandler->h_toDeleteLeaf[1] > numParticles) {
        MPI_Finalize();
        Logger(ERROR) << "numParticlesLocal + receiveLength = " << treeHandler->h_toDeleteLeaf[1] << " > " << " numParticles = " << numParticles;
        Logger(ERROR) << "Restart simulation with more memory! exiting ...";
        exit(1);
    }

    //int debugOffset = 0;
    //for (int proc=0; proc<subDomainKeyTreeHandler->h_numProcesses; proc++) {
    //    gpuErrorcheck(cudaMemset(buffer->d_integerVal, 0, sizeof(integer)));
    //    CudaUtils::Kernel::Launch::findDuplicateEntries(&particleHandler->d_x[treeIndex + debugOffset],
    //                                                    &particleHandler->d_y[treeIndex + debugOffset],
    //                                                    buffer->d_integerVal,
    //                                                    particleReceiveLengths[proc]);
    //    integer duplicates;
    //    gpuErrorcheck(cudaMemcpy(&duplicates, buffer->d_integerVal, sizeof(integer), cudaMemcpyDeviceToHost));
    //    Logger(INFO) << "duplicates: " << duplicates << " between: " << treeIndex + debugOffset << " and " << treeIndex + particleReceiveLengths[proc] + debugOffset;
    //
    //    debugOffset += particleReceiveLengths[proc];
    //}

    //#if DEBUGGING
    //gpuErrorcheck(cudaMemset(buffer->d_integerVal, 0, sizeof(integer)));
    //CudaUtils::Kernel::Launch::findDuplicateEntries(&particleHandler->d_x[treeHandler->h_toDeleteLeaf[0]],
    //                                                &particleHandler->d_y[treeHandler->h_toDeleteLeaf[0]],
    //                                                buffer->d_integerVal,
    //                                                particleTotalReceiveLength);
    //integer duplicates;
    //gpuErrorcheck(cudaMemcpy(&duplicates, buffer->d_integerVal, sizeof(integer), cudaMemcpyDeviceToHost));
    //Logger(INFO) << "duplicates: " << duplicates << " between: " << treeHandler->h_toDeleteLeaf[0] << " and " << treeHandler->h_toDeleteLeaf[0] + particleTotalReceiveLength;
    //#endif

    //#if DEBUGGING
    // debugging
    //gpuErrorcheck(cudaMemset(buffer->d_integerVal, 0, sizeof(integer)));
    //CudaUtils::Kernel::Launch::findDuplicateEntries(&particleHandler->d_x[0],
    //                                                &particleHandler->d_y[0],
    //                                                buffer->d_integerVal,
    //                                                numParticlesLocal + particleTotalReceiveLength);
    //integer duplicates;
    //gpuErrorcheck(cudaMemcpy(&duplicates, buffer->d_integerVal, sizeof(integer), cudaMemcpyDeviceToHost));
    //Logger(INFO) << "duplicates: " << duplicates << " between: " << 0 << " and " << numParticlesLocal + particleTotalReceiveLength;
    // end: debugging
    //#endif

    treeHandler->h_toDeleteNode[0] = treeIndex;
    treeHandler->h_toDeleteNode[1] = treeIndex + pseudoParticleTotalReceiveLength;

    if (treeHandler->h_toDeleteNode[1] > numNodes) {
        MPI_Finalize();
        Logger(ERROR) << "needed numNodes = " << treeHandler->h_toDeleteNode[1] << " > "
                           << " numNodes = " << numNodes;
        Logger(ERROR) << "Restart simulation with more memory! exiting ...";
        exit(1);
    }

    cuda::copy(treeHandler->h_toDeleteNode, treeHandler->d_toDeleteNode, 2, To::device);

#if DEBUGGING
    Logger(INFO) << "toDeleteLeaf: " << treeHandler->h_toDeleteLeaf[0] << " : " << treeHandler->h_toDeleteLeaf[1];
    Logger(INFO) << "toDeleteNode: " << treeHandler->h_toDeleteNode[0] << " : " << treeHandler->h_toDeleteNode[1];
#endif

    time = 0;
    // insert received pseudo-particles per level in order to ensure correct order (avoid race condition)
    for (int level=0; level<MAX_LEVEL; level++) {
        // -------------------------------------------------------------------------------------------------------------
        time += Gravity::Kernel::Launch::insertReceivedPseudoParticles(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                                       treeHandler->d_tree,
                                                                       particleHandler->d_particles,
                                                                       d_pseudoParticles2ReceiveLevels, level,
                                                                       numParticles, numParticles);
        // -------------------------------------------------------------------------------------------------------------
    }

    totalTime += time;
    Logger(TIME) << "parallel_gravity: inserting received pseudo-particles: " << time << " ms";
    profiler.value2file(ProfilerIds::Time::Gravity::insertReceivedPseudoParticles, time);

    time = 0;
    //if (treeHandler->h_toDeleteLeaf[0] < treeHandler->h_toDeleteLeaf[1]) {
    // ---------------------------------------------------------
    time += Gravity::Kernel::Launch::insertReceivedParticles(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                             treeHandler->d_tree, particleHandler->d_particles,
                                                             domainListHandler->d_domainList,
                                                             lowestDomainListHandler->d_domainList,
                                                             numParticles, numParticles);
    // ---------------------------------------------------------
    //}
    totalTime += time;
    Logger(TIME) << "parallel_gravity: inserting received particles: " << time << " ms";
    profiler.value2file(ProfilerIds::Time::Gravity::insertReceivedParticles, time);

    Logger(DEBUG) << "Finished inserting received particles!";

#if UNIT_TESTING
    //if (subStep == 10) {
        int actualTreeIndex;
        cuda::copy(&actualTreeIndex, treeHandler->d_index, 1, To::host);
        Logger(TRACE) << "[" << subStep << "] Checking Masses ...";
        UnitTesting::Kernel::Launch::test_localTree(treeHandler->d_tree, particleHandler->d_particles,
                                                    numParticles, actualTreeIndex); //numNodes);
    //}
#endif

    time = 0;

    //DomainListNS::Kernel::Launch::info(particleHandler->d_particles, domainListHandler->d_domainList);
    //TreeNS::Kernel::Launch::testTree(treeHandler->d_tree, particleHandler->d_particles, numParticlesLocal, numParticles);

    // SELECT compute forces version:
    // 0: similiar to burtscher with presorting according to the space-filling curves
    // 1: similiar to burtscher
    // 2: miluphcuda version with additional presorting according to the space-filling curves
    // 3: miluphcuda version
    // TODO: need to be verified (possible to use this additional shared memory?)
    // 4: miluphcuda version with additional presorting according to the space-filling curves and shared memory
    int computeForcesVersion = simulationParameters.gravityForceVersion;

    Logger(DEBUG) << "computeForcesVersion: " << computeForcesVersion;

    // preparations for computing forces
    treeHandler->copy(To::host);
    real x_radius = 0.5 * (*treeHandler->h_maxX - (*treeHandler->h_minX));
#if DIM > 1
    real y_radius = 0.5 *  (*treeHandler->h_maxY - (*treeHandler->h_minY));
#if DIM == 3
    real z_radius = 0.5 *  (*treeHandler->h_maxZ - (*treeHandler->h_minZ));
#endif
#endif

#if DIM == 1
    real radius = x_radius;
#elif DIM == 2
    real radius = std::max(x_radius, y_radius);
#else
    real radius_max = std::max(x_radius, y_radius);
    real radius = std::max(radius_max, z_radius);
#endif
    //radius *= 0.5; //0.5; // TODO: was 1.0
    Logger(INFO) << "radius: " << radius;

    // end: preparations for computing forces

    // needed for version 0 and 1
    int warp = 32;
    int stackSize = 128; //128; //64;
    int blockSize = 256;
    // end: needed for version 0 and 1
    if (computeForcesVersion == 0) {

        //int sortVersion = 0;
        //if (sortVersion == 0) {
        //int treeIndexBeforeSorting;
        //cuda::copy(&treeIndexBeforeSorting, treeHandler->d_index, 1, To::host);
        //Logger(INFO) << "treeIndexBeforeSorting: " << treeIndexBeforeSorting;
        //time = TreeNS::Kernel::Launch::sort(treeHandler->d_tree, numParticlesLocal, numParticles, true);
        //Logger(TIME) << "sorting: " << time << " ms";
        //}
        //else if (sortVersion == 1) {

        // THUS, instead:
        // presorting using keys...
        real timeSorting = 0.;
        timeSorting += TreeNS::Kernel::Launch::prepareSorting(treeHandler->d_tree, particleHandler->d_particles,
                                                              numParticlesLocal, numParticles);

        keyType *d_keys;
        //cuda::malloc(d_keys, numParticlesLocal);
        d_keys = buffer->d_keyTypeBuffer;
        timeSorting += SubDomainKeyTreeNS::Kernel::Launch::getParticleKeys(
                subDomainKeyTreeHandler->d_subDomainKeyTree,
                treeHandler->d_tree, particleHandler->d_particles,
                d_keys, 21, numParticlesLocal, curveType);

        timeSorting += HelperNS::sortArray(treeHandler->d_start, treeHandler->d_sorted, d_keys,
                                           buffer->d_keyTypeBuffer1, //helperHandler->d_keyTypeBuffer,
                                           numParticlesLocal);
        //cuda::free(d_keys);

        Logger(TIME) << "gravity: presorting: " << timeSorting << " ms";
        //end: presorting using keys...

        //}

        //actual (local) force
        // -------------------------------------------------------------------------------------------------------------
        time = Gravity::Kernel::Launch::computeForces_v2(treeHandler->d_tree, particleHandler->d_particles, radius,
                                                         numParticlesLocal, numParticles, blockSize, warp,
                                                         stackSize, subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                         simulationParameters.theta, simulationParameters.smoothing,
                                                         simulationParameters.calculateEnergy);
        // -------------------------------------------------------------------------------------------------------------

    }
    else if (computeForcesVersion == 1) {
        // -------------------------------------------------------------------------------------------------------------
        time = Gravity::Kernel::Launch::computeForces_v2_1(treeHandler->d_tree, particleHandler->d_particles,
                                                           numParticlesLocal, numParticles, blockSize, warp, stackSize,
                                                           subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                           simulationParameters.theta, simulationParameters.smoothing,
                                                           simulationParameters.calculateEnergy);
        // -------------------------------------------------------------------------------------------------------------
    }
    else if (computeForcesVersion == 2) {

        //int treeIndexBeforeSorting;
        //cuda::copy(&treeIndexBeforeSorting, treeHandler->d_index, 1, To::host);
        //Logger(INFO) << "treeIndexBeforeSorting: " << treeIndexBeforeSorting;
        //time = TreeNS::Kernel::Launch::sort(treeHandler->d_tree, numParticlesLocal, numParticles, true);
        //Logger(TIME) << "sorting: " << time << " ms";

        // presorting using keys...
        real timeSorting = 0.;
        timeSorting += TreeNS::Kernel::Launch::prepareSorting(treeHandler->d_tree, particleHandler->d_particles,
                                                              numParticlesLocal, numParticles);

        keyType *d_keys;
        //cuda::malloc(d_keys, numParticlesLocal + particleTotalReceiveLength);
        d_keys = buffer->d_keyTypeBuffer;
        timeSorting += SubDomainKeyTreeNS::Kernel::Launch::getParticleKeys(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                                           treeHandler->d_tree,
                                                                           particleHandler->d_particles,
                                                                           d_keys, 21,numParticlesLocal /*+
                                                                           particleTotalReceiveLength*/, curveType);

        timeSorting += HelperNS::sortArray(treeHandler->d_start, treeHandler->d_sorted, d_keys,
                                           buffer->d_keyTypeBuffer1, //helperHandler->d_keyTypeBuffer,
                                           numParticlesLocal); // + particleTotalReceiveLength);
        //cuda::free(d_keys);

        Logger(TIME) << "gravity: presorting: " << timeSorting << " ms";
        //end: presorting using keys...

        // -------------------------------------------------------------------------------------------------------------
        time = Gravity::Kernel::Launch::computeForces_v1(treeHandler->d_tree, particleHandler->d_particles,
                                                         radius, numParticles, numParticles,
                                                         subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                         simulationParameters.theta, simulationParameters.smoothing,
                                                         simulationParameters.calculateEnergy);
        // -------------------------------------------------------------------------------------------------------------
    }
    else if (computeForcesVersion == 3) {
        // -------------------------------------------------------------------------------------------------------------
        time = Gravity::Kernel::Launch::computeForces_v1_1(treeHandler->d_tree, particleHandler->d_particles,
                                                           radius, numParticles, numParticles,
                                                           subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                           simulationParameters.theta, simulationParameters.smoothing,
                                                           simulationParameters.calculateEnergy);
        // -------------------------------------------------------------------------------------------------------------
    }
    else if (computeForcesVersion == 4) {
        // presorting using keys...
        real timeSorting = 0.;
        timeSorting += TreeNS::Kernel::Launch::prepareSorting(treeHandler->d_tree, particleHandler->d_particles,
                                                              numParticlesLocal, numParticles);

        keyType *d_keys;
        //cuda::malloc(d_keys, numParticlesLocal + particleTotalReceiveLength);
        d_keys = buffer->d_keyTypeBuffer;
        timeSorting += SubDomainKeyTreeNS::Kernel::Launch::getParticleKeys(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                                           treeHandler->d_tree, particleHandler->d_particles,
                                                                           d_keys, 21, numParticlesLocal /*+
                                                                           particleTotalReceiveLength*/, curveType);

        timeSorting += HelperNS::sortArray(treeHandler->d_start, treeHandler->d_sorted, d_keys,
                                           buffer->d_keyTypeBuffer1, //helperHandler->d_keyTypeBuffer,
                                           numParticlesLocal); // + particleTotalReceiveLength);
        //cuda::free(d_keys);

        Logger(TIME) << "gravity: presorting: " << timeSorting << " ms";
        //end: presorting using keys...

        // -------------------------------------------------------------------------------------------------------------
        time = Gravity::Kernel::Launch::computeForces_v1_2(treeHandler->d_tree, particleHandler->d_particles,
                                                           radius, numParticles, numParticles,
                                                           subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                           simulationParameters.theta, simulationParameters.smoothing,
                                                           simulationParameters.calculateEnergy);
        // -------------------------------------------------------------------------------------------------------------
    }
    else {
        MPI_Finalize();
        Logger(ERROR) << "select proper compute forces version!";
        exit(0);
    }

    //NOTE: time(computeForces) < time(computeForceMiluphcuda) for kepler disk, but
    // time(computeForces) >> time(computeForceMiluphcuda) for plummer!!!

    totalTime += time;
    Logger(TIME) << "computeForces: " << time << " ms";
    profiler.value2file(ProfilerIds::Time::Gravity::force, time);

    // repairTree
    // necessary? Tree is build for every iteration: ONLY necessary if subsequent SPH?
    int debug_lowestDomainListIndex;
    cuda::copy(&debug_lowestDomainListIndex, lowestDomainListHandler->d_domainListIndex, 1, To::host);
    Logger(DEBUG) << "lowest Domain list index: " << debug_lowestDomainListIndex;

    integer toDeleteLeaf0;
    integer toDeleteLeaf1;
    cuda::copy(&toDeleteLeaf0, &treeHandler->d_toDeleteLeaf[0], 1, To::host);
    cuda::copy(&toDeleteLeaf1, &treeHandler->d_toDeleteLeaf[1], 1, To::host);
    Logger(INFO) << "toDeleteLeaf: " << toDeleteLeaf0 << ", " << toDeleteLeaf1;

    integer toDeleteNode0;
    integer toDeleteNode1;
    cuda::copy(&toDeleteNode0, &treeHandler->d_toDeleteNode[0], 1, To::host);
    cuda::copy(&toDeleteNode1, &treeHandler->d_toDeleteNode[1], 1, To::host);
    Logger(INFO) << "toDeleteNode: " << toDeleteNode0 << ", " << toDeleteNode1;

    Logger(DEBUG) << "repairing tree...";
    // -----------------------------------------------------------------------------------------------------------------
    time = SubDomainKeyTreeNS::Kernel::Launch::repairTree(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                          treeHandler->d_tree, particleHandler->d_particles,
                                                          domainListHandler->d_domainList,
                                                          lowestDomainListHandler->d_domainList,
                                                          numParticles, numNodes, curveType);

    // -----------------------------------------------------------------------------------------------------------------

    profiler.value2file(ProfilerIds::Time::Gravity::repairTree, time);

    //TreeNS::Kernel::Launch::info(treeHandler->d_tree, particleHandler->d_particles,
    // treeIndex, treeIndex + pseudoParticleTotalReceiveLength);
    //TreeNS::Kernel::Launch::info(treeHandler->d_tree, particleHandler->d_particles,
    // numParticlesLocal, numParticlesLocal + particleTotalReceiveLength);

    delete [] h_relevantDomainListProcess;
    delete [] h_particles2SendCount;
    delete [] h_pseudoParticles2SendCount;

    return totalTime;
}
#endif // TARGET_GPU

real Miluphpc::sph() {

    real time = 0.;
#if TARGET_GPU
    time = parallel_sph();
#else
    time = cpu_sph();
#endif // TARGET_GPU

    return time;
}

real Miluphpc::cpu_sph() {

    // TODO: CPU sph()

    return 0.;
}
#if TARGET_GPU
// IN PRINCIPLE it should be possible to reuse already sent particles from (parallel) gravity
real Miluphpc::parallel_sph() {

    real time;
    real totalTime = 0;

#if SPH_SIM

    Timer timer;
    real timeElapsed;

    integer *d_particles2SendIndices = buffer->d_integerBuffer1;
    cuda::set(d_particles2SendIndices, -1, numParticles);
    integer *d_particles2SendCount = buffer->d_sendCount;
    cuda::set(d_particles2SendCount, 0, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses);

    cuda::set(lowestDomainListHandler->d_domainListCounter, 0, 1);

    // -----------------------------------------------------------------------------------------------------------------
    time = SPH::Kernel::Launch::compTheta(subDomainKeyTreeHandler->d_subDomainKeyTree, treeHandler->d_tree,
                                          particleHandler->d_particles, lowestDomainListHandler->d_domainList,
                                          curveType);
    // -----------------------------------------------------------------------------------------------------------------

    totalTime += time;
    Logger(TIME) << "sph: compTheta: " << time << " ms";
    profiler.value2file(ProfilerIds::Time::SPH::compTheta, time);

    integer relevantIndicesCounter;
    cuda::copy(&relevantIndicesCounter, lowestDomainListHandler->d_domainListCounter, 1, To::host);
    //Logger(DEBUG) << "relevantIndicesCounter: " << relevantIndicesCounter;

    integer particlesOffset = 0;
    integer particlesOffsetBuffer;

    integer *h_relevantDomainListProcess;
    h_relevantDomainListProcess = new integer[relevantIndicesCounter];
    cuda::copy(h_relevantDomainListProcess, lowestDomainListHandler->d_relevantDomainListProcess,
               relevantIndicesCounter, To::host);

    integer *d_markedSendIndices = buffer->d_integerBuffer;
    real *d_collectedEntries = buffer->d_realBuffer;
    integer *h_particles2SendCount = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];

    timer.reset();
    // determine search radius
    boost::mpi::communicator comm;

    // DETERMINE search radius
    // [1] either: use max(sml):
    //const unsigned int blockSizeReduction = 256;
    //real *d_searchRadii;
    //cuda::malloc(d_searchRadii, blockSizeReduction);
    //cuda::set(d_searchRadii, (real)0., blockSizeReduction);
    //time += CudaUtils::Kernel::Launch::reduceBlockwise<real, blockSizeReduction>(particleHandler->d_sml,
    // d_searchRadii, numParticlesLocal);

    //real *d_intermediateResult;
    //cuda::malloc(d_intermediateResult, 1);
    //cuda::set(d_intermediateResult, (real)0., 1);
    //time += CudaUtils::Kernel::Launch::blockReduction<real, blockSizeReduction>(d_searchRadii, d_intermediateResult);
    //cuda::copy(&h_searchRadius, d_intermediateResult, 1, To::host);
    //h_searchRadius /= subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses;
    //all_reduce(comm, boost::mpi::inplace_t<real*>(&h_searchRadius), 1, std::plus<real>());
    //cuda::free(d_searchRadii);
    //cuda::free(d_intermediateResult);

    // [2] or: calculate search radii as sml - min(distance to other process) for all particles
    // real *d_intermediateResult;
    // //cuda::malloc(d_intermediateResult, 1);
    // d_intermediateResult = buffer->d_realVal1;
    // real *d_potentialSearchRadii;
    // //cuda::malloc(d_potentialSearchRadii, numParticlesLocal);
    // d_potentialSearchRadii = buffer->d_realBuffer1;
    // -----------------------------------------------------------------------------------------------------------------
    // time = SPH::Kernel::Launch::determineSearchRadii(subDomainKeyTreeHandler->d_subDomainKeyTree, treeHandler->d_tree,
    //                                           particleHandler->d_particles, domainListHandler->d_domainList,
    //                                           lowestDomainListHandler->d_domainList, d_potentialSearchRadii,
    //                                           numParticlesLocal, 0, curveType);
    // -----------------------------------------------------------------------------------------------------------------

    // Logger(TIME) << "sph: determineSearchRadii: " << time << " ms";
    // h_searchRadius = HelperNS::reduceAndGlobalize(d_potentialSearchRadii, d_intermediateResult,
    //                                              numParticlesLocal, Reduction::max);

    real *d_intermediateResult;
    d_intermediateResult = buffer->d_realVal1;
    h_searchRadius = HelperNS::reduceAndGlobalize(particleHandler->d_sml, d_intermediateResult,
                                                  numParticlesLocal, Reduction::max);

    h_searchRadius *= 2.; // *= 1.;
    //h_searchRadius = 2 * 0.080001;

    //cuda::free(d_potentialSearchRadii);

    //cuda::free(d_intermediateResult);

    Logger(INFO) << "search radius: " << h_searchRadius;
    // end: determine search radius
    timeElapsed = timer.elapsed();
    profiler.value2file(ProfilerIds::Time::SPH::determineSearchRadii, timeElapsed);

    time = 0;

    // original master version
    /*
    Logger(DEBUG) << "relevant indices counter: " << relevantIndicesCounter;
    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            cuda::set(d_markedSendIndices, -1, numParticles); //numParticlesLocal should be sufficient
            for (int relevantIndex = 0; relevantIndex < relevantIndicesCounter; relevantIndex++) {
                if (h_relevantDomainListProcess[relevantIndex] == proc) {
                    //Logger(TRACE) << "h_relevantDomainListProcess[" << relevantIndex << "] = "
                    //             << h_relevantDomainListProcess[relevantIndex];
                    // ---------------------------------------------------------
                    time += SPH::Kernel::Launch::symbolicForce(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                               treeHandler->d_tree, particleHandler->d_particles,
                                                               lowestDomainListHandler->d_domainList,
                                                               d_markedSendIndices, h_searchRadius, numParticlesLocal,
                                                               numParticles,
                                                               relevantIndex, curveType);
                    // ---------------------------------------------------------
                }
            }
            // ---------------------------------------------------------
            time += SPH::Kernel::Launch::collectSendIndices(treeHandler->d_tree, particleHandler->d_particles,
                                                            d_markedSendIndices, &d_particles2SendIndices[particlesOffset],
                                                            &d_particles2SendCount[proc], numParticles, numParticles, curveType);
            // ---------------------------------------------------------
            cuda::copy(&particlesOffsetBuffer, &d_particles2SendCount[proc], 1, To::host);
            Logger(INFO) << "particles2SendCount[" << proc << "] = " << particlesOffsetBuffer;
            particlesOffset += particlesOffsetBuffer;
        }
    } */
    // end: original master version

    Logger(INFO) << "relevantIndicesCounter: " << relevantIndicesCounter;

    // test version
    /*
    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            cuda::set(d_markedSendIndices, -1, numParticles); //numParticlesLocal should be sufficient
            time += SPH::Kernel::Launch::symbolicForce_test(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                            treeHandler->d_tree, particleHandler->d_particles,
                                                            lowestDomainListHandler->d_domainList,
                                                            d_markedSendIndices, h_searchRadius, numParticlesLocal,
                                                            numParticles, proc, relevantIndicesCounter,
                                                            0, curveType);

            // ---------------------------------------------------------------------------------------------------------
            time += SPH::Kernel::Launch::collectSendIndices(treeHandler->d_tree, particleHandler->d_particles,
                                                            d_markedSendIndices,
                                                            &d_particles2SendIndices[particlesOffset],
                                                            &d_particles2SendCount[proc], numParticles,
                                                            numParticles, curveType); // numParticlesLocal should be sufficient
            // ---------------------------------------------------------------------------------------------------------
            cuda::copy(&particlesOffsetBuffer, &d_particles2SendCount[proc], 1, To::host);

            Logger(INFO) << "particles2SendCount[" << proc << "] = " << particlesOffsetBuffer;

            particlesOffset += particlesOffsetBuffer;
        }
    }
    */
    // end: test version

    // test version 2

    cuda::set(d_markedSendIndices, 0, numParticles); //numParticlesLocal should be sufficient
    time += SPH::Kernel::Launch::symbolicForce_test2(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                     treeHandler->d_tree, particleHandler->d_particles,
                                                     lowestDomainListHandler->d_domainList,
                                                     d_markedSendIndices, h_searchRadius, numParticlesLocal,
                                                     numParticles, 0, relevantIndicesCounter,
                                                     curveType);

    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {

        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {

            // ---------------------------------------------------------------------------------------------------------
            time += SPH::Kernel::Launch::collectSendIndices_test2(treeHandler->d_tree, particleHandler->d_particles,
                                                            d_markedSendIndices,
                                                            &d_particles2SendIndices[particlesOffset],
                                                            &d_particles2SendCount[proc], numParticlesLocal,
                                                            numParticles, 0, proc, curveType); // numParticlesLocal should be sufficient
            // ---------------------------------------------------------------------------------------------------------
            cuda::copy(&particlesOffsetBuffer, &d_particles2SendCount[proc], 1, To::host);

            Logger(INFO) << "particles2SendCount[" << proc << "] = " << particlesOffsetBuffer;

            particlesOffset += particlesOffsetBuffer;
        }
    }
    // end: test version 2


    profiler.value2file(ProfilerIds::Time::SPH::symbolicForce, time);
    totalTime += time;
    Logger(TIME) << "sph: symbolicForce: " << time << " ms";

    cuda::copy(h_particles2SendCount, d_particles2SendCount, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses,
               To::host);

    timer.reset();

    integer *particleSendLengths;
    particleSendLengths = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    particleSendLengths[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] = 0;
    integer *particleReceiveLengths;
    particleReceiveLengths = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    particleReceiveLengths[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] = 0;


    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        //Logger(INFO) << "sph: h_particles2SendCount[" << proc << "] = " << h_particles2SendCount[proc];
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            particleSendLengths[proc] = h_particles2SendCount[proc];
        }
    }

    mpi::messageLengths(subDomainKeyTreeHandler->h_subDomainKeyTree, particleSendLengths, particleReceiveLengths);

    profiler.vector2file(ProfilerIds::SendLengths::sph, particleSendLengths);
    profiler.vector2file(ProfilerIds::ReceiveLengths::sph, particleReceiveLengths);

    integer particleTotalReceiveLength = 0;
    integer particleTotalSendLength = 0;
    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            particleTotalReceiveLength += particleReceiveLengths[proc];
            particleTotalSendLength += particleSendLengths[proc];
        }
    }

    if (simulationParameters.particlesSent2H5) {
        // writing particles to send to h5 file
        int *h_particles2SendIndices = new int[particleTotalSendLength];
        cuda::copy(h_particles2SendIndices, d_particles2SendIndices, particleTotalSendLength, To::host);
        std::string filename = "SPH2Send";
        particles2file(filename, h_particles2SendIndices, particleTotalSendLength);
        delete[] h_particles2SendIndices;
        // end: writing particles to send to h5 file
    }

    Logger(INFO) << "sph: particleTotalReceiveLength: " << particleTotalReceiveLength;
    Logger(INFO) << "sph: particleTotalSendLength: " << particleTotalSendLength;

    delete [] h_relevantDomainListProcess;
    delete [] h_particles2SendCount;
    //delete [] h_pseudoParticles2SendCount;

    treeHandler->h_toDeleteLeaf[0] = numParticlesLocal;
    treeHandler->h_toDeleteLeaf[1] = numParticlesLocal + particleTotalReceiveLength;
    cuda::copy(treeHandler->h_toDeleteLeaf, treeHandler->d_toDeleteLeaf, 2, To::device);

    if (treeHandler->h_toDeleteLeaf[1] > numParticles) {
        MPI_Finalize();
        Logger(ERROR) << "numParticlesLocal + receiveLength = " << treeHandler->h_toDeleteLeaf[1] << " > "
                            << " numParticles = " << numParticles;
        Logger(ERROR) << "Restart simulation with more memory! exiting ...";
        exit(1);
    }

    // x-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_x, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_x[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
    // x-vel-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_vx, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_vx[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
    // x-acc-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_ax, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_ax[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
#if DIM > 1
    // y-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_y, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_y[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
    // y-vel-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_vy, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_vy[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
    // y-acc-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_ay, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_ay[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
#if DIM == 3
    // z-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_z, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_z[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
    // z-vel-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_vz, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_vz[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
    // z-acc-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_az, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_az[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
#endif
#endif
    // mass-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_mass, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_mass[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);

    // sml-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_sml, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_sml[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);

    // rho-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_rho, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_rho[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);

    // pressure-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_p, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_p[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);

    // cs-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_cs, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_cs[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);

    // cs-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_e, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_e[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
    // necessary for all entries... (material id, ...) ?
    // should not be necessary to send all entries (since they will be resent)

    time = timer.elapsed();
    totalTime += time;
    Logger(TIME) << "sph: sending particles: " << time;
    profiler.value2file(ProfilerIds::Time::SPH::sending, time);

    //Logger(INFO) << "checking for nans before assigning particles...";
    //ParticlesNS::Kernel::Launch::check4nans(particleHandler->d_particles, numParticlesLocal);

    cuda::copy(&treeHandler->h_toDeleteNode[0], treeHandler->d_index, 1, To::host);

#if DEBUGGING
    // debug
    gpuErrorcheck(cudaMemset(buffer->d_integerVal, 0, sizeof(integer)));
    CudaUtils::Kernel::Launch::findDuplicateEntries(&particleHandler->d_x[0],
                                                    &particleHandler->d_y[0],
                                                    &particleHandler->d_z[0],
                                                    buffer->d_integerVal,
                                                    numParticlesLocal + particleTotalReceiveLength);
    integer duplicates;
    gpuErrorcheck(cudaMemcpy(&duplicates, buffer->d_integerVal, sizeof(integer), cudaMemcpyDeviceToHost));
    Logger(DEBUG) << "duplicates: " << duplicates << " between: " << 0 << " and "
                    << numParticlesLocal + particleTotalReceiveLength;
    if (duplicates > 0) {
        MPI_Finalize();
        exit(0);
    }
    //end: debug
#endif

    //TreeNS::Kernel::Launch::info(treeHandler->d_tree, particleHandler->d_particles, numParticles, numNodes);
    //DomainListNS::Kernel::Launch::info(particleHandler->d_particles, domainListHandler->d_domainList);

    //if (treeHandler->h_toDeleteLeaf[1] > treeHandler->h_toDeleteLeaf[0]) {
    // -----------------------------------------------------------------------------------------------------------------
    time = SPH::Kernel::Launch::insertReceivedParticles(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                        treeHandler->d_tree,
                                                        particleHandler->d_particles, domainListHandler->d_domainList,
                                                        lowestDomainListHandler->d_domainList,
                                                        numParticles, numParticles);
    // -----------------------------------------------------------------------------------------------------------------
    //}
    profiler.value2file(ProfilerIds::Time::SPH::insertReceivedParticles, time);

    time = 0;
    //for (int level=MAX_LEVEL; level>0; --level) {
    //    time += SPH::Kernel::Launch::calculateCentersOfMass(treeHandler->d_tree, particleHandler->d_particles, level);
    //}
    Logger(TIME) << "sph: calculate centers of mass: " << time << " ms";

    totalTime += time;
    Logger(TIME) << "sph: inserting received particles: " << time << " ms";

    cuda::copy(&treeHandler->h_toDeleteNode[1], treeHandler->d_index, 1, To::host);
    cuda::copy(treeHandler->h_toDeleteNode, treeHandler->d_toDeleteNode, 2, To::device);

    Logger(DEBUG) << "treeHandler->h_toDeleteNode[0]: " << treeHandler->h_toDeleteNode[0];
    Logger(DEBUG) << "treeHandler->h_toDeleteNode[1]: " << treeHandler->h_toDeleteNode[1];

    if (treeHandler->h_toDeleteNode[1] > numNodes) {
        MPI_Finalize();
        Logger(ERROR) << "needed numNodes = " << treeHandler->h_toDeleteNode[1] << " > " << " numNodes = " << numNodes;
        Logger(ERROR) << "Restart simulation with more memory! exiting ...";
        exit(1);
    }

    treeHandler->copy(To::host);

#if CUBIC_DOMAINS
    real diam = std::abs(*treeHandler->h_maxX) + std::abs(*treeHandler->h_minX);
    //Logger(INFO) << "diam: " << diam;
#else // !CUBIC DOMAINS
    real diam_x = std::abs(*treeHandler->h_maxX) + std::abs(*treeHandler->h_minX);
#if DIM > 1
    real diam_y = std::abs(*treeHandler->h_maxY) + std::abs(*treeHandler->h_minY);
#if DIM == 3
    real diam_z = std::abs(*treeHandler->h_maxZ) + std::abs(*treeHandler->h_minZ);
#endif
#endif
#if DIM == 1
    real diam = diam_x;
    //Logger(INFO) << "diam: " << diam << "  (x = " << diam_x << ")";
#elif DIM == 2
    real diam = std::max({diam_x, diam_y});
    //Logger(INFO) << "diam: " << diam << "  (x = " << diam_x << ", y = " << diam_y << ")";
#else
    real diam = std::max({diam_x, diam_y, diam_z});
    //Logger(INFO) << "diam: " << diam << "  (x = " << diam_x << ", y = " << diam_y << ", z = " << diam_z << ")";
#endif
#endif

    time = 0.;
#if VARIABLE_SML
    // -----------------------------------------------------------------------------------------------------------------
    time += SPH::Kernel::Launch::fixedRadiusNN_variableSML(materialHandler->d_materials, treeHandler->d_tree,
                                                           particleHandler->d_particles, particleHandler->d_nnl,
                                                           numParticlesLocal, numParticles, numNodes);
    // -----------------------------------------------------------------------------------------------------------------
    // TODO: for variable SML
    // überprüfen inwiefern sich die sml geändert hat, sml_new <= sml_global_search // if sml_new > sml_global_search
#endif

    //Logger(TRACE) << "diam: " << diam; // << "  (x = " << diam_x << ", y = " << diam_y << ", z = " << diam_z << ")";

    //real timeSorting = 0.;
    //timeSorting += TreeNS::Kernel::Launch::prepareSorting(treeHandler->d_tree, particleHandler->d_particles, numParticlesLocal, numParticles);
    //keyType *d_keys;
    //cuda::malloc(d_keys, numParticlesLocal);
    //timeSorting += SubDomainKeyTreeNS::Kernel::Launch::getParticleKeys(subDomainKeyTreeHandler->d_subDomainKeyTree,
    //                                                    treeHandler->d_tree, particleHandler->d_particles,
    //                                                    d_keys, 21, numParticlesLocal, curveType);
    //timeSorting += HelperNS::sortArray(treeHandler->d_start, treeHandler->d_sorted, d_keys,
    //                                   helperHandler->d_keyTypeBuffer, numParticlesLocal);

    //int *h_sorted = new int[numParticlesLocal];
    //cuda::copy(h_sorted, treeHandler->d_sorted, numParticlesLocal, To::host);
    //for (int i=0; i<100; i++) {
    //    Logger(INFO) << "i: " << i << " sorted: " << h_sorted[i];
    //}
    //delete [] h_sorted;
    //cuda::free(d_keys);
    //Logger(INFO) << "sph: presorting: " << timeSorting << " ms";

    /*real timeSorting = 0.;
    timeSorting += TreeNS::Kernel::Launch::prepareSorting(treeHandler->d_tree, particleHandler->d_particles,
                                                          numParticlesLocal, numParticles);

    keyType *d_keys;
    cuda::malloc(d_keys, numParticlesLocal);
    timeSorting += SubDomainKeyTreeNS::Kernel::Launch::getParticleKeys(
            subDomainKeyTreeHandler->d_subDomainKeyTree,
            treeHandler->d_tree, particleHandler->d_particles,
            d_keys, 21, numParticlesLocal, curveType);

    timeSorting += HelperNS::sortArray(treeHandler->d_start, treeHandler->d_sorted, d_keys,
                                       helperHandler->d_keyTypeBuffer,
                                       numParticlesLocal);
    cuda::free(d_keys);

    Logger(TIME) << "sph: presorting: " << timeSorting << " ms";*/

    int fixedRadiusNN_version = simulationParameters.sphFixedRadiusNNVersion;
    if (fixedRadiusNN_version == 0) {
        // -------------------------------------------------------------------------------------------------------------
        time += SPH::Kernel::Launch::fixedRadiusNN(treeHandler->d_tree, particleHandler->d_particles,
                                                   particleHandler->d_nnl,
                                                   diam, numParticlesLocal, numParticles, numNodes);
        // -------------------------------------------------------------------------------------------------------------
    }
    else if (fixedRadiusNN_version == 1) {
        // ATTENTION: brute-force method
        time += SPH::Kernel::Launch::fixedRadiusNN_bruteForce(treeHandler->d_tree, particleHandler->d_particles,
                                                              particleHandler->d_nnl, numParticlesLocal, numParticles,
                                                              numNodes);
    }
    else if (fixedRadiusNN_version == 2) {
        // using shared memory (not beneficial)
        time += SPH::Kernel::Launch::fixedRadiusNN_sharedMemory(treeHandler->d_tree, particleHandler->d_particles,
                                                                particleHandler->d_nnl, numParticlesLocal, numParticles,
                                                                numNodes);
    }
    else if (fixedRadiusNN_version == 3) {
        // test version
        time = SPH::Kernel::Launch::fixedRadiusNN_withinBox(treeHandler->d_tree, particleHandler->d_particles,
                                                            particleHandler->d_nnl, numParticlesLocal, numParticles,
                                                            numNodes);
    }
    else {
        Logger(ERROR) << "fixedRadiusNN version not available! Exiting ...";
        MPI_Finalize();
        exit(1);
    }

    profiler.value2file(ProfilerIds::Time::SPH::fixedRadiusNN, time);
    totalTime += time;
    Logger(TIME) << "sph: fixedRadiusNN: " << time << " ms";

    // TODO: investigate presorting/cache efficiency for:
    //  * calculateDensity
    //  * calculateSoundSpeed
    //  * calculatePressure
    //  * internalForces

    Logger(DEBUG) << "calculate density";
    // -----------------------------------------------------------------------------------------------------------------
    time = SPH::Kernel::Launch::calculateDensity(kernelHandler.kernel, treeHandler->d_tree,
                                                 particleHandler->d_particles, particleHandler->d_nnl,
                                                 numParticlesLocal); //treeHandler->h_toDeleteLeaf[1]);
    // -----------------------------------------------------------------------------------------------------------------
    Logger(TIME) << "sph: calculateDensity: " << time << " ms";
    profiler.value2file(ProfilerIds::Time::SPH::density, time);

    Logger(DEBUG) << "calculate sound speed";
    // -----------------------------------------------------------------------------------------------------------------
    time = SPH::Kernel::Launch::calculateSoundSpeed(particleHandler->d_particles, materialHandler->d_materials,
                                                    numParticlesLocal); // treeHandler->h_toDeleteLeaf[1]);
    // -----------------------------------------------------------------------------------------------------------------
    Logger(TIME) << "sph: calculateSoundSpeed: " << time << " ms";
    profiler.value2file(ProfilerIds::Time::SPH::soundSpeed, time);

#if BALSARA_SWITCH
    time = SPH::Kernel::Launch::CalcDivvandCurlv(kernelHandler.kernel, particleHandler->d_particles, particleHandler->d_nnl,
                                                 numParticlesLocal);
#endif

    Logger(DEBUG) << "calculate pressure";
    // -----------------------------------------------------------------------------------------------------------------
    time = SPH::Kernel::Launch::calculatePressure(materialHandler->d_materials, particleHandler->d_particles,
                                                  numParticlesLocal); // treeHandler->h_toDeleteLeaf[1]);
    // ---------------------------------------------------------
    Logger(TIME) << "sph: calculatePressure: " << time << " ms";
    profiler.value2file(ProfilerIds::Time::SPH::pressure, time);

    timer.reset();
    //Timer timerTemp;
    // updating necessary particle entries

    // not really necessary but beneficial for timing
    comm.barrier();

    // sml-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_sml, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_sml[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);

    // pressure-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_p, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_p[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);

    // rho-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_rho, d_collectedEntries,
                                                                      particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_rho[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);

    // cs-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_cs, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_cs[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
    // end: updating necessary particle entries


    time = timer.elapsed();
    Logger(TIME) << "sph: sending particles (again): " << time;
    //Logger(TIME) << "sph: sending particles (again) collectingValues: " << sendingParticlesAgain;
    profiler.value2file(ProfilerIds::Time::SPH::resend, time);

    totalTime += time;
    Logger(DEBUG) << "internal forces";

    time = SPH::Kernel::Launch::internalForces(kernelHandler.kernel, materialHandler->d_materials, treeHandler->d_tree,
                                        particleHandler->d_particles, particleHandler->d_nnl, numParticlesLocal);
    Logger(TIME) << "sph: internalForces: " << time << " ms";
    profiler.value2file(ProfilerIds::Time::SPH::internalForces, time);

    time = SubDomainKeyTreeNS::Kernel::Launch::repairTree(subDomainKeyTreeHandler->d_subDomainKeyTree, treeHandler->d_tree,
                                                          particleHandler->d_particles, domainListHandler->d_domainList,
                                                          lowestDomainListHandler->d_domainList,
                                                          numParticles, numNodes, curveType);

    Logger(TIME) << "sph: totalTime: " << totalTime << " ms";
    profiler.value2file(ProfilerIds::Time::SPH::repairTree, time);

#endif // SPH_SIM

    return totalTime;

}
#endif // TARGET_GPU

void Miluphpc::fixedLoadBalancing() {

    Logger(INFO) << "fixedLoadBalancing()";

    keyType rangePerProc = (1UL << (21 * DIM))/(subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses);
    Logger(INFO) << "rangePerProc: " << rangePerProc;
    for (int i=0; i<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; i++) {
        subDomainKeyTreeHandler->h_subDomainKeyTree->range[i] = (keyType)i * rangePerProc;
    }
    subDomainKeyTreeHandler->h_subDomainKeyTree->range[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses] = KEY_MAX;

    Logger(INFO) << "keyMax: " << (keyType)KEY_MAX;

    //subDomainKeyTreeHandler->h_subDomainKeyTree->range[0] = 0UL;
    //subDomainKeyTreeHandler->h_subDomainKeyTree->range[1] = 0UL;
    //subDomainKeyTreeHandler->h_subDomainKeyTree->range[1] = (2UL << 60) + (4UL << 57); // + (4UL << 54) + (2UL << 51) + (1UL << 39);
    // 3|4|5|0|6|4|5|1|2|0|4|7|5|6|2|3|0|2|2|4|6|
    subDomainKeyTreeHandler->h_subDomainKeyTree->range[1] = (4UL << 60) + (1UL << 57); // + (2UL << 54);
    //        + (4UL << 57); // + (5UL << 54) + (0UL << 51) + (6UL << 48)
            //+ (4UL << 45) + (5UL << 42) + (1UL << 39) + (2UL << 36)
            //+ (0UL << 33) + (4UL << 30) + (7UL << 27) + (5UL << 24)
            //+ (6UL << 21) + (2UL << 18) + (3UL << 15) + (0UL << 12)
            //+ (2UL << 9) + (2UL << 6) + (4UL << 3) + (6UL << 0);
    //subDomainKeyTreeHandler->h_subDomainKeyTree->range[1] = (4UL << 60) + (4UL << 57) + (3UL << 54) + (2UL << 51) + (3UL << 30);
    //subDomainKeyTreeHandler->h_subDomainKeyTree->range[2] = (4UL << 60) + (2UL << 57);
    //subDomainKeyTreeHandler->h_subDomainKeyTree->range[3] = (6UL << 60) + (1UL << 57);
    //subDomainKeyTreeHandler->h_subDomainKeyTree->range[4] = KEY_MAX;
    // FOR TESTING PURPOSES:
    // 1, 0, 0, 0, ...: 1152921504606846976
    // 2, 0, 0, 0, ...: 2305843009213693952;
    // 3, 0, 0, 0, ...: 3458764513820540928;
    //subDomainKeyTreeHandler->h_subDomainKeyTree->range[0] = 0UL;
    //subDomainKeyTreeHandler->h_subDomainKeyTree->range[1] = 1048576; //2199023255552;//4194304; //  + (4UL << 57); // + (3UL << 54) + (1UL << 42) + (2UL << 18) + (1UL << 3) + (4);
    // // |4 (60)|0 (57)|0 (54)|4 (51)|0 (48)|6 (45)|1 (42)|1 (39)|1 (36)|5 (33)|6 (30)|4 (27)|5 (24)|7 (21)|0 (18)|6 (15)|5 (12)|1 (9)|1 (6)|4 (3)|3 (0)|
    // //subDomainKeyTreeHandler->h_subDomainKeyTree->range[1] = (4UL << 60) + (4UL << 51) + (6UL << 45) + (1UL << 42) + (1UL << 39) + (1UL << 36) + (5UL << 33)
    // //        + (6UL << 30) + (4UL << 27) + (5UL << 24) + (7UL << 21) + (6UL << 15) + (5UL << 12) + (1UL << 9) + (1UL << 6) + (4UL << 3);
    //subDomainKeyTreeHandler->h_subDomainKeyTree->range[2] = KEY_MAX; //9223372036854775808;

    for (int i=0; i<=subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; i++) {
        //printf("range[%i] = %lu\n", i, subDomainKeyTreeHandler->h_subDomainKeyTree->range[i]);
        Logger(TRACE) << "initialized range[" << i << "] = " << subDomainKeyTreeHandler->h_subDomainKeyTree->range[i];
    }
#if TARGET_GPU
    subDomainKeyTreeHandler->copy(To::device, true, true);
#endif

}

void Miluphpc::dynamicLoadBalancing(int bins) {

    boost::mpi::communicator comm;

    Logger(INFO) << "dynamicLoadBalancing()";

    int *processParticleCounts = new int[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];

    all_gather(comm, &numParticlesLocal, 1, processParticleCounts);

    int totalAmountOfParticles = 0;
    for (int i=0; i<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; i++) {
        //Logger(INFO) << "numParticles on process: " << i << " = " << processParticleCounts[i];
        totalAmountOfParticles += processParticleCounts[i];
    }

    int aimedParticlesPerProcess = totalAmountOfParticles/subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses;
//#if DEBUGGING
    Logger(INFO) << "aimedParticlesPerProcess = " << aimedParticlesPerProcess;
//#endif

    //updateRangeApproximately(aimedParticlesPerProcess, bins);
#if TARGET_GPU
    updateRange(aimedParticlesPerProcess);
#else
    cpu_updateRange(aimedParticlesPerProcess);
#endif // TARGET_GPU

    delete [] processParticleCounts;
}

void Miluphpc::cpu_updateRange(int aimedParticlesPerProcess) {
    // TODO: CPU updateRange()
}

#if TARGET_GPU
void Miluphpc::updateRange(int aimedParticlesPerProcess) {

    // update bounding boxes here! // TODO: if bounding boxes are calculated here, they shouldn't be calculated again !?
    boundingBox();

    boost::mpi::communicator comm;

    sumParticles = numParticlesLocal;
    all_reduce(comm, boost::mpi::inplace_t<integer*>(&sumParticles), 1, std::plus<integer>());

    keyType *d_localKeys;
    //cuda::malloc(d_localKeys, numParticlesLocal);
    d_localKeys = buffer->d_keyTypeBuffer;
    //keyType *h_localKeys = new keyType[sumParticles];

    SubDomainKeyTreeNS::Kernel::Launch::getParticleKeys(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                        treeHandler->d_tree, particleHandler->d_particles,
                                                        d_localKeys, 21, numParticlesLocal, curveType);

    int *numParticlesPerProcess = new int[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    all_gather(comm, numParticlesLocal, numParticlesPerProcess);
    std::vector<int> sizes;
    for (int i = 0; i < subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; i++) {
        sizes.push_back(numParticlesPerProcess[i]);
    }

    keyType *d_keys;
    //cuda::malloc(d_keys, sumParticles);
    d_keys = buffer->d_keyTypeBuffer1;

    //for (int i = 0; i < subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; i++) {
    //    Logger(INFO) << "numParticlesPerProcess[" << i << "] = " << numParticlesPerProcess[i]
    //    << " (numParticlesLocal: " << sumParticles << ")";
    //}

    all_gatherv(comm, d_localKeys, d_keys, sizes);

    keyType *d_sortedKeys;
    //cuda::malloc(d_sortedKeys, sumParticles);
    d_sortedKeys = buffer->d_keyTypeBuffer2;

    keyType *h_sortedKeys = new keyType[sumParticles];
    cuda::copy(h_sortedKeys, d_keys, sumParticles, To::host);

    //for (int i=aimedParticlesPerProcess - 10; i<aimedParticlesPerProcess + 10; i++) {
    //    Logger(INFO) << "sortedKeys[" << i << "] = " << h_sortedKeys[i] << "(" << h_sortedKeys[0]
    //    << ", " << h_sortedKeys[sumParticles - 1] << ")";
    //}

    HelperNS::sortKeys(d_keys, d_sortedKeys, sumParticles);

    keyType newRange;
    for (int proc=1; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        cuda::copy(&subDomainKeyTreeHandler->h_subDomainKeyTree->range[proc],
                   &d_sortedKeys[proc * aimedParticlesPerProcess], 1, To::host);
        Logger(INFO) << "new range: " << newRange;
    }

    for (int proc=1; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        subDomainKeyTreeHandler->h_subDomainKeyTree->range[proc] = (subDomainKeyTreeHandler->h_subDomainKeyTree->range[proc] >> 30) << 30;
    }

    subDomainKeyTreeHandler->copy(To::device);

    getMemoryInfo();

    //cuda::free(d_keys);
    //cuda::free(d_sortedKeys);
    //cuda::free(d_localKeys);

    delete [] numParticlesPerProcess;
    delete [] h_sortedKeys;

}
#endif // TARGET_GPU

#if TARGET_GPU
// deprecated
void Miluphpc::updateRangeApproximately(int aimedParticlesPerProcess, int bins) {

    // introduce "bin size" regarding keys
    //  keyHistRanges = [0, 1 * binSize, 2 * binSize, ... ]
    // calculate key of particles on the fly and assign to keyHistRanges
    //  keyHistNumbers = [1023, 50032, ...]
    // take corresponding keyHistRange as new range if (sum(keyHistRange[i]) > aimNumberOfParticles ...
    // communicate new ranges among processes

    boost::mpi::communicator comm;

    buffer->reset();

    SubDomainKeyTreeNS::Kernel::Launch::createKeyHistRanges(buffer->d_helper, bins);

    SubDomainKeyTreeNS::Kernel::Launch::keyHistCounter(treeHandler->d_tree, particleHandler->d_particles,
                                            subDomainKeyTreeHandler->d_subDomainKeyTree, buffer->d_helper,
                                            bins, numParticlesLocal, curveType);

    all_reduce(comm, boost::mpi::inplace_t<integer*>(buffer->d_integerBuffer), bins - 1, std::plus<integer>());

    // ------------------------------------------------------------------------------
    // //if (subDomainKeyTreeHandler->h_subDomainKeyTree->rank == 0) {
    //    int *histogram = new int[bins];
    //    cuda::copy(histogram, helperHandler->d_integerBuffer, bins, To::host);
    //
    //    HighFive::File h5file(simulationParameters.logDirectory + "histogram.h5",
    //                          HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate,
    //                          HighFive::MPIOFileDriver(MPI_COMM_WORLD, MPI_INFO_NULL));
    //    Logger(INFO) << "histogram bins: " << bins;
    //    for (int i = 0; i < bins; i++) {
    //        Logger(INFO) << "histogram[" << i << "]: " << histogram[i];
    //    }
    //    HighFive::DataSet h5_ranges = h5file.createDataSet<keyType>("/bins", HighFive::DataSpace(bins));
    //    h5_ranges.write(histogram);
    //    // h5file.close();
    //    delete[] histogram;
    // //}
    // ------------------------------------------------------------------------------

    SubDomainKeyTreeNS::Kernel::Launch::calculateNewRange(subDomainKeyTreeHandler->d_subDomainKeyTree, buffer->d_helper,
                                               bins, aimedParticlesPerProcess, curveType);
    keyType keyMax = (keyType)KEY_MAX;
    cuda::set(&subDomainKeyTreeHandler->d_range[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses], keyMax, 1);
    subDomainKeyTreeHandler->copy(To::host, true, true);

    //Logger(INFO) << "numProcesses: " << subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses;
    for (int i=0; i<=subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; i++) {
        Logger(INFO) << "range[" << i << "] = " << subDomainKeyTreeHandler->h_subDomainKeyTree->range[i];
    }
}
#endif

real Miluphpc::removeParticles() {

#if TARGET_GPU
    int *d_particles2remove = buffer->d_integerBuffer; //d_particles2removeBuffer; //&buffer->d_integerBuffer[0];
    int *d_particles2remove_counter = buffer->d_integerVal; //d_particles2removeVal; //buffer->d_integerVal;

    real *d_temp = buffer->d_realBuffer;
    integer *d_tempInt = buffer->d_integerBuffer1;
    //cuda::malloc(d_tempInt, numParticles);

    cuda::set(d_particles2remove_counter, 0, 1);

    auto time = ParticlesNS::Kernel::Launch::mark2remove(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                         treeHandler->d_tree, particleHandler->d_particles,
                                                         d_particles2remove, d_particles2remove_counter,
                                                         simulationParameters.removeParticlesCriterion,
                                                         simulationParameters.removeParticlesDimension,
                                                         numParticlesLocal);

    int h_particles2remove_counter;
    cuda::copy(&h_particles2remove_counter, d_particles2remove_counter, 1, To::host);
    Logger(INFO) << "#particles to be removed: " << h_particles2remove_counter;

    time += HelperNS::sortArray(particleHandler->d_x, d_temp, d_particles2remove, buffer->d_integerBuffer2,
                                numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_x, d_temp, numParticlesLocal);
    time += HelperNS::sortArray(particleHandler->d_vx, d_temp, d_particles2remove, buffer->d_integerBuffer2,
                                numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_vx, d_temp, numParticlesLocal);
    time += HelperNS::sortArray(particleHandler->d_ax, d_temp, d_particles2remove, buffer->d_integerBuffer2,
                                numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_ax, d_temp, numParticlesLocal);
#if DIM > 1
    time += HelperNS::sortArray(particleHandler->d_y, d_temp, d_particles2remove, buffer->d_integerBuffer2,
                                numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_y, d_temp, numParticlesLocal);
    time += HelperNS::sortArray(particleHandler->d_vy, d_temp, d_particles2remove, buffer->d_integerBuffer2,
                                numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_vy, d_temp, numParticlesLocal);
    time += HelperNS::sortArray(particleHandler->d_ay, d_temp, d_particles2remove, buffer->d_integerBuffer2,
                                numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_ay, d_temp, numParticlesLocal);
#if DIM == 3
    time += HelperNS::sortArray(particleHandler->d_z, d_temp, d_particles2remove, buffer->d_integerBuffer2,
                                numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_z, d_temp, numParticlesLocal);
    time += HelperNS::sortArray(particleHandler->d_vz, d_temp, d_particles2remove, buffer->d_integerBuffer2,
                                numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_vz, d_temp, numParticlesLocal);
    time += HelperNS::sortArray(particleHandler->d_az, d_temp, d_particles2remove, buffer->d_integerBuffer2,
                                numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_az, d_temp, numParticlesLocal);
#endif
#endif
    time += HelperNS::sortArray(particleHandler->d_mass, d_temp, d_particles2remove, buffer->d_integerBuffer2,
                                numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_mass, d_temp, numParticlesLocal);
    time += HelperNS::sortArray(particleHandler->d_uid, d_tempInt, d_particles2remove, buffer->d_integerBuffer2,
                                numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_uid, d_tempInt, numParticlesLocal);
#if SPH_SIM
    time += HelperNS::sortArray(particleHandler->d_materialId, d_tempInt, d_particles2remove, buffer->d_integerBuffer2,
                                numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_materialId, d_tempInt, numParticlesLocal);
    time += HelperNS::sortArray(particleHandler->d_sml, d_temp, d_particles2remove, buffer->d_integerBuffer2,
                                numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_sml, d_temp, numParticlesLocal);
    time += HelperNS::sortArray(particleHandler->d_rho, d_temp, d_particles2remove, buffer->d_integerBuffer2,
                                numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_rho, d_temp, numParticlesLocal);
    time += HelperNS::sortArray(particleHandler->d_p, d_temp, d_particles2remove, buffer->d_integerBuffer2,
                                numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_p, d_temp, numParticlesLocal);
    time += HelperNS::sortArray(particleHandler->d_e, d_temp, d_particles2remove, buffer->d_integerBuffer2,
                                numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_e, d_temp, numParticlesLocal);
    time += HelperNS::sortArray(particleHandler->d_cs, d_temp, d_particles2remove, buffer->d_integerBuffer2,
                                numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_cs, d_temp, numParticlesLocal);
#endif


    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_x[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_vx[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_ax[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
#if DIM > 1
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_y[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_vy[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_ay[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
#if DIM == 3
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_z[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_vz[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_az[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
#endif
#endif
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_mass[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_uid[numParticlesLocal-h_particles2remove_counter],
                                                 (integer)0, h_particles2remove_counter);
#if SPH_SIM
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_materialId[numParticlesLocal-h_particles2remove_counter],
                                                 (integer)0, h_particles2remove_counter);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_sml[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_rho[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_p[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_e[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_cs[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
#endif

    //TODO: all entries (removing particles)

    //cuda::free(d_tempInt);

    numParticlesLocal -= h_particles2remove_counter;
    Logger(INFO) << "removing #" << h_particles2remove_counter << " particles!";


#else

    // TODO: CPU removeParticles()
    real time = 0.;

#endif // TARGET_GPU

    return time;
}

// used for gravity and sph
template <typename T>
integer Miluphpc::sendParticles(T *sendBuffer, T *receiveBuffer, integer *sendLengths, integer *receiveLengths) {

    //Timer temp;
    boost::mpi::communicator comm;

    std::vector<boost::mpi::request> reqParticles;
    std::vector<boost::mpi::status> statParticles;

    integer reqCounter = 0;
    integer receiveOffset = 0;
    integer sendOffset = 0;

    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            reqParticles.push_back(comm.isend(proc, 17, &sendBuffer[sendOffset], sendLengths[proc]));
            statParticles.push_back(comm.recv(proc, 17, &receiveBuffer[receiveOffset], receiveLengths[proc]));

            receiveOffset += receiveLengths[proc];
            sendOffset += sendLengths[proc];
        }
    }

    boost::mpi::wait_all(reqParticles.begin(), reqParticles.end());

    //real elapsed = temp.elapsed();
    //Logger(TIME) << "sph: resending: " << elapsed;
    //if (elapsed > 10.) {
    //Logger(INFO) << "sph: resending: receiveOffset: " << receiveOffset << ", sendOffset: " << sendOffset;
    //}

}

// used for assigning particles to corresponding process
template <typename T>
integer Miluphpc::sendParticlesEntry(integer *sendLengths, integer *receiveLengths, T *entry, T *entryBuffer, T *copyBuffer) {

    boost::mpi::communicator comm;

    std::vector<boost::mpi::request> reqParticles;
    std::vector<boost::mpi::status> statParticles;

    integer reqCounter = 0;
    integer receiveOffset = 0;
    integer sendOffset = 0;

    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            if (proc == 0) {
                reqParticles.push_back(comm.isend(proc, 17, &entry[0], sendLengths[proc]));
                //Logger(INFO) << "Sending from: " << 0 << " to proc: " << proc;
            }
            else {
                reqParticles.push_back(comm.isend(proc, 17,
                                                  &entry[subDomainKeyTreeHandler->h_procParticleCounter[proc-1] + sendOffset],
                                                  sendLengths[proc]));

                //Logger(INFO) << "Sending from: " << subDomainKeyTreeHandler->h_procParticleCounter[proc-1] + sendOffset << " to proc: " << proc;
            }
            //reqParticles.push_back(comm.isend(proc, 17,&entry[sendOffset], sendLengths[proc]));
            statParticles.push_back(comm.recv(proc, 17, &entryBuffer[0] + receiveOffset,
                                              receiveLengths[proc]));

            //Logger(INFO) << "Receiving at " << receiveOffset << " from proc: " << proc;

            //sendOffset += subDomainKeyTreeHandler->h_procParticleCounter[proc-1]; //sendLengths[proc];
            receiveOffset += receiveLengths[proc];
        }
        sendOffset += subDomainKeyTreeHandler->h_procParticleCounter[proc-1]; //sendLengths[proc];
    }

    boost::mpi::wait_all(reqParticles.begin(), reqParticles.end());

    integer offset = 0;
    for (int i=0; i < subDomainKeyTreeHandler->h_subDomainKeyTree->rank; i++) {
        offset += subDomainKeyTreeHandler->h_procParticleCounter[i];
    }

    if (subDomainKeyTreeHandler->h_subDomainKeyTree->rank != 0) {
        if (offset > 0 && (subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] - offset) > 0) {
            //TODO: following line needed? (probably not)
            //HelperNS::Kernel::Launch::copyArray(&entry[0], &entry[subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] - offset], offset);
            //KernelHandler.copyArray(&entry[0], &entry[h_procCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] - offset], offset);
        }
#if TARGET_GPU
        HelperNS::Kernel::Launch::copyArray(&copyBuffer[0], &entry[offset], subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank]);
        HelperNS::Kernel::Launch::copyArray(&entry[0], &copyBuffer[0], subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank]);
#else
       // TODO: CPU copyArray
#endif // TARGET_GPU
        //KernelHandler.copyArray(&d_tempArray_2[0], &entry[offset], subDomainKeyTreeHandler->d_h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank]);
        //KernelHandler.copyArray(&entry[0], &d_tempArray_2[0], subDomainKeyTreeHandler->d_h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank]);
        //Logger(INFO) << "moving from offet: " << offset << " length: " << subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank];
    }

#if TARGET_GPU
    HelperNS::Kernel::Launch::resetArray(&entry[subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank]],
                                         (T)0, numParticles-subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank]);
    HelperNS::Kernel::Launch::copyArray(&entry[subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank]],
                                        entryBuffer, receiveOffset);
#else
    // TODO: CPU copyArray()
#endif // TARGET_GPU

     //KernelHandler.resetFloatArray(&entry[h_procCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank]], 0, numParticles-h_procCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank]); //resetFloatArrayKernel(float *array, float value, int n)
    //KernelHandler.copyArray(&entry[h_procCounter[h_subDomainHandler->rank]], d_tempArray, receiveOffset);

    //Logger(INFO) << "numParticlesLocal = " << receiveOffset << " + " << subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank];
    return receiveOffset + subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank];
}

real Miluphpc::particles2file(int step) {

    Timer timer;

    boost::mpi::communicator comm;
    sumParticles = numParticlesLocal;
    all_reduce(comm, boost::mpi::inplace_t<integer*>(&sumParticles), 1, std::plus<integer>());

    std::stringstream stepss;
    stepss << std::setw(6) << std::setfill('0') << step;

    HighFive::File h5file(simulationParameters.directory + "ts" + stepss.str() + ".h5",
                          HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate,
                          HighFive::MPIOFileDriver(MPI_COMM_WORLD, MPI_INFO_NULL));

    configParameters2file(h5file);

    std::vector <size_t> dataSpaceDims(2);
    dataSpaceDims[0] = std::size_t(sumParticles);
    dataSpaceDims[1] = DIM;

    HighFive::DataSet ranges = h5file.createDataSet<keyType>("/ranges", HighFive::DataSpace(subDomainKeyTreeHandler->h_numProcesses + 1));

    keyType *rangeValues;
    rangeValues = new keyType[subDomainKeyTreeHandler->h_numProcesses + 1];

    subDomainKeyTreeHandler->copy(To::host, true, false);
    for (int i=0; i<=subDomainKeyTreeHandler->h_numProcesses; i++) {
        rangeValues[i] = subDomainKeyTreeHandler->h_subDomainKeyTree->range[i];
        Logger(INFO) << "rangeValues[" << i << "] = " << rangeValues[i];
    }

    ranges.write(rangeValues);

    delete [] rangeValues;

    // TODO: add uid (and other entries?)
    std::vector<real> time;
#if TARGET_GPU
    simulationTimeHandler->copy(To::host);
#endif // TARGET_GPU
    time.push_back(*simulationTimeHandler->h_currentTime); //step*simulationParameters.timestep);
    HighFive::DataSet h5_time = h5file.createDataSet<real>("/time", HighFive::DataSpace::From(time));

    HighFive::DataSet pos = h5file.createDataSet<real>("/x", HighFive::DataSpace(dataSpaceDims));
    HighFive::DataSet vel = h5file.createDataSet<real>("/v", HighFive::DataSpace(dataSpaceDims));
    HighFive::DataSet key = h5file.createDataSet<keyType>("/key", HighFive::DataSpace(sumParticles));
    HighFive::DataSet h5_mass = h5file.createDataSet<real>("/m", HighFive::DataSpace(sumParticles));
    HighFive::DataSet h5_proc = h5file.createDataSet<int>("/proc", HighFive::DataSpace(sumParticles));
#if SPH_SIM
    HighFive::DataSet h5_rho = h5file.createDataSet<real>("/rho", HighFive::DataSpace(sumParticles));
    HighFive::DataSet h5_p = h5file.createDataSet<real>("/p", HighFive::DataSpace(sumParticles));
    HighFive::DataSet h5_e = h5file.createDataSet<real>("/e", HighFive::DataSpace(sumParticles));
    HighFive::DataSet h5_sml = h5file.createDataSet<real>("/sml", HighFive::DataSpace(sumParticles));
    HighFive::DataSet h5_noi = h5file.createDataSet<integer>("/noi", HighFive::DataSpace(sumParticles));
    HighFive::DataSet h5_cs = h5file.createDataSet<real>("/cs", HighFive::DataSpace(sumParticles));
#endif

    HighFive::DataSet h5_totalEnergy;
    if (simulationParameters.calculateEnergy) {
        h5_totalEnergy = h5file.createDataSet<real>("/totalEnergy",
                        HighFive::DataSpace(subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses));
    }
    HighFive::DataSet h5_totalAngularMomentum;
    if (simulationParameters.calculateAngularMomentum) {
        h5_totalAngularMomentum = h5file.createDataSet<real>("/totalAngularMomentum",
                             HighFive::DataSpace(subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses));
    }

    Logger(INFO) << "creating datasets ...";
    // ----------

    std::vector<std::vector<real>> x, v; // two dimensional vector for 3D vector data
    std::vector<keyType> k; // one dimensional vector holding particle keys
    std::vector<real> mass;
    std::vector<int> particleProc;
#if SPH_SIM
    std::vector<real> rho, p, e, sml, cs;
    std::vector<integer> noi;
#endif

    Logger(INFO) << "copying particles ...";

#if TARGET_GPU
    particleHandler->copyDistribution(To::host, true, false);
#if SPH_SIM
    particleHandler->copySPH(To::host);
#endif
#endif // TARGET_GPU

    Logger(INFO) << "getting particle keys ...";
#if TARGET_GPU
    keyType *d_keys = buffer->d_keyTypeBuffer;
#endif
    //cuda::malloc(d_keys, numParticlesLocal);

    //TreeNS::Kernel::Launch::getParticleKeys(treeHandler->d_tree, particleHandler->d_particles,
    //                                        d_keys, 21, numParticlesLocal, curveType);
#if TARGET_GPU
    SubDomainKeyTreeNS::Kernel::Launch::getParticleKeys(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                        treeHandler->d_tree, particleHandler->d_particles,
                                                        d_keys, 21, numParticlesLocal, curveType);
#else
    // TODO: CPU getParticleKeys()
#endif
    //SubDomainKeyTreeNS::Kernel::Launch::getParticleKeys(subDomainKeyTreeHandler->d_subDomainKeyTree,
    //                                                    treeHandler->d_tree, particleHandler->d_particles,
    //                                                    d_keys, 21, numParticlesLocal, curveType);


    keyType *h_keys;
    h_keys = new keyType[numParticlesLocal];
#if TARGET_GPU
    cuda::copy(h_keys, d_keys, numParticlesLocal, To::host);
#endif

    integer keyProc;

    Logger(INFO) << "filling vectors ...";

    for (int i=0; i<numParticlesLocal; i++) {
#if DIM == 1
        x.push_back({particleHandler->h_x[i]});
        v.push_back({particleHandler->h_vx[i]});
#elif DIM == 2
        x.push_back({particleHandler->h_x[i], particleHandler->h_y[i]});
        v.push_back({particleHandler->h_vx[i], particleHandler->h_vy[i]});
#else
        x.push_back({particleHandler->h_x[i], particleHandler->h_y[i], particleHandler->h_z[i]});
        v.push_back({particleHandler->h_vx[i], particleHandler->h_vy[i], particleHandler->h_vz[i]});
#endif
        k.push_back(h_keys[i]);
        mass.push_back(particleHandler->h_mass[i]);
        particleProc.push_back(subDomainKeyTreeHandler->h_subDomainKeyTree->rank);
        //Logger(INFO) << "mass[" << i << "] = " << mass[i];
#if SPH_SIM
        rho.push_back(particleHandler->h_rho[i]);
        p.push_back(particleHandler->h_p[i]);
        e.push_back(particleHandler->h_e[i]);
        sml.push_back(particleHandler->h_sml[i]);
        noi.push_back(particleHandler->h_noi[i]);
        cs.push_back(particleHandler->h_cs[i]);
#endif
    }

    //cuda::free(d_keys);
    delete [] h_keys;

    // receive buffer
    int procN[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];

    // send buffer
    int sendProcN[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++){
        sendProcN[proc] = subDomainKeyTreeHandler->h_subDomainKeyTree->rank == proc ? numParticlesLocal : 0;
    }

    all_reduce(comm, sendProcN, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses, procN,
               boost::mpi::maximum<integer>());

    std::size_t nOffset = 0;
    // count total particles on other processes
    for (int proc = 0; proc < subDomainKeyTreeHandler->h_subDomainKeyTree->rank; proc++){
        nOffset += procN[proc];
    }
    Logger(DEBUG) << "Offset to write to datasets: " << std::to_string(nOffset);

    Logger(INFO) << "writing to h5 ...";

    h5_time.write(time);
    // write to associated datasets in h5 file
    // only working when load balancing has been completed and even number of particles
    pos.select({nOffset, 0},
                {std::size_t(numParticlesLocal), std::size_t(DIM)}).write(x);
    vel.select({nOffset, 0},
                {std::size_t(numParticlesLocal), std::size_t(DIM)}).write(v);
    key.select({nOffset}, {std::size_t(numParticlesLocal)}).write(k);
    h5_mass.select({nOffset}, {std::size_t(numParticlesLocal)}).write(mass);
    h5_proc.select({nOffset}, {std::size_t(numParticlesLocal)}).write(particleProc);
#if SPH_SIM
    h5_rho.select({nOffset}, {std::size_t(numParticlesLocal)}).write(rho);
    h5_p.select({nOffset}, {std::size_t(numParticlesLocal)}).write(p);
    h5_e.select({nOffset}, {std::size_t(numParticlesLocal)}).write(e);
    h5_sml.select({nOffset}, {std::size_t(numParticlesLocal)}).write(sml);
    h5_noi.select({nOffset}, {std::size_t(numParticlesLocal)}).write(noi);
    h5_cs.select({nOffset}, {std::size_t(numParticlesLocal)}).write(cs);
#endif

#if TARGET_GPU
    if (simulationParameters.calculateEnergy) {
        h5_totalEnergy.select({subDomainKeyTreeHandler->h_subDomainKeyTree->rank}, {1}).write(totalEnergy);
    }
    if (simulationParameters.calculateAngularMomentum) {
        h5_totalAngularMomentum.select({subDomainKeyTreeHandler->h_subDomainKeyTree->rank}, {1}).write(totalAngularMomentum);
    }
#endif // TARGET_GPU

    if (simulationParameters.calculateCenterOfMass) {
        real *h_com = new real[DIM];
#if TARGET_GPU
        real *d_com = buffer->d_realBuffer;
        //cuda::malloc(d_com, DIM);
        TreeNS::Kernel::Launch::globalCOM(treeHandler->d_tree, particleHandler->d_particles, d_com);
        cuda::copy(h_com, d_com, DIM, To::host);
#endif // TARGET_GPU
        for (int i=0; i<DIM; i++) {
            Logger(DEBUG) << "com[" << i << "] = " << h_com[i];
        }
        //cuda::free(d_com);

        HighFive::DataSet _com = h5file.createDataSet<real>("/COM", HighFive::DataSpace(DIM));
        std::vector<real> centerOfMass;
        centerOfMass.push_back(h_com[0]);
#if DIM > 1
        centerOfMass.push_back(h_com[1]);
#if DIM == 3
        centerOfMass.push_back(h_com[2]);
#endif
#endif
        _com.select({0}, {std::size_t(DIM)}).write(centerOfMass);
        delete [] h_com;
    }

    return timer.elapsed();

}

real Miluphpc::particles2file(const std::string& filename, int *particleIndices, int length) {

    Timer timer;

    boost::mpi::communicator comm;
    int totalLength = length;
    all_reduce(comm, boost::mpi::inplace_t<integer*>(&totalLength), 1, std::plus<integer>());

    std::stringstream file;
    file << simulationParameters.logDirectory <<  filename.c_str() << ".h5";
    //stepss << std::setw(6) << std::setfill('0') << step;

    HighFive::File h5file(file.str(),
                          HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate,
                          HighFive::MPIOFileDriver(MPI_COMM_WORLD, MPI_INFO_NULL));

    std::vector <size_t> dataSpaceDims(2);
    dataSpaceDims[0] = std::size_t(totalLength);
    dataSpaceDims[1] = DIM;

    HighFive::DataSet ranges = h5file.createDataSet<keyType>("/ranges",
                                                             HighFive::DataSpace(subDomainKeyTreeHandler->h_numProcesses + 1));

    keyType *rangeValues;
    rangeValues = new keyType[subDomainKeyTreeHandler->h_numProcesses + 1];

    subDomainKeyTreeHandler->copy(To::host, true, false);
    for (int i=0; i<=subDomainKeyTreeHandler->h_numProcesses; i++) {
        rangeValues[i] = subDomainKeyTreeHandler->h_subDomainKeyTree->range[i];
        Logger(INFO) << "rangeValues[" << i << "] = " << rangeValues[i];
    }

    ranges.write(rangeValues);

    delete [] rangeValues;

    HighFive::DataSet pos = h5file.createDataSet<real>("/x", HighFive::DataSpace(dataSpaceDims));
    //HighFive::DataSet vel = h5file.createDataSet<real>("/v", HighFive::DataSpace(dataSpaceDims));
    HighFive::DataSet key = h5file.createDataSet<keyType>("/key", HighFive::DataSpace(totalLength));

    // ----------

    std::vector<std::vector<real>> x, v; // two dimensional vector for 3D vector data
    std::vector<keyType> k; // one dimensional vector holding particle keys

#if TARGET_GPU
    particleHandler->copyDistribution(To::host, true, false, true);

    keyType *d_keys = buffer->d_keyTypeBuffer;
    //cuda::malloc(d_keys, numNodes);

    //TreeNS::Kernel::Launch::getParticleKeys(treeHandler->d_tree, particleHandler->d_particles,
    //                                        d_keys, 21, numParticlesLocal, curveType);
    SubDomainKeyTreeNS::Kernel::Launch::getParticleKeys(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                        treeHandler->d_tree, particleHandler->d_particles,
                                                        d_keys, 21, numNodes, curveType);
#else
    // TODO: ...
#endif // TARGET_GPU
    //SubDomainKeyTreeNS::Kernel::Launch::getParticleKeys(subDomainKeyTreeHandler->d_subDomainKeyTree,
    //                                                    treeHandler->d_tree, particleHandler->d_particles,
    //                                                    d_keys, 21, numParticlesLocal, curveType);


    keyType *h_keys;
    h_keys = new keyType[numNodes];
#if TARGET_GPU
    cuda::copy(h_keys, d_keys, numNodes, To::host);
#endif // TARGET_GPU

    integer keyProc;

    for (int i=0; i<length; i++) {
#if DIM == 1
        x.push_back({particleHandler->h_x[particleIndices[i]]});
        //v.push_back({particleHandler->h_vx[particleIndices[i]]});
#elif DIM == 2
        x.push_back({particleHandler->h_x[particleIndices[i]], particleHandler->h_y[particleIndices[i]]});
        //v.push_back({particleHandler->h_vx[i], particleHandler->h_vy[i]});
#else
        x.push_back({particleHandler->h_x[particleIndices[i]], particleHandler->h_y[particleIndices[i]],
                     particleHandler->h_z[particleIndices[i]]});
        //v.push_back({particleHandler->h_vx[particleIndices[i]], particleHandler->h_vy[particleIndices[i]],
        //             particleHandler->h_vz[particleIndices[i]]});
#endif
        k.push_back(h_keys[particleIndices[i]]);
    }

    //cuda::free(d_keys);
    delete [] h_keys;

    // receive buffer
    int procN[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];

    // send buffer
    int sendProcN[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++){
        sendProcN[proc] = subDomainKeyTreeHandler->h_subDomainKeyTree->rank == proc ? length : 0;
    }

    all_reduce(comm, sendProcN, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses, procN,
               boost::mpi::maximum<integer>());

    std::size_t nOffset = 0;
    // count total particles on other processes
    for (int proc = 0; proc < subDomainKeyTreeHandler->h_subDomainKeyTree->rank; proc++){
        nOffset += procN[proc];
    }
    Logger(DEBUG) << "Offset to write to datasets: " << std::to_string(nOffset);

    // write to associated datasets in h5 file
    // only working when load balancing has been completed and even number of particles
    pos.select({nOffset, 0},
               {std::size_t(length), std::size_t(DIM)}).write(x);
    //vel.select({nOffset, 0},
    //           {std::size_t(length), std::size_t(DIM)}).write(v);
    key.select({nOffset}, {std::size_t(length)}).write(k);

    return timer.elapsed();

}

real Miluphpc::configParameters2file(HighFive::File &h5file) {

    // TODO: write config parameters to HDF5 file

    // not working if only one rank writes attributes ...
    //if (subDomainKeyTreeHandler->h_subDomainKeyTree->rank == 0) {
    //    HighFive::Group header = h5file.createGroup("/Header");
    //    //header.createAttribute<int>("test", 1);
    //}

    // testing
    HighFive::Group header = h5file.createGroup("config");
    int test_1 = 10;
    HighFive::Attribute b_1 = header.createAttribute<int>("test_1", HighFive::DataSpace::From(test_1));
    b_1.write(test_1);
    double test_2 = 1.5;
    HighFive::Attribute b_2 = header.createAttribute<double>("test_2", test_2);
    // end: testing

    // which information are necessary
    // - time (to know from which start time to continue)
    // - ...
}

void Miluphpc::getMemoryInfo() {

#if TARGET_GPU
    // TODO: simulate other memory allocations to include "peak memory usage" ?
    //  - including updateRange()
    //  - sorting using CUDA cub
    //  - ...

    size_t free_bytes, total_bytes, used_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    used_bytes = total_bytes - free_bytes;

    //Logger(INFO) << "free bytes:  " << (double)free_bytes << " B = " << 9.31e-10 * free_bytes << " GB ("
    //             << (double)free_bytes/(double)total_bytes * 100. << " %)";
    //Logger(INFO) << "used bytes:  " << (double)used_bytes << " B = " << 9.31e-10 * used_bytes << " GB ("
    //             << (double)used_bytes/(double)total_bytes * 100. << " %)";
    //Logger(INFO) << "total bytes: " << (double)total_bytes << " B = " << 9.31e-10 * total_bytes << " GB";

    Logger(INFO) << "MEMORY INFO: used: " << std::setprecision(4) << 9.31e-10 * used_bytes << " GB ("
                 << (double)used_bytes/(double)total_bytes * 100. << " %)"
                 << " free: " << 9.31e-10 * free_bytes << " GB ("
                 << (double)free_bytes/(double)total_bytes * 100. << " %)"
                 << " available: " << 9.31e-10 * total_bytes << " GB";

#else
    // TODO: CPU getMemoryInfo()
#endif // TARGET_GPU

}
