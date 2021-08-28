#include "../include/miluphpc.h"

Miluphpc::Miluphpc(integer numParticles, integer numNodes) : numParticles(numParticles), numNodes(numNodes) {

    //TODO: how to distinguish/intialize numParticlesLocal vs numParticles
    //numParticlesLocal = numParticles/2;

    gpuErrorcheck(cudaMalloc((void**)&d_mutex, sizeof(integer)));
    helperHandler = new HelperHandler(numParticles);
    buffer = new HelperHandler(numParticles);
    particleHandler = new ParticleHandler(numParticles, numNodes);
    subDomainKeyTreeHandler = new SubDomainKeyTreeHandler();
    treeHandler = new TreeHandler(numParticles, numNodes);

    numParticlesLocal = numParticles/2 + subDomainKeyTreeHandler->h_subDomainKeyTree->rank * 5000;

}

Miluphpc::~Miluphpc() {

    delete helperHandler;
    delete buffer;
    delete particleHandler;
    delete subDomainKeyTreeHandler;
    delete treeHandler;

}

void Miluphpc::initDistribution(ParticleDistribution::Type particleDistribution) {

    switch(particleDistribution) {
        case ParticleDistribution::disk:
            diskModel();
            break;
        case ParticleDistribution::plummer:
            //
            break;
        default:
            diskModel();
    }

    particleHandler->distributionToDevice();
}

void Miluphpc::diskModel() {

    real a = 1.0;
    real pi = 3.14159265;
    std::default_random_engine generator;
    std::uniform_real_distribution<real> distribution(1.5, 12.0);
    std::uniform_real_distribution<real> distribution_theta_angle(0.0, 2 * pi);

    real solarMass = 100000;

    // loop through all particles
    for (int i = 0; i < numParticlesLocal; i++) {

        real theta_angle = distribution_theta_angle(generator);
        real r = distribution(generator);

        // set mass and position of particle
        if (subDomainKeyTreeHandler->h_subDomainKeyTree->rank == 0) {
            if (i == 0) {
                particleHandler->h_particles->mass[i] = 2 * solarMass / numParticlesLocal; //solarMass; //100000; 2 * solarMass / numParticles;
                particleHandler->h_particles->x[i] = 0;
                particleHandler->h_particles->y[i] = 0;
                particleHandler->h_particles->z[i] = 0;
            } else {
                particleHandler->h_particles->mass[i] = 2 * solarMass / numParticlesLocal;
                particleHandler->h_particles->x[i] = r * cos(theta_angle);
                //y[i] = r * sin(theta);
                particleHandler->h_particles->z[i] = r * sin(theta_angle);

                if (i % 2 == 0) {
                    particleHandler->h_particles->y[i] = i * 1e-7;//z[i] = i * 1e-7;
                } else {
                    particleHandler->h_particles->y[i] = i * -1e-7;//z[i] = i * -1e-7;
                }
            }
        }
        else {
            particleHandler->h_particles->mass[i] = 2 * solarMass / numParticlesLocal;
            particleHandler->h_particles->x[i] = (r + subDomainKeyTreeHandler->h_subDomainKeyTree->rank * 1.1e-1) *
                    cos(theta_angle) + 1.0e-2*subDomainKeyTreeHandler->h_subDomainKeyTree->rank;
            //y[i] = (r + h_subDomainHandler->rank * 1.3e-1) * sin(theta) + 1.1e-2*h_subDomainHandler->rank;
            particleHandler->h_particles->z[i] = (r + subDomainKeyTreeHandler->h_subDomainKeyTree->rank * 1.3e-1) *
                    sin(theta_angle) + 1.1e-2*subDomainKeyTreeHandler->h_subDomainKeyTree->rank;

            if (i % 2 == 0) {
                //z[i] = i * 1e-7 * h_subDomainHandler->rank + 0.5e-7*h_subDomainHandler->rank;
                particleHandler->h_particles->y[i] = i * 1e-7 * subDomainKeyTreeHandler->h_subDomainKeyTree->rank +
                        0.5e-7*subDomainKeyTreeHandler->h_subDomainKeyTree->rank;
            } else {
                //z[i] = i * -1e-7 * h_subDomainHandler->rank + 0.4e-7*h_subDomainHandler->rank;
                particleHandler->h_particles->y[i] = i * -1e-7 * subDomainKeyTreeHandler->h_subDomainKeyTree->rank
                        + 0.4e-7*subDomainKeyTreeHandler->h_subDomainKeyTree->rank;
            }
        }


        // set velocity of particle
        real rotation = 1;  // 1: clockwise   -1: counter-clockwise
        real v = sqrt(solarMass / (r));

        if (i == 0) {
            particleHandler->h_particles->vx[0] = 0.0;
            particleHandler->h_particles->vy[0] = 0.0;
            particleHandler->h_particles->vz[0] = 0.0;
        }
        else{
            particleHandler->h_particles->vx[i] = rotation*v*sin(theta_angle);
            //y_vel[i] = -rotation*v*cos(theta);
            particleHandler->h_particles->vz[i] = -rotation*v*cos(theta_angle);
            //z_vel[i] = 0.0;
            particleHandler->h_particles->vy[i] = 0.0;
        }

        // set acceleration to zero
        particleHandler->h_particles->ax[i] = 0.0;
        particleHandler->h_particles->ay[i] = 0.0;
        particleHandler->h_particles->az[i] = 0.0;
    }

}

void Miluphpc::run() {

    real time;

    Logger(INFO) << "Starting ...";

    Logger(INFO) << "initialize particle distribution ...";
    initDistribution();

    for (int i=0; i<numParticles; i++) {
        if (i % 10000 == 0) {
            printf("host: x[%i] = (%f, %f, %f)\n", i, particleHandler->h_x[i], particleHandler->h_y[i],
                   particleHandler->h_z[i]);
        }
    }

    time = ParticlesNS::Kernel::Launch::test(particleHandler->d_particles, true);
    Logger(TIME) << "test: " << time << " ms";
    //treeHandler->toHost();
    //treeHandler->toDevice();

    Logger(INFO) << "resetting (device) arrays ...";
    time = Kernel::Launch::resetArrays(treeHandler->d_tree, particleHandler->d_particles, d_mutex, numParticles,
                                       numNodes, true);
    Logger(TIME) << "resetArrays: " << time << " ms";

    Logger(INFO) << "computing bounding box ...";
    //TreeNS::computeBoundingBoxKernel(treeHandler->d_tree, particleHandler->d_particles, d_mutex, numNodes, 256);
    time = TreeNS::Kernel::Launch::computeBoundingBox(treeHandler->d_tree, particleHandler->d_particles, d_mutex,
                                                           numParticles, 256, true);
    Logger(TIME) << "computeBoundingBox: " << time << " ms";

    treeHandler->toHost();
    printf("min/max: x = (%f, %f), y = (%f, %f), z = (%f, %f)\n", *treeHandler->h_minX, *treeHandler->h_maxX,
           *treeHandler->h_minY, *treeHandler->h_maxY, *treeHandler->h_minZ, *treeHandler->h_maxZ);

    treeHandler->globalizeBoundingBox(Execution::device);
    treeHandler->toHost();
    printf("min/max: x = (%f, %f), y = (%f, %f), z = (%f, %f)\n", *treeHandler->h_minX, *treeHandler->h_maxX,
           *treeHandler->h_minY, *treeHandler->h_maxY, *treeHandler->h_minZ, *treeHandler->h_maxZ);

    Logger(INFO) << "building tree ...";
    time = TreeNS::Kernel::Launch::buildTree(treeHandler->d_tree, particleHandler->d_particles, numParticles,
                                             numParticles, true);
    Logger(TIME) << "buildTree: " << time << " ms";

    Logger(INFO) << "center of mass ...";
    time = TreeNS::Kernel::Launch::centerOfMass(treeHandler->d_tree, particleHandler->d_particles,
                                                numParticles, true);
    Logger(TIME) << "centerOfMass: " << time << " ms";

    Logger(INFO) << "sorting ...";
    time = TreeNS::Kernel::Launch::sort(treeHandler->d_tree, numParticles, numNodes, true);
    Logger(TIME) << "sort: " << time << " ms";


    printf("host: subDomainKeyTree->rank = %i\n", subDomainKeyTreeHandler->h_subDomainKeyTree->rank);
    printf("host: subDomainKeyTree->numProcesses = %i\n", subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses);
    SubDomainKeyTreeNS::Kernel::Launch::test(subDomainKeyTreeHandler->d_subDomainKeyTree);


    Integrator defaultIntegrator;
    defaultIntegrator.integrate();

    Integrator eulerIntegrator(IntegratorSelection::euler);
    eulerIntegrator.integrate();

    Integrator predictorCorrectorIntegrator(IntegratorSelection::predictor_corrector);
    predictorCorrectorIntegrator.integrate();

    /*MaterialHandler materialHandler(1, subDomainKeyTreeHandler->h_subDomainKeyTree->rank,
                                    100, 3.5, 2.5);
    materialHandler.h_materials[0].info();
    //materialHandler.communicate(0, 1);
    materialHandler.broadcast();
    materialHandler.h_materials[0].info();*/

    MaterialHandler materialHandler("config/material.cfg");

    for (int i=0; i<materialHandler.numMaterials; i++) {
        materialHandler.h_materials[i].info();
    }

    //TESTING: cuda utils: findDuplicates()
    //gpuErrorcheck(cudaMemset(treeHandler->d_index, 0, sizeof(integer)));
    //CudaUtils::Kernel::Launch::findDuplicates(particleHandler->d_x,
    //                                          treeHandler->d_index, 1000);
    //integer duplicates;
    //gpuErrorcheck(cudaMemcpy(&duplicates, treeHandler->d_index, sizeof(integer), cudaMemcpyDeviceToHost));
    //Logger(INFO) << "duplicates: " << duplicates;
    //END: TESTING

    //TESTING: Helper: sortArray()
    //HelperHandler *helperHandler = new HelperHandler(1000);
    //HelperHandler helperHandler(10000);
    //gpuErrorcheck(cudaMemset(&helperHandler.d_integerBuffer[0], 10, 200 * sizeof(integer)));
    //gpuErrorcheck(cudaMemset(&helperHandler.d_integerBuffer[200], 100, 200 * sizeof(integer)));
    //gpuErrorcheck(cudaMemset(&helperHandler.d_integerBuffer[400], 1, 200 * sizeof(integer)));
    //gpuErrorcheck(cudaMemset(&helperHandler.d_integerBuffer[600], 155, 200 * sizeof(integer)));
    //gpuErrorcheck(cudaMemset(&helperHandler.d_integerBuffer[800], 17, 200 * sizeof(integer)));
    // Works both!
    //HelperNS::sortArray<real, integer>(particleHandler->d_x, helperHandler.d_realBuffer, &helperHandler.d_integerBuffer[0],
    //                    helperHandler.d_integerBuffer[&5000], 1000);
    //HelperNS::sortArray(particleHandler->d_x, helperHandler.d_realBuffer, &helperHandler.d_integerBuffer[0],
    //                    &helperHandler.d_integerBuffer[5000], 1000);


    int *toReceive = new int[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    int *toSend = new int[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];

    for (int i=0; i<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; i++) {
        toReceive[i] = 0;
        if (i != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            toSend[i] = 10 + subDomainKeyTreeHandler->h_subDomainKeyTree->rank;
        }
    }

    mpi::messageLengths(subDomainKeyTreeHandler->h_subDomainKeyTree, toSend, toReceive);

    for (int i=0; i<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; i++) {
        Logger(INFO) << "toReceive[" << i << "] = " << toReceive[i];
    }

    delete [] toReceive;
    delete [] toSend;

}

void Miluphpc::barnesHut() {

    real time;

    Logger(INFO) << "Starting ...";

    Logger(INFO) << "initialize particle distribution ...";
    initDistribution();

    if (true/*parameters.loadBalancing*/) {
        Logger(INFO) << "load balancing ...";
        if (true/*step == 0 || step % parameters.loadBalancingInterval == 0*/) {
            newLoadDistribution();
        }
    }

    Logger(INFO) << "resetting (device) arrays ...";
    time = Kernel::Launch::resetArrays(treeHandler->d_tree, particleHandler->d_particles, d_mutex, numParticles,
                                       numNodes, true);
    helperHandler->reset();

    Logger(TIME) << "resetArrays: " << time << " ms";

    Logger(INFO) << "computing bounding box ...";
    //TreeNS::computeBoundingBoxKernel(treeHandler->d_tree, particleHandler->d_particles, d_mutex, numNodes, 256);
    time = TreeNS::Kernel::Launch::computeBoundingBox(treeHandler->d_tree, particleHandler->d_particles, d_mutex,
                                                      numParticlesLocal, 256, true);
    Logger(TIME) << "computeBoundingBox: " << time << " ms";

    treeHandler->toHost();
    printf("min/max: x = (%f, %f), y = (%f, %f), z = (%f, %f)\n", *treeHandler->h_minX, *treeHandler->h_maxX,
           *treeHandler->h_minY, *treeHandler->h_maxY, *treeHandler->h_minZ, *treeHandler->h_maxZ);

    treeHandler->globalizeBoundingBox(Execution::device);
    treeHandler->toHost();
    printf("min/max: x = (%f, %f), y = (%f, %f), z = (%f, %f)\n", *treeHandler->h_minX, *treeHandler->h_maxX,
           *treeHandler->h_minY, *treeHandler->h_maxY, *treeHandler->h_minZ, *treeHandler->h_maxZ);

    SubDomainKeyTreeNS::Kernel::Launch::particlesPerProcess(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                            treeHandler->d_tree, particleHandler->d_particles,
                                                            numParticlesLocal, numNodes);

    SubDomainKeyTreeNS::Kernel::Launch::markParticlesProcess(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                             treeHandler->d_tree, particleHandler->d_particles,
                                                             numParticlesLocal, numNodes,
                                                             helperHandler->d_integerBuffer);

    float elapsedTimeSorting = 0.f;
    cudaEvent_t start_t_sorting, stop_t_sorting; // used for timing
    cudaEventCreate(&start_t_sorting);
    cudaEventCreate(&stop_t_sorting);
    cudaEventRecord(start_t_sorting, 0);

    // position: x
    HelperNS::sortArray(particleHandler->d_x, helperHandler->d_realBuffer, helperHandler->d_integerBuffer,
                        buffer->d_integerBuffer, numParticlesLocal);
    HelperNS::Kernel::Launch::copyArray(particleHandler->d_x, helperHandler->d_realBuffer, numParticlesLocal);
    // position: y
    HelperNS::sortArray(particleHandler->d_y, helperHandler->d_realBuffer, helperHandler->d_integerBuffer,
                        buffer->d_integerBuffer, numParticlesLocal);
    HelperNS::Kernel::Launch::copyArray(particleHandler->d_y, helperHandler->d_realBuffer, numParticlesLocal);
    // position: z
    HelperNS::sortArray(particleHandler->d_z, helperHandler->d_realBuffer, helperHandler->d_integerBuffer,
                        buffer->d_integerBuffer, numParticlesLocal);
    HelperNS::Kernel::Launch::copyArray(particleHandler->d_z, helperHandler->d_realBuffer, numParticlesLocal);


    // velocity: x
    HelperNS::sortArray(particleHandler->d_vx, helperHandler->d_realBuffer, helperHandler->d_integerBuffer,
                        buffer->d_integerBuffer, numParticlesLocal);
    HelperNS::Kernel::Launch::copyArray(particleHandler->d_vx, helperHandler->d_realBuffer, numParticlesLocal);
    // velocity: y
    HelperNS::sortArray(particleHandler->d_vy, helperHandler->d_realBuffer, helperHandler->d_integerBuffer,
                        buffer->d_integerBuffer, numParticlesLocal);
    HelperNS::Kernel::Launch::copyArray(particleHandler->d_vy, helperHandler->d_realBuffer, numParticlesLocal);
    // velocity: z
    HelperNS::sortArray(particleHandler->d_vz, helperHandler->d_realBuffer, helperHandler->d_integerBuffer,
                        buffer->d_integerBuffer, numParticlesLocal);
    HelperNS::Kernel::Launch::copyArray(particleHandler->d_vz, helperHandler->d_realBuffer, numParticlesLocal);

    // mass
    HelperNS::sortArray(particleHandler->d_mass, helperHandler->d_realBuffer, helperHandler->d_integerBuffer,
                        buffer->d_integerBuffer, numParticlesLocal);
    HelperNS::Kernel::Launch::copyArray(particleHandler->d_mass, helperHandler->d_realBuffer, numParticlesLocal);

    //TODO: for all entries ...

    cudaEventRecord(stop_t_sorting, 0);
    cudaEventSynchronize(stop_t_sorting);
    cudaEventElapsedTime(&elapsedTimeSorting, start_t_sorting, stop_t_sorting);
    cudaEventDestroy(start_t_sorting);
    cudaEventDestroy(stop_t_sorting);

    //TODO: next: sending particle entries (see: SPH: sendParticlesEntry...)

    Logger(TIME) << "\tSorting for process: " << elapsedTimeSorting << " ms";

    Logger(INFO) << "building tree ...";
    time = TreeNS::Kernel::Launch::buildTree(treeHandler->d_tree, particleHandler->d_particles, numParticlesLocal,
                                             numParticles, true);
    Logger(TIME) << "buildTree: " << time << " ms";

    Logger(INFO) << "center of mass ...";
    time = TreeNS::Kernel::Launch::centerOfMass(treeHandler->d_tree, particleHandler->d_particles,
                                                numParticlesLocal, true);
    Logger(TIME) << "centerOfMass: " << time << " ms";

    Logger(INFO) << "sorting ...";
    time = TreeNS::Kernel::Launch::sort(treeHandler->d_tree, numParticlesLocal, numNodes, true);
    Logger(TIME) << "sort: " << time << " ms";


    //particleHandler->


}

void Miluphpc::newLoadDistribution() {

    boost::mpi::communicator comm;

    Logger(INFO) << "numParticlesLocal = " << numParticlesLocal;

    int *processParticleCounts = new int[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];

    all_gather(comm, &numParticlesLocal, 1, processParticleCounts);

    int totalAmountOfParticles = 0;
    for (int i=0; i<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; i++) {
        Logger(INFO) << "numParticles on process: " << i << " = " << processParticleCounts[i];
        totalAmountOfParticles += processParticleCounts[i];
    }

    int aimedParticlesPerProcess = totalAmountOfParticles/subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses;
    Logger(INFO) << "aimedParticlesPerProcess = " << aimedParticlesPerProcess;

    updateRangeApproximately(aimedParticlesPerProcess, 15);

    delete [] processParticleCounts;
}

void Miluphpc::updateRangeApproximately(int aimedParticlesPerProcess, int bins) {

    // introduce "bin size" regarding keys
    //  keyHistRanges = [0, 1 * binSize, 2 * binSize, ... ]
    // calculate key of particles on the fly and assign to keyHistRanges
    //  keyHistNumbers = [1023, 50032, ...]
    // take corresponding keyHistRange as new range if (sum(keyHistRange[i]) > aimNumberOfParticles ...
    // communicate new ranges among processes

    boost::mpi::communicator comm;

    helperHandler->reset();

    Gravity::Kernel::Launch::createKeyHistRanges(helperHandler->d_helper, bins);

    Gravity::Kernel::Launch::keyHistCounter(treeHandler->d_tree, particleHandler->d_particles,
                                            subDomainKeyTreeHandler->d_subDomainKeyTree, helperHandler->d_helper,
                                            bins, numParticlesLocal);

    all_reduce(comm, boost::mpi::inplace_t<integer*>(helperHandler->d_integerBuffer), 1, std::plus<integer>());

    Gravity::Kernel::Launch::calculateNewRange(subDomainKeyTreeHandler->d_subDomainKeyTree, helperHandler->d_helper,
                                               bins, aimedParticlesPerProcess);

    subDomainKeyTreeHandler->toHost();

    Logger(INFO) << "numProcesses: " << subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses;
    for(int i=0; i<=subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; i++) {
        Logger(INFO) << "range[" << i << "] = " << subDomainKeyTreeHandler->h_subDomainKeyTree->range[i];
    }

    int h_sum;
    cudaMemcpy(&h_sum, helperHandler->d_integerBuffer, sizeof(integer), cudaMemcpyDeviceToHost);
    Logger(INFO) << "h_sum = " << h_sum;

}