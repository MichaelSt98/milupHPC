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
    domainListHandler = new DomainListHandler(DOMAIN_LIST_SIZE);
    lowestDomainListHandler = new DomainListHandler(DOMAIN_LIST_SIZE);

    numParticlesLocal = numParticles/2 + subDomainKeyTreeHandler->h_subDomainKeyTree->rank * 5000;
    Logger(INFO) << "numParticlesLocal = " << numParticlesLocal;

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

    time = TreeNS::Kernel::Launch::computeBoundingBox(treeHandler->d_tree, particleHandler->d_particles, d_mutex,
                                                      numParticlesLocal, 256, true);
    treeHandler->toHost();
    printf("min/max: x = (%f, %f), y = (%f, %f), z = (%f, %f)\n", *treeHandler->h_minX, *treeHandler->h_maxX,
           *treeHandler->h_minY, *treeHandler->h_maxY, *treeHandler->h_minZ, *treeHandler->h_maxZ);
    treeHandler->globalizeBoundingBox(Execution::device);

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

    subDomainKeyTreeHandler->toHost(); //TODO: needed?

    integer *sendLengths;
    sendLengths = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    sendLengths[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] = 0;
    integer *receiveLengths;
    receiveLengths = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    receiveLengths[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] = 0;

    for (int proc=0; proc < subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        printf("[rank %i] subDomainKeyTreeHandler->h_procParticleCounter[%i] = %i\n", subDomainKeyTreeHandler->h_rank,
               proc, subDomainKeyTreeHandler->h_procParticleCounter[proc]);
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            sendLengths[proc] = subDomainKeyTreeHandler->h_procParticleCounter[proc];
        }
    }
    mpi::messageLengths(subDomainKeyTreeHandler->h_subDomainKeyTree, sendLengths, receiveLengths);

    for (int proc = 0; proc < subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        Logger(INFO) << "sendLengths[" << proc << "] = " << sendLengths[proc];
        Logger(INFO) << "receiveLengths[" << proc << "] = " << receiveLengths[proc];
    }

    //TODO: next: sending particle entries (see: SPH: sendParticlesEntry...)
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_x);
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_vx);
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_ax);
#if DIM > 1
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_y);
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_vy);
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_ay);
#if DIM == 3
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_z);
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_vz);
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_az);
#endif
#endif

    numParticlesLocal = sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_mass);

    delete [] sendLengths;
    delete [] receiveLengths;

    HelperNS::Kernel::Launch::resetArray(&particleHandler->d_x[numParticlesLocal], (real)0, numParticles-numParticlesLocal);
    HelperNS::Kernel::Launch::resetArray(&particleHandler->d_vx[numParticlesLocal], (real)0, numParticles-numParticlesLocal);
    HelperNS::Kernel::Launch::resetArray(&particleHandler->d_ax[numParticlesLocal], (real)0, numParticles-numParticlesLocal);
#if DIM > 1
    HelperNS::Kernel::Launch::resetArray(&particleHandler->d_y[numParticlesLocal], (real)0, numParticles-numParticlesLocal);
    HelperNS::Kernel::Launch::resetArray(&particleHandler->d_vy[numParticlesLocal], (real)0, numParticles-numParticlesLocal);
    HelperNS::Kernel::Launch::resetArray(&particleHandler->d_ay[numParticlesLocal], (real)0, numParticles-numParticlesLocal);
#if DIM == 3
    HelperNS::Kernel::Launch::resetArray(&particleHandler->d_z[numParticlesLocal], (real)0, numParticles-numParticlesLocal);
    HelperNS::Kernel::Launch::resetArray(&particleHandler->d_vz[numParticlesLocal], (real)0, numParticles-numParticlesLocal);
    HelperNS::Kernel::Launch::resetArray(&particleHandler->d_az[numParticlesLocal], (real)0, numParticles-numParticlesLocal);
#endif
#endif
    HelperNS::Kernel::Launch::resetArray(&particleHandler->d_mass[numParticlesLocal], (real)0, numParticles-numParticlesLocal);

    Logger(TIME) << "\tSorting for process: " << elapsedTimeSorting << " ms";

    Logger(INFO) << "building domain list ...";
    time = DomainListNS::Kernel::Launch::createDomainList(subDomainKeyTreeHandler->d_subDomainKeyTree, domainListHandler->d_domainList,
                                                   MAX_LEVEL);
    Logger(TIME) << "createDomainList: " << time << " ms";

    integer domainListLength;
    cudaMemcpy(&domainListLength, domainListHandler->d_domainListIndex, sizeof(integer), cudaMemcpyDeviceToHost);
    Logger(INFO) << "domainListLength = " << domainListLength;

    Logger(INFO) << "building tree ...";
    time = TreeNS::Kernel::Launch::buildTree(treeHandler->d_tree, particleHandler->d_particles, numParticlesLocal,
                                             numParticles, true);
    Logger(TIME) << "buildTree: " << time << " ms";

    Logger(INFO) << "building domain tree ...";
    time = SubDomainKeyTreeNS::Kernel::Launch::buildDomainTree(treeHandler->d_tree, particleHandler->d_particles,
                                                         domainListHandler->d_domainList, numParticlesLocal,
                                                         numNodes);
    Logger(TIME) << "build(Domain)Tree: " << time << " ms";

    compPseudoParticlesParallel();

    parallelForce();

    //Logger(INFO) << "center of mass ...";
    //time = TreeNS::Kernel::Launch::centerOfMass(treeHandler->d_tree, particleHandler->d_particles,
    //                                            numParticlesLocal, true);
    //Logger(TIME) << "centerOfMass: " << time << " ms";

    /*Logger(INFO) << "sorting ...";
    time = TreeNS::Kernel::Launch::sort(treeHandler->d_tree, numParticlesLocal, numNodes, true);
    Logger(TIME) << "sort: " << time << " ms";


    //particleHandler->*/


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

    all_reduce(comm, boost::mpi::inplace_t<integer*>(helperHandler->d_integerBuffer), bins - 1, std::plus<integer>());

    Gravity::Kernel::Launch::calculateNewRange(subDomainKeyTreeHandler->d_subDomainKeyTree, helperHandler->d_helper,
                                               bins, aimedParticlesPerProcess);
    gpuErrorcheck(cudaMemset(&subDomainKeyTreeHandler->d_range[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses],
                             KEY_MAX, sizeof(keyType)));
    subDomainKeyTreeHandler->toHost();

    Logger(INFO) << "numProcesses: " << subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses;
    for(int i=0; i<=subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; i++) {
        Logger(INFO) << "range[" << i << "] = " << subDomainKeyTreeHandler->h_subDomainKeyTree->range[i];
    }

    int h_sum;
    cudaMemcpy(&h_sum, helperHandler->d_integerVal, sizeof(integer), cudaMemcpyDeviceToHost);
    Logger(INFO) << "h_sum = " << h_sum;

}

template <typename T>
integer Miluphpc::sendParticlesEntry(integer *sendLengths, integer *receiveLengths, T *entry) {

    boost::mpi::communicator comm;

    std::vector<boost::mpi::request> reqParticles;
    std::vector<boost::mpi::status> statParticles;

    integer reqCounter = 0;
    integer receiveOffset = 0;

    for (integer proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            if (proc == 0) {
                reqParticles.push_back(comm.isend(proc, 17, &entry[0], sendLengths[proc]));
            }
            else {
                reqParticles.push_back(comm.isend(proc, 17, &entry[subDomainKeyTreeHandler->h_procParticleCounter[proc-1]],
                                                  sendLengths[proc]));
            }
            statParticles.push_back(comm.recv(proc, 17, &helperHandler->d_realBuffer[0] + receiveOffset, receiveLengths[proc]));
            receiveOffset += receiveLengths[proc];
        }
    }

    boost::mpi::wait_all(reqParticles.begin(), reqParticles.end());

    integer offset = 0;
    for (integer i=0; i < subDomainKeyTreeHandler->h_subDomainKeyTree->rank; i++) {
        offset += subDomainKeyTreeHandler->h_procParticleCounter[i];
    }

    if (subDomainKeyTreeHandler->h_subDomainKeyTree->rank != 0) {
        if (offset > 0 && (subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] - offset) > 0) {
            HelperNS::Kernel::Launch::copyArray(&entry[0], &entry[subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] - offset], offset);
            //KernelHandler.copyArray(&entry[0], &entry[h_procCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] - offset], offset);
        }
        HelperNS::Kernel::Launch::copyArray(&buffer->d_realBuffer[0], &entry[offset], subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank]);
        HelperNS::Kernel::Launch::copyArray(&entry[0], &buffer->d_realBuffer[0], subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank]);
        //KernelHandler.copyArray(&d_tempArray_2[0], &entry[offset], subDomainKeyTreeHandler->d_h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank]);
        //KernelHandler.copyArray(&entry[0], &d_tempArray_2[0], subDomainKeyTreeHandler->d_h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank]);
    }

    HelperNS::Kernel::Launch::resetArray(&entry[subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank]],
                                         (real)0, numParticles-subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank]);
    HelperNS::Kernel::Launch::copyArray(&entry[subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank]],
                                        helperHandler->d_realBuffer, receiveOffset);
     //KernelHandler.resetFloatArray(&entry[h_procCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank]], 0, numParticles-h_procCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank]); //resetFloatArrayKernel(float *array, float value, int n)
    //KernelHandler.copyArray(&entry[h_procCounter[h_subDomainHandler->rank]], d_tempArray, receiveOffset);

    return receiveOffset + subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank];
}

void Miluphpc::compPseudoParticlesParallel() {

    DomainListNS::Kernel::Launch::lowestDomainList(subDomainKeyTreeHandler->d_subDomainKeyTree, treeHandler->d_tree,
                                                   domainListHandler->d_domainList,
                                                   lowestDomainListHandler->d_domainList, numParticles, numNodes);


    // zero domain list nodes (if needed)
    //KernelHandler.zeroDomainListNodes(d_domainListIndex, d_domainListIndices, d_lowestDomainListIndex,
    //                                  d_lowestDomainListIndices, d_x, d_y, d_z, d_mass, false); //TODO: needed to zero domain list nodes?

    // compute local pseudo particles (not for domain list nodes, at least not for the upper domain list nodes)
    Gravity::Kernel::Launch::compLocalPseudoParticles(treeHandler->d_tree, particleHandler->d_particles,
                                                      domainListHandler->d_domainList, numParticles);

    integer domainListIndex;
    integer lowestDomainListIndex;
    // x ----------------------------------------------------------------------------------------------
    cudaMemcpy(&domainListIndex, domainListHandler->d_domainListIndex, sizeof(integer), cudaMemcpyDeviceToHost);
    cudaMemcpy(&lowestDomainListIndex, lowestDomainListHandler->d_domainListIndex, sizeof(integer),
               cudaMemcpyDeviceToHost);
    Logger(INFO) << "domainListIndex: " << domainListIndex << " | lowestDomainListIndex: " << lowestDomainListIndex;

    //KernelHandler.prepareLowestDomainExchange(d_x, d_mass, d_tempArray, d_lowestDomainListIndices,
    //                                          d_lowestDomainListIndex, d_lowestDomainListKeys,
    //                                          d_lowestDomainListCounter, false);

    boost::mpi::communicator comm;

    gpuErrorcheck(cudaMemset(lowestDomainListHandler->d_domainListCounter, 0, sizeof(integer)));
    Gravity::Kernel::Launch::prepareLowestDomainExchange(particleHandler->d_particles, lowestDomainListHandler->d_domainList,
                                                         helperHandler->d_helper, Entry::x);

    HelperNS::sortArray(helperHandler->d_realBuffer, &helperHandler->d_realBuffer[DOMAIN_LIST_SIZE],
                        lowestDomainListHandler->d_domainListKeys, lowestDomainListHandler->d_sortedDomainListKeys,
                        domainListIndex);


    // share among processes
    //TODO: domainListIndex or lowestDomainListIndex?
    //MPI_Allreduce(MPI_IN_PLACE, &d_tempArray[DOMAIN_LIST_SIZE], domainListIndex, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    all_reduce(comm, boost::mpi::inplace_t<real*>(&helperHandler->d_realBuffer[DOMAIN_LIST_SIZE]), domainListIndex,
               std::plus<integer>());

    gpuErrorcheck(cudaMemset(lowestDomainListHandler->d_domainListCounter, 0, sizeof(integer)));

    Gravity::Kernel::Launch::updateLowestDomainListNodes(particleHandler->d_particles, lowestDomainListHandler->d_domainList,
                                                         helperHandler->d_helper, Entry::x);

    // y ----------------------------------------------------------------------------------------------

    //cudaMemcpy(&domainListIndex, d_domainListIndex, sizeof(int), cudaMemcpyDeviceToHost);
    //Logger(INFO) << "domainListIndex: " << domainListIndex;
    gpuErrorcheck(cudaMemset(lowestDomainListHandler->d_domainListCounter, 0, sizeof(integer)));
    Gravity::Kernel::Launch::prepareLowestDomainExchange(particleHandler->d_particles, lowestDomainListHandler->d_domainList,
                                                         helperHandler->d_helper, Entry::y);

    HelperNS::sortArray(helperHandler->d_realBuffer, &helperHandler->d_realBuffer[DOMAIN_LIST_SIZE],
                        lowestDomainListHandler->d_domainListKeys, lowestDomainListHandler->d_sortedDomainListKeys,
                        domainListIndex);


    // share among processes
    //TODO: domainListIndex or lowestDomainListIndex?
    //MPI_Allreduce(MPI_IN_PLACE, &d_tempArray[DOMAIN_LIST_SIZE], domainListIndex, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    all_reduce(comm, boost::mpi::inplace_t<real*>(&helperHandler->d_realBuffer[DOMAIN_LIST_SIZE]), domainListIndex,
               std::plus<integer>());

    gpuErrorcheck(cudaMemset(lowestDomainListHandler->d_domainListCounter, 0, sizeof(integer)));

    Gravity::Kernel::Launch::updateLowestDomainListNodes(particleHandler->d_particles, lowestDomainListHandler->d_domainList,
                                                         helperHandler->d_helper, Entry::y);

    // z ----------------------------------------------------------------------------------------------

    //cudaMemcpy(&domainListIndex, d_domainListIndex, sizeof(int), cudaMemcpyDeviceToHost);
    //Logger(INFO) << "domainListIndex: " << domainListIndex;
    gpuErrorcheck(cudaMemset(lowestDomainListHandler->d_domainListCounter, 0, sizeof(integer)));
    Gravity::Kernel::Launch::prepareLowestDomainExchange(particleHandler->d_particles, lowestDomainListHandler->d_domainList,
                                                         helperHandler->d_helper, Entry::z);

    HelperNS::sortArray(helperHandler->d_realBuffer, &helperHandler->d_realBuffer[DOMAIN_LIST_SIZE],
                        lowestDomainListHandler->d_domainListKeys, lowestDomainListHandler->d_sortedDomainListKeys,
                        domainListIndex);


    // share among processes
    //TODO: domainListIndex or lowestDomainListIndex?
    //MPI_Allreduce(MPI_IN_PLACE, &d_tempArray[DOMAIN_LIST_SIZE], domainListIndex, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    all_reduce(comm, boost::mpi::inplace_t<real*>(&helperHandler->d_realBuffer[DOMAIN_LIST_SIZE]), domainListIndex,
               std::plus<integer>());

    gpuErrorcheck(cudaMemset(lowestDomainListHandler->d_domainListCounter, 0, sizeof(integer)));

    Gravity::Kernel::Launch::updateLowestDomainListNodes(particleHandler->d_particles, lowestDomainListHandler->d_domainList,
                                                         helperHandler->d_helper, Entry::z);

    // m ----------------------------------------------------------------------------------------------

    //cudaMemcpy(&domainListIndex, d_domainListIndex, sizeof(int), cudaMemcpyDeviceToHost);
    //Logger(INFO) << "domainListIndex: " << domainListIndex;
    gpuErrorcheck(cudaMemset(lowestDomainListHandler->d_domainListCounter, 0, sizeof(integer)));
    Gravity::Kernel::Launch::prepareLowestDomainExchange(particleHandler->d_particles, lowestDomainListHandler->d_domainList,
                                                         helperHandler->d_helper, Entry::mass);

    HelperNS::sortArray(helperHandler->d_realBuffer, &helperHandler->d_realBuffer[DOMAIN_LIST_SIZE],
                        lowestDomainListHandler->d_domainListKeys, lowestDomainListHandler->d_sortedDomainListKeys,
                        domainListIndex);


    // share among processes
    //TODO: domainListIndex or lowestDomainListIndex?
    //MPI_Allreduce(MPI_IN_PLACE, &d_tempArray[DOMAIN_LIST_SIZE], domainListIndex, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    all_reduce(comm, boost::mpi::inplace_t<real*>(&helperHandler->d_realBuffer[DOMAIN_LIST_SIZE]), domainListIndex,
               std::plus<integer>());

    gpuErrorcheck(cudaMemset(lowestDomainListHandler->d_domainListCounter, 0, sizeof(integer)));

    Gravity::Kernel::Launch::updateLowestDomainListNodes(particleHandler->d_particles, lowestDomainListHandler->d_domainList,
                                                         helperHandler->d_helper, Entry::mass);

    // ------------------------------------------------------------------------------------------------

    Gravity::Kernel::Launch::compLowestDomainListNodes(particleHandler->d_particles,
                                                       lowestDomainListHandler->d_domainList);
    //end: for all entries!
    //KernelHandler.compLowestDomainListNodes(d_x, d_y, d_z, d_mass, d_lowestDomainListIndices, d_lowestDomainListIndex,
    //                                        d_lowestDomainListKeys, d_sortedLowestDomainListKeys,
    //                                        d_lowestDomainListCounter, false);


    Gravity::Kernel::Launch::compDomainListPseudoParticles(treeHandler->d_tree, particleHandler->d_particles,
                                                           domainListHandler->d_domainList, lowestDomainListHandler->d_domainList,
                                                           numParticles);

    // compute for the rest of the domain list nodes the values
    //KernelHandler.compDomainListPseudoParticlesPar(d_x, d_y, d_z, d_mass, d_child, d_index, numParticles, d_domainListIndices,
    //                                               d_domainListIndex, d_domainListLevels, d_lowestDomainListIndices, d_lowestDomainListIndex,
    //                                               false);

    Logger(INFO) << "Finished: compPseudoParticlesParallel()";


}

void Miluphpc::parallelForce() {

    /*//debugging
    KernelHandler.resetFloatArray(d_tempArray, 0.f, 2*numParticles, false);

    gpuErrorcheck(cudaMemset(d_domainListCounter, 0, sizeof(int)));

    //KernelHandler.domainListInfo(d_x, d_y, d_z, d_mass, d_child, d_index, numParticlesLocal,
    //                             d_domainListIndices, d_domainListIndex, d_domainListLevels, d_lowestDomainListIndices,
    //                             d_lowestDomainListIndex, d_subDomainHandler, false);

    //compTheta
    KernelHandler.compTheta(d_x, d_y, d_z, d_min_x, d_max_x, d_min_y, d_max_y, d_min_z, d_max_z, d_domainListIndex, d_domainListCounter,
                            d_domainListKeys, d_domainListIndices, d_domainListLevels, d_relevantDomainListIndices,
                            d_subDomainHandler, parameters.curveType, false);

    int relevantIndicesCounter;
    gpuErrorcheck(cudaMemcpy(&relevantIndicesCounter, d_domainListCounter, sizeof(int), cudaMemcpyDeviceToHost));

    Logger(INFO) << "relevantIndicesCounter: " << relevantIndicesCounter;

    //cudaMemcpy(&domainListIndex, d_relevantDomainListIndices, relevantIndicesCounter*sizeof(int), cudaMemcpyDeviceToHost);

    gpuErrorcheck(cudaMemcpy(h_min_x, d_min_x, sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrorcheck(cudaMemcpy(h_max_x, d_max_x, sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrorcheck(cudaMemcpy(h_min_y, d_min_y, sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrorcheck(cudaMemcpy(h_max_y, d_max_y, sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrorcheck(cudaMemcpy(h_min_z, d_min_z, sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrorcheck(cudaMemcpy(h_max_z, d_max_z, sizeof(float), cudaMemcpyDeviceToHost));

    float diam_x = std::abs(*h_max_x) + std::abs(*h_min_x);
    float diam_y = std::abs(*h_max_y) + std::abs(*h_min_y);
    float diam_z = std::abs(*h_max_z) + std::abs(*h_min_z);

    float diam = std::max({diam_x, diam_y, diam_z});
    Logger(INFO) << "diam: " << diam << "  (x = " << diam_x << ", y = " << diam_y << ", z = " << diam_z << ")";
    float theta = 0.5f;

    gpuErrorcheck(cudaMemset(d_domainListCounter, 0, sizeof(int)));
    int currentDomainListCounter;
    float massOfDomainListNode;
    for (int relevantIndex=0; relevantIndex<relevantIndicesCounter; relevantIndex++) {
        gpuErrorcheck(cudaMemcpy(&currentDomainListCounter, d_domainListCounter, sizeof(int), cudaMemcpyDeviceToHost));
        //gpuErrorcheck(cudaMemset(d_mutex, 0, sizeof(int)));
        //Logger(INFO) << "current value of domain list counter: " << currentDomainListCounter;

        KernelHandler.symbolicForce(relevantIndex, d_x, d_y, d_z, d_mass, d_min_x, d_max_x, d_min_y, d_max_y, d_min_z, d_max_z,
                                    d_child, d_domainListIndex, d_domainListKeys, d_domainListIndices, d_domainListLevels,
                                    d_domainListCounter, d_sendIndicesTemp, d_index, d_procCounter, d_subDomainHandler,
                                    numParticles, numNodes, diam, theta, d_mutex, d_relevantDomainListIndices, false);

        // removing duplicates
        // TODO: remove duplicates by overwriting same array with index of to send and afterwards remove empty entries
        int sendCountTemp;
        gpuErrorcheck(cudaMemcpy(&sendCountTemp, d_domainListCounter, sizeof(int), cudaMemcpyDeviceToHost));

        gpuErrorcheck(cudaMemset(d_domainListCounter, 0, sizeof(int)));
        KernelHandler.markDuplicates(d_sendIndicesTemp, d_x, d_y, d_z, d_mass, d_subDomainHandler, d_domainListCounter,
                                     sendCountTemp, false);
        int duplicatesCounter;
        gpuErrorcheck(cudaMemcpy(&duplicatesCounter, d_domainListCounter, sizeof(int), cudaMemcpyDeviceToHost));
        //Logger(INFO) << "duplicatesCounter: " << duplicatesCounter;
        //Logger(INFO) << "now resetting d_domainListCounter..";
        gpuErrorcheck(cudaMemset(d_domainListCounter, 0, sizeof(int)));
        //Logger(INFO) << "now removing duplicates..";
        KernelHandler.removeDuplicates(d_sendIndicesTemp, d_sendIndices, d_domainListCounter, sendCountTemp, false);
        int sendCount;
        gpuErrorcheck(cudaMemcpy(&sendCount, d_domainListCounter, sizeof(int), cudaMemcpyDeviceToHost));
        //Logger(INFO) << "sendCount: " << sendCount;
        // end: removing duplicates
    }

    gpuErrorcheck(cudaMemcpy(h_procCounter, d_procCounter, h_subDomainHandler->numProcesses*sizeof(int), cudaMemcpyDeviceToHost));

    int sendCountTemp;
    gpuErrorcheck(cudaMemcpy(&sendCountTemp, d_domainListCounter, sizeof(int), cudaMemcpyDeviceToHost));
    Logger(INFO) << "sendCountTemp: " << sendCountTemp;

    int newSendCount;
    gpuErrorcheck(cudaMemset(d_domainListCounter, 0, sizeof(int)));
    KernelHandler.markDuplicates(d_sendIndicesTemp, d_x, d_y, d_z, d_mass, d_subDomainHandler, d_domainListCounter,
                                 sendCountTemp, false);
    int duplicatesCounter;
    gpuErrorcheck(cudaMemcpy(&duplicatesCounter, d_domainListCounter, sizeof(int), cudaMemcpyDeviceToHost));
    Logger(INFO) << "duplicatesCounter: " << duplicatesCounter;
    Logger(INFO) << "now resetting d_domainListCounter..";
    gpuErrorcheck(cudaMemset(d_domainListCounter, 0, sizeof(int)));
    Logger(INFO) << "now removing duplicates..";
    KernelHandler.removeDuplicates(d_sendIndicesTemp, d_sendIndices, d_domainListCounter, sendCountTemp, false);
    int sendCount;
    gpuErrorcheck(cudaMemcpy(&sendCount, d_domainListCounter, sizeof(int), cudaMemcpyDeviceToHost));
    Logger(INFO) << "sendCount: " << sendCount;


    int *sendLengths;
    sendLengths = new int[h_subDomainHandler->numProcesses];
    sendLengths[h_subDomainHandler->rank] = 0;
    int *receiveLengths;
    receiveLengths = new int[h_subDomainHandler->numProcesses];
    receiveLengths[h_subDomainHandler->rank] = 0;

    for (int proc=0; proc < h_subDomainHandler->numProcesses; proc++) {
        if (proc != h_subDomainHandler->rank) {
            sendLengths[proc] = sendCount;
        }
    }

    int reqCounter = 0;
    MPI_Request reqMessageLengths[h_subDomainHandler->numProcesses-1];
    MPI_Status statMessageLengths[h_subDomainHandler->numProcesses-1];

    for (int proc=0; proc < h_subDomainHandler->numProcesses; proc++) {
        if (proc != h_subDomainHandler->rank) {
            MPI_Isend(&sendLengths[proc], 1, MPI_INT, proc, 17, MPI_COMM_WORLD, &reqMessageLengths[reqCounter]);
            MPI_Recv(&receiveLengths[proc], 1, MPI_INT, proc, 17, MPI_COMM_WORLD, &statMessageLengths[reqCounter]);
            reqCounter++;
        }
    }

    MPI_Waitall(h_subDomainHandler->numProcesses-1, reqMessageLengths, statMessageLengths);

    int totalReceiveLength = 0;
    for (int proc=0; proc<h_subDomainHandler->numProcesses; proc++) {
        if (proc != h_subDomainHandler->rank) {
            totalReceiveLength += receiveLengths[proc];
        }
    }

    Logger(INFO) << "totalReceiveLength = " << totalReceiveLength;

    int to_delete_leaf_0 = numParticlesLocal;
    int to_delete_leaf_1 = numParticlesLocal + totalReceiveLength; //+ sendCount;
    //cudaMemcpy(&d_to_delete_leaf[0], &h_procCounter[h_subDomainHandler->rank], sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(&d_to_delete_leaf[1], &to_delete_leaf_1, sizeof(int),
    //         cudaMemcpyHostToDevice);
    gpuErrorcheck(cudaMemcpy(&d_to_delete_leaf[0], &to_delete_leaf_0, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(&d_to_delete_leaf[1], &to_delete_leaf_1, sizeof(int),
                             cudaMemcpyHostToDevice));

    //copy values[indices] into d_tempArray (float)

    // x
    KernelHandler.collectSendIndices(d_sendIndices, d_x, d_tempArray, d_domainListCounter, sendCount);
    //debugging
    //KernelHandler.debug(d_x, d_y, d_z, d_mass, d_count, d_start, d_child, d_index, d_min_x, d_max_x, d_min_y, d_max_y,
    //                                    d_min_z, d_max_z, numParticlesLocal, numNodes, d_subDomainHandler, d_procCounter, d_tempArray,
    //                                    d_sortArray, d_sortArrayOut);
    exchangeParticleEntry(sendLengths, receiveLengths, d_x);
    // y
    KernelHandler.collectSendIndices(d_sendIndices, d_y, d_tempArray, d_domainListCounter, sendCount);
    exchangeParticleEntry(sendLengths, receiveLengths, d_y);
    // z
    KernelHandler.collectSendIndices(d_sendIndices, d_z, d_tempArray, d_domainListCounter, sendCount);
    exchangeParticleEntry(sendLengths, receiveLengths, d_z);

    // vx
    KernelHandler.collectSendIndices(d_sendIndices, d_vx, d_tempArray, d_domainListCounter, sendCount);
    exchangeParticleEntry(sendLengths, receiveLengths, d_vx);
    // vy
    KernelHandler.collectSendIndices(d_sendIndices, d_vy, d_tempArray, d_domainListCounter, sendCount);
    exchangeParticleEntry(sendLengths, receiveLengths, d_vy);
    // vz
    KernelHandler.collectSendIndices(d_sendIndices, d_vz, d_tempArray, d_domainListCounter, sendCount);
    exchangeParticleEntry(sendLengths, receiveLengths, d_vz);

    // ax
    KernelHandler.collectSendIndices(d_sendIndices, d_ax, d_tempArray, d_domainListCounter, sendCount);
    exchangeParticleEntry(sendLengths, receiveLengths, d_ax);
    // ay
    KernelHandler.collectSendIndices(d_sendIndices, d_ay, d_tempArray, d_domainListCounter, sendCount);
    exchangeParticleEntry(sendLengths, receiveLengths, d_ay);
    // az
    KernelHandler.collectSendIndices(d_sendIndices, d_az, d_tempArray, d_domainListCounter, sendCount);
    exchangeParticleEntry(sendLengths, receiveLengths, d_az);

    // mass
    KernelHandler.collectSendIndices(d_sendIndices, d_mass, d_tempArray, d_domainListCounter, sendCount);
    exchangeParticleEntry(sendLengths, receiveLengths, d_mass);

    //insert into tree // remember within to_delete_cell
    //remember index
    int indexBeforeInserting;
    gpuErrorcheck(cudaMemcpy(&indexBeforeInserting, d_index, sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrorcheck(cudaMemcpy(h_min_x, d_min_x, sizeof(float), cudaMemcpyDeviceToHost));


    //Logger(INFO) << "duplicateCounterCounter = " << duplicateCounterCounter;

    //KernelHandler.treeInfo(d_x, d_y, d_z, d_mass, d_count, d_start, d_child, d_index, d_min_x, d_max_x, d_min_y, d_max_y,
    //                       d_min_z, d_max_z, numParticlesLocal, numNodes, d_procCounter, d_subDomainHandler, d_sortArray,
    //                       d_sortArrayOut);

    Logger(INFO) << "Starting inserting particles...";
    KernelHandler.insertReceivedParticles(d_x, d_y, d_z, d_mass, d_count, d_start, d_child, d_index, d_min_x, d_max_x,
                                          d_min_y, d_max_y, d_min_z, d_max_z, d_to_delete_leaf, d_domainListIndices,
                                          d_domainListIndex, d_lowestDomainListIndices, d_lowestDomainListIndex,
            to_delete_leaf_1, numParticles, false);


    //KernelHandler.treeInfo(d_x, d_y, d_z, d_mass, d_count, d_start, d_child, d_index, d_min_x, d_max_x, d_min_y, d_max_y,
    //                       d_min_z, d_max_z, numParticlesLocal, numNodes, d_procCounter, d_subDomainHandler, d_sortArray,
    //                       d_sortArrayOut);

    int indexAfterInserting;
    gpuErrorcheck(cudaMemcpy(&indexAfterInserting, d_index, sizeof(int), cudaMemcpyDeviceToHost));

    Logger(INFO) << "to_delete_leaf[0] = " << to_delete_leaf_0
                 << " | " << "to_delete_leaf[1] = " << to_delete_leaf_1;

    Logger(INFO) << "to_delete_cell[0] = " << indexBeforeInserting << " | " << "to_delete_cell[1] = "
                 << indexAfterInserting;

    gpuErrorcheck(cudaMemcpy(&d_to_delete_cell[0], &indexBeforeInserting, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(&d_to_delete_cell[1], &indexAfterInserting, sizeof(int), cudaMemcpyHostToDevice));

    KernelHandler.centreOfMassReceivedParticles(d_x, d_y, d_z, d_mass, &d_to_delete_cell[0], &d_to_delete_cell[1],
                                                numParticlesLocal, false);

    Logger(INFO) << "Finished inserting received particles!";

    //debug

    //TODO: reset index on device? -> no, not working anymore
    //gpuErrorcheck(cudaMemset(d_index, indexBeforeInserting, sizeof(int)));

    float elapsedTime = 0.f;

    KernelHandler.sort(d_count, d_start, d_sorted, d_child, d_index, numParticles, numParticles, false); //TODO: numParticlesLocal or numParticles?

    //actual (local) force
    elapsedTime = KernelHandler.computeForces(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_ax, d_ay, d_az, d_mass, d_sorted, d_child,
                                              d_min_x, d_max_x, d_min_y, d_max_y, d_min_z, d_max_z, numParticlesLocal,
                                              numParticles, parameters.gravity, d_subDomainHandler, true); //TODO: numParticlesLocal or numParticles?


    // repairTree
    //TODO: necessary? Tree is build for every iteration
    KernelHandler.repairTree(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_ax, d_ay, d_az, d_mass, d_count, d_start, d_child,
                             d_index, d_min_x, d_max_x, d_min_y, d_max_y, d_min_z, d_max_z, d_to_delete_cell, d_to_delete_leaf,
                             d_domainListIndices, numParticlesLocal, numNodes, false);


    //gpuErrorcheck(cudaMemcpy(d_index, &indexBeforeInserting, sizeof(int), cudaMemcpyHostToDevice));
    //gpuErrorcheck(cudaMemcpy(&d_to_delete_leaf[0], &numParticlesLocal, sizeof(int), cudaMemcpyHostToDevice));

    return elapsedTime;*/

}