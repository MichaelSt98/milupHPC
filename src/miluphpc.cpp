#include "../include/miluphpc.h"

Miluphpc::Miluphpc(integer numParticles, integer numNodes) : numParticles(numParticles), numNodes(numNodes) {


    curveType = Curve::lebesgue;
    //curveType = Curve::hilbert;


    //TODO: how to distinguish/intialize numParticlesLocal vs numParticles
    //numParticlesLocal = numParticles/2;

    cuda::malloc(d_mutex, 1);
    helperHandler = new HelperHandler(numParticles);
    buffer = new HelperHandler(numParticles);
    particleHandler = new ParticleHandler(numParticles, numNodes);
    subDomainKeyTreeHandler = new SubDomainKeyTreeHandler();
    treeHandler = new TreeHandler(numParticles, numNodes);
    domainListHandler = new DomainListHandler(DOMAIN_LIST_SIZE);
    lowestDomainListHandler = new DomainListHandler(DOMAIN_LIST_SIZE);

    numParticlesLocal = numParticles/2; //+ subDomainKeyTreeHandler->h_subDomainKeyTree->rank * 5000;
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

    //gpuErrorcheck(cudaMemcpy(particleHandler->d_sml, particleHandler->h_sml, numParticlesLocal*sizeof(real),
    //                         cudaMemcpyHostToDevice));

    cuda::copy(particleHandler->h_sml, particleHandler->d_sml, numParticlesLocal, To::device);

    Logger(INFO) << "reduction: max:";
    HelperNS::reduceAndGlobalize(particleHandler->d_sml, helperHandler->d_realVal, numParticlesLocal, Reduction::max);
    Logger(INFO) << "reduction: min:";
    HelperNS::reduceAndGlobalize(particleHandler->d_sml, helperHandler->d_realVal, numParticlesLocal, Reduction::min);
    Logger(INFO) << "reduction: sum:";
    HelperNS::reduceAndGlobalize(particleHandler->d_sml, helperHandler->d_realVal, numParticlesLocal, Reduction::sum);

    particleHandler->copyDistribution(To::device, true, true);
}

void Miluphpc::diskModel() {

    real a = 1.0;
    real pi = 3.14159265;
    std::default_random_engine generator;
    std::uniform_real_distribution<real> distribution(1.5, 12.0);
    std::uniform_real_distribution<real> distribution_theta_angle(0.0, 2 * pi);

    real solarMass = 100000;

    switch (curveType) {

        case Curve::hilbert: {
            // loop through all particles
            for (int i = 0; i < numParticlesLocal; i++) {

                real theta_angle = distribution_theta_angle(generator);
                real r = distribution(generator);

                // set mass and position of particle
                if (subDomainKeyTreeHandler->h_subDomainKeyTree->rank == 0) {
                    if (i == 0) {
                        particleHandler->h_particles->mass[i] =
                                2 * solarMass / numParticlesLocal; //solarMass; //100000; 2 * solarMass / numParticles;
                        particleHandler->h_particles->x[i] = 0;
                        particleHandler->h_particles->y[i] = 0;
                        particleHandler->h_particles->z[i] = 0;
                    } else {
                        particleHandler->h_particles->mass[i] = 2 * solarMass / numParticlesLocal;
                        particleHandler->h_particles->x[i] = r * cos(theta_angle);
                        particleHandler->h_particles->y[i] = r * sin(theta_angle);
                        if (i % 2 == 0) {
                            particleHandler->h_particles->z[i] = i * 1e-7;//z[i] = i * 1e-7;
                        } else {
                            particleHandler->h_particles->z[i] = i * -1e-7;//z[i] = i * -1e-7;
                        }
                    }
                } else {
                    particleHandler->h_particles->mass[i] = 2 * solarMass / numParticlesLocal;
                    particleHandler->h_particles->x[i] =
                            (r + subDomainKeyTreeHandler->h_subDomainKeyTree->rank * 1.1e-1) *
                            cos(theta_angle) +
                            1.0e-2 * subDomainKeyTreeHandler->h_subDomainKeyTree->rank;
                    //y[i] = (r + h_subDomainHandler->rank * 1.3e-1) * sin(theta) + 1.1e-2*h_subDomainHandler->rank;
                    particleHandler->h_particles->y[i] =
                            (r + subDomainKeyTreeHandler->h_subDomainKeyTree->rank * 1.3e-1) *
                            sin(theta_angle) +
                            1.1e-2 * subDomainKeyTreeHandler->h_subDomainKeyTree->rank;

                    if (i % 2 == 0) {
                        //z[i] = i * 1e-7 * h_subDomainHandler->rank + 0.5e-7*h_subDomainHandler->rank;
                        particleHandler->h_particles->z[i] =
                                i * 1e-7 * subDomainKeyTreeHandler->h_subDomainKeyTree->rank +
                                0.5e-7 * subDomainKeyTreeHandler->h_subDomainKeyTree->rank;
                    } else {
                        //z[i] = i * -1e-7 * h_subDomainHandler->rank + 0.4e-7*h_subDomainHandler->rank;
                        particleHandler->h_particles->z[i] =
                                i * -1e-7 * subDomainKeyTreeHandler->h_subDomainKeyTree->rank
                                + 0.4e-7 * subDomainKeyTreeHandler->h_subDomainKeyTree->rank;
                    }
                }


                // set velocity of particle
                real rotation = 1;  // 1: clockwise   -1: counter-clockwise
                real v = sqrt(solarMass / (r));

                if (true/*i == 0*/) {
                    particleHandler->h_particles->vx[0] = 0.0;
                    particleHandler->h_particles->vy[0] = 0.0;
                    particleHandler->h_particles->vz[0] = 0.0;
                } else {
                    particleHandler->h_particles->vx[i] = rotation * v * sin(theta_angle);
                    //y_vel[i] = -rotation*v*cos(theta);
                    particleHandler->h_particles->vy[i] = -rotation * v * cos(theta_angle);
                    //z_vel[i] = 0.0;
                    particleHandler->h_particles->vz[i] = 0.0;
                }

                // set acceleration to zero
                particleHandler->h_particles->ax[i] = 0.0;
                particleHandler->h_particles->ay[i] = 0.0;
                particleHandler->h_particles->az[i] = 0.0;

                particleHandler->h_particles->sml[i] = 0.05; //theta_angle + subDomainKeyTreeHandler->h_subDomainKeyTree->rank;
                //if (i % 1000 == 0) {
                //    printf("maxSML: particleHandler->h_sml[%i] = %f\n", i, particleHandler->h_sml[i]);
                //}
            }
            break;
        }
        case Curve::lebesgue: {
            // loop through all particles
            for (int i = 0; i < numParticlesLocal; i++) {

                real theta_angle = distribution_theta_angle(generator);
                real r = distribution(generator);

                // set mass and position of particle
                if (subDomainKeyTreeHandler->h_subDomainKeyTree->rank == 0) {
                    if (i == 0) {
                        particleHandler->h_particles->mass[i] =
                                2 * solarMass / numParticlesLocal; //solarMass; //100000; 2 * solarMass / numParticles;
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
                } else {
                    particleHandler->h_particles->mass[i] = 2 * solarMass / numParticlesLocal;
                    particleHandler->h_particles->x[i] =
                            (r + subDomainKeyTreeHandler->h_subDomainKeyTree->rank * 1.1e-1) *
                            cos(theta_angle) +
                            1.0e-2 * subDomainKeyTreeHandler->h_subDomainKeyTree->rank;
                    //y[i] = (r + h_subDomainHandler->rank * 1.3e-1) * sin(theta) + 1.1e-2*h_subDomainHandler->rank;
                    particleHandler->h_particles->z[i] =
                            (r + subDomainKeyTreeHandler->h_subDomainKeyTree->rank * 1.3e-1) *
                            sin(theta_angle) +
                            1.1e-2 * subDomainKeyTreeHandler->h_subDomainKeyTree->rank;

                    if (i % 2 == 0) {
                        //z[i] = i * 1e-7 * h_subDomainHandler->rank + 0.5e-7*h_subDomainHandler->rank;
                        particleHandler->h_particles->y[i] =
                                i * 1e-7 * subDomainKeyTreeHandler->h_subDomainKeyTree->rank +
                                0.5e-7 * subDomainKeyTreeHandler->h_subDomainKeyTree->rank;
                    } else {
                        //z[i] = i * -1e-7 * h_subDomainHandler->rank + 0.4e-7*h_subDomainHandler->rank;
                        particleHandler->h_particles->y[i] =
                                i * -1e-7 * subDomainKeyTreeHandler->h_subDomainKeyTree->rank
                                + 0.4e-7 * subDomainKeyTreeHandler->h_subDomainKeyTree->rank;
                    }
                }


                // set velocity of particle
                real rotation = 1;  // 1: clockwise   -1: counter-clockwise
                real v = sqrt(solarMass / (r));

                if (true/*i == 0*/) {
                    particleHandler->h_particles->vx[0] = 0.0;
                    particleHandler->h_particles->vy[0] = 0.0;
                    particleHandler->h_particles->vz[0] = 0.0;
                } else {
                    particleHandler->h_particles->vx[i] = rotation * v * sin(theta_angle);
                    //y_vel[i] = -rotation*v*cos(theta);
                    particleHandler->h_particles->vz[i] = -rotation * v * cos(theta_angle);
                    //z_vel[i] = 0.0;
                    particleHandler->h_particles->vy[i] = 0.0;
                }

                // set acceleration to zero
                particleHandler->h_particles->ax[i] = 0.0;
                particleHandler->h_particles->ay[i] = 0.0;
                particleHandler->h_particles->az[i] = 0.0;

                particleHandler->h_particles->sml[i] = 0.05; //theta_angle + subDomainKeyTreeHandler->h_subDomainKeyTree->rank;
            }
            break;
        }
        default:
            Logger(INFO) << "Curve type not available!";
            exit(0);
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

    buffer->reset();
    helperHandler->reset();
    domainListHandler->reset();
    lowestDomainListHandler->reset();
    subDomainKeyTreeHandler->reset();

    Logger(TIME) << "resetArrays: " << time << " ms";

    Logger(INFO) << "computing bounding box ...";
    //TreeNS::computeBoundingBoxKernel(treeHandler->d_tree, particleHandler->d_particles, d_mutex, numNodes, 256);
    time = TreeNS::Kernel::Launch::computeBoundingBox(treeHandler->d_tree, particleHandler->d_particles, d_mutex,
                                                           numParticles, 256, true);
    Logger(TIME) << "computeBoundingBox: " << time << " ms";

    treeHandler->copy(To::host); //treeHandler->toHost();
    printf("min/max: x = (%f, %f), y = (%f, %f), z = (%f, %f)\n", *treeHandler->h_minX, *treeHandler->h_maxX,
           *treeHandler->h_minY, *treeHandler->h_maxY, *treeHandler->h_minZ, *treeHandler->h_maxZ);

    treeHandler->globalizeBoundingBox(Execution::device);
    treeHandler->copy(To::host); //treeHandler->toHost();
    printf("min/max: x = (%f, %f), y = (%f, %f), z = (%f, %f)\n", *treeHandler->h_minX, *treeHandler->h_maxX,
           *treeHandler->h_minY, *treeHandler->h_maxY, *treeHandler->h_minZ, *treeHandler->h_maxZ);

    Logger(INFO) << "building tree ...";
    //time = TreeNS::Kernel::Launch::buildTree(treeHandler->d_tree, particleHandler->d_particles, numParticles,
    //                                         numParticles, true);
    Logger(TIME) << "buildTree: " << time << " ms";

    Logger(INFO) << "center of mass ...";
    //time = TreeNS::Kernel::Launch::centerOfMass(treeHandler->d_tree, particleHandler->d_particles,
    //                                            numParticles, true);
    Logger(TIME) << "centerOfMass: " << time << " ms";

    Logger(INFO) << "sorting ...";
    //time = TreeNS::Kernel::Launch::sort(treeHandler->d_tree, numParticles, numNodes, true);
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

void Miluphpc::initBarnesHut() {

    real time;

    Logger(INFO) << "Starting ...";

    Logger(INFO) << "initialize particle distribution ...";
    initDistribution();

    time = TreeNS::Kernel::Launch::computeBoundingBox(treeHandler->d_tree, particleHandler->d_particles, d_mutex,
                                                      numParticlesLocal, 256, true);
    treeHandler->copy(To::host); //treeHandler->toHost();
    printf("min/max: x = (%f, %f), y = (%f, %f), z = (%f, %f)\n", *treeHandler->h_minX, *treeHandler->h_maxX,
           *treeHandler->h_minY, *treeHandler->h_maxY, *treeHandler->h_minZ, *treeHandler->h_maxZ);
    treeHandler->globalizeBoundingBox(Execution::device);

    if (false/*parameters.loadBalancing*/) {
        dynamicLoadDistribution();
    }
    else {
        fixedLoadDistribution();
    }
}

void Miluphpc::barnesHut() {

    if (false/*parameters.loadBalancing*/) {
        Logger(INFO) << "load balancing ...";
        if (true/*step == 0 || step % parameters.loadBalancingInterval == 0*/) {
            dynamicLoadDistribution();
        }
    }

    real time;

    Logger(INFO) << "resetting (device) arrays ...";
    time = Kernel::Launch::resetArrays(treeHandler->d_tree, particleHandler->d_particles, d_mutex, numParticles,
                                       numNodes, true);
    helperHandler->reset();
    buffer->reset();
    domainListHandler->reset();
    lowestDomainListHandler->reset();
    subDomainKeyTreeHandler->reset();

    Logger(TIME) << "resetArrays: " << time << " ms";

    Logger(INFO) << "computing bounding box ...";

    time = TreeNS::Kernel::Launch::computeBoundingBox(treeHandler->d_tree, particleHandler->d_particles, d_mutex,
                                                      numParticlesLocal, 256, true);
    Logger(TIME) << "computeBoundingBox: " << time << " ms";

    treeHandler->globalizeBoundingBox(Execution::device);
    treeHandler->copy(To::host); //treeHandler->toHost();

    printf("bounding box: x = (%f, %f), y = (%f, %f), z = (%f, %f)\n", *treeHandler->h_minX, *treeHandler->h_maxX,
           *treeHandler->h_minY, *treeHandler->h_maxY, *treeHandler->h_minZ, *treeHandler->h_maxZ);


    //subDomainKeyTreeHandler->toHost(); //TODO: needed?
    //for (int proc=0; proc < subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
    //    printf("[rank %i] before particlesPerProcess(): subDomainKeyTreeHandler->h_procParticleCounter[%i] = %i\n", subDomainKeyTreeHandler->h_rank,
    //           proc, subDomainKeyTreeHandler->h_procParticleCounter[proc]);
    //}

    SubDomainKeyTreeNS::Kernel::Launch::particlesPerProcess(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                            treeHandler->d_tree, particleHandler->d_particles,
                                                            numParticlesLocal, numNodes, curveType);

    //subDomainKeyTreeHandler->toHost(); //TODO: needed?
    //for (int proc=0; proc < subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
    //    printf("[rank %i] before particlesPerProcess(): subDomainKeyTreeHandler->h_procParticleCounter[%i] = %i\n", subDomainKeyTreeHandler->h_rank,
    //           proc, subDomainKeyTreeHandler->h_procParticleCounter[proc]);
    //}

    SubDomainKeyTreeNS::Kernel::Launch::markParticlesProcess(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                             treeHandler->d_tree, particleHandler->d_particles,
                                                             numParticlesLocal, numNodes,
                                                             helperHandler->d_integerBuffer, curveType);

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

    subDomainKeyTreeHandler->copy(To::host, true, true); //TODO: needed?

    integer *sendLengths;
    sendLengths = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    sendLengths[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] = 0;
    integer *receiveLengths;
    receiveLengths = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    receiveLengths[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] = 0;

    for (int proc=0; proc < subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        //printf("[rank %i] subDomainKeyTreeHandler->h_procParticleCounter[%i] = %i\n", subDomainKeyTreeHandler->h_rank,
        //       proc, subDomainKeyTreeHandler->h_procParticleCounter[proc]);
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            sendLengths[proc] = subDomainKeyTreeHandler->h_procParticleCounter[proc];
        }
    }
    mpi::messageLengths(subDomainKeyTreeHandler->h_subDomainKeyTree, sendLengths, receiveLengths);

    //for (int proc = 0; proc < subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
    //    Logger(INFO) << "(numParticles) loadBalancing: sendLengths[" << proc << "] = " << sendLengths[proc];
    //    Logger(INFO) << "(numParticles) loadBalancing: receiveLengths[" << proc << "] = " << receiveLengths[proc];
    //}

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

    //Logger(INFO) << "numParticlesLocal before sending = " << numParticlesLocal;
    numParticlesLocal = sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_mass);
    //Logger(INFO) << "numParticlesLocal after sending = " << numParticlesLocal;

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

    // DEBUG
    //particleHandler->distributionToHost(true, false);
    //keyType *d_keys;
    //gpuErrorcheck(cudaMalloc((void**)&d_keys, numParticlesLocal*sizeof(keyType)));
    //SubDomainKeyTreeNS::Kernel::Launch::getParticleKeys(subDomainKeyTreeHandler->d_subDomainKeyTree,
    //                                                    treeHandler->d_tree, particleHandler->d_particles,
    //                                                    d_keys, 21, numParticlesLocal, curveType);
    //gpuErrorcheck(cudaFree(d_keys));
    // end: DEBUG


    Logger(TIME) << "\tSorting for process: " << elapsedTimeSorting << " ms";

    Logger(INFO) << "building domain list ...";
    time = DomainListNS::Kernel::Launch::createDomainList(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                          domainListHandler->d_domainList, MAX_LEVEL,
                                                          curveType);
    Logger(TIME) << "createDomainList: " << time << " ms";

    integer domainListLength;
    cudaMemcpy(&domainListLength, domainListHandler->d_domainListIndex, sizeof(integer), cudaMemcpyDeviceToHost);
    Logger(INFO) << "domainListLength = " << domainListLength;

    integer treeIndexBeforeBuildingTree;
    gpuErrorcheck(cudaMemcpy(&treeIndexBeforeBuildingTree, treeHandler->d_index, sizeof(integer),
                             cudaMemcpyDeviceToHost));
    Logger(INFO) << "treeIndexBeforeBuildingTree: " << treeIndexBeforeBuildingTree;

    Logger(INFO) << "building tree ...";
    time = TreeNS::Kernel::Launch::buildTree(treeHandler->d_tree, particleHandler->d_particles, numParticlesLocal,
                                             numParticles, true);
    Logger(TIME) << "buildTree: " << time << " ms";

    integer treeIndex;
    gpuErrorcheck(cudaMemcpy(&treeIndex, treeHandler->d_index, sizeof(integer),
                             cudaMemcpyDeviceToHost));

    Logger(INFO) << "numParticlesLocal: " << numParticlesLocal;
    Logger(INFO) << "numParticles: " << numParticles;
    Logger(INFO) << "numNodes: " << numNodes;
    Logger(INFO) << "treeIndex: " << treeIndex;
    integer numParticlesSum = numParticlesLocal;
    boost::mpi::communicator comm;
    all_reduce(comm, boost::mpi::inplace_t<integer*>(&numParticlesSum), 1, std::plus<integer>());
    Logger(INFO) << "numParticlesSum: " << numParticlesSum;
    //ParticlesNS::Kernel::Launch::info(particleHandler->d_particles, numParticlesLocal, numParticles, treeIndex);

    Logger(INFO) << "building domain tree ...";
    time = SubDomainKeyTreeNS::Kernel::Launch::buildDomainTree(treeHandler->d_tree, particleHandler->d_particles,
                                                         domainListHandler->d_domainList, numParticlesLocal,
                                                         numNodes);
    Logger(TIME) << "build(Domain)Tree: " << time << " ms";




    //DomainListNS::Kernel::Launch::info(particleHandler->d_particles, domainListHandler->d_domainList);

    //DomainListNS::Kernel::Launch::info(particleHandler->d_particles, domainListHandler->d_domainList);
    //DomainListNS::Kernel::Launch::info(particleHandler->d_particles, domainListHandler->d_domainList, lowestDomainListHandler->d_domainList);

    compPseudoParticlesParallel();

    //DomainListNS::Kernel::Launch::info(particleHandler->d_particles, domainListHandler->d_domainList);
    //DomainListNS::Kernel::Launch::info(particleHandler->d_particles, domainListHandler->d_domainList, lowestDomainListHandler->d_domainList);
    //DomainListNS::Kernel::Launch::info(particleHandler->d_particles, lowestDomainListHandler->d_domainList);

    /*time = TreeNS::Kernel::Launch::centerOfMass(treeHandler->d_tree, particleHandler->d_particles,
                                                        numParticlesLocal, true);

    TreeNS::Kernel::Launch::sort(treeHandler->d_tree, numParticles, numParticles, false);
    //KernelHandler.sort(d_count, d_start, d_sorted, d_child, d_index, numParticles, numParticles, false); //TODO: numParticlesLocal or numParticles?

    //actual (local) force
    integer warp = 32;
    integer stackSize = 64;
    integer blockSize = 256;
    Gravity::Kernel::Launch::computeForces(treeHandler->d_tree, particleHandler->d_particles, numParticlesLocal, numParticles,
                                           blockSize, warp, stackSize);*/

    //ParticlesNS::Kernel::Launch::info(particleHandler->d_particles, numParticlesLocal, numParticles, numParticles);

    parallelForce();

    Gravity::Kernel::Launch::update(particleHandler->d_particles, numParticlesLocal, 0.001, 1.);

}

void Miluphpc::sph() {

    // in order to build tree, domain list, ...
    barnesHut();

    int sphInsertOffset = 50000;

    integer *d_sphSendCount;
    integer *d_alreadyInserted;
    cuda::malloc(d_alreadyInserted, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses);
    cuda::malloc(d_sphSendCount, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses);

    cuda::set(d_sphSendCount, 0, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses);
    //gpuErrorcheck(cudaMemset(d_sphSendCount, 0, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses*sizeof(integer)));
    gpuErrorcheck(cudaMemset(helperHandler->d_integerBuffer, -1, numParticles*sizeof(integer)));

    SPH::Kernel::Launch::particles2Send(subDomainKeyTreeHandler->d_subDomainKeyTree, treeHandler->d_tree,
                                        particleHandler->d_particles, domainListHandler->d_domainList,
                                        lowestDomainListHandler->d_domainList, 21, helperHandler->d_integerBuffer,
                                        d_sphSendCount, d_alreadyInserted, sphInsertOffset,
                                        numParticlesLocal, numParticles, numNodes, 1e-1, curveType);

    //KernelHandler.sphParticles2Send(numParticlesLocal, numParticles, numNodes, 1e-1,
    //                                d_x, d_y, d_z, d_min_x, d_max_x, d_min_y, d_max_y, d_min_z, d_max_z,
    //                                d_subDomainHandler, d_domainListIndex, d_domainListKeys,
    //                                d_domainListIndices, d_domainListLevels,
    //                                d_lowestDomainListIndices, d_lowestDomainListIndex,
    //                                d_lowestDomainListKeys, d_lowestDomainListLevels, 1e-1, 21, parameters.curveType,
    //                                d_sortArray, d_sphSendCount, d_alreadyInserted, sphInsertOffset, false);

    integer totalSendCount = 0;

    integer *particles2SendSPH;
    particles2SendSPH = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    gpuErrorcheck(cudaMemcpy(particles2SendSPH, d_sphSendCount, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses*sizeof(integer),
                             cudaMemcpyDeviceToHost));
    for (int i=0; i<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; i++) {
        Logger(INFO) << "particles2SendSPH[" << i << "] = " << particles2SendSPH[i];
        totalSendCount += particles2SendSPH[i];
    }

    Logger(INFO) << "totalSendCount: " << totalSendCount;

    int *particles2ReceiveSPH;
    particles2ReceiveSPH = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    particles2ReceiveSPH[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] = 0;

    mpi::messageLengths(subDomainKeyTreeHandler->h_subDomainKeyTree, particles2SendSPH, particles2ReceiveSPH);

    integer totalReceiveLength = 0;
    for (int i=0; i<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; i++) {
        Logger(INFO) << "particles2ReceiveSPH[" << i << "]: " << particles2ReceiveSPH[i];
        totalReceiveLength += particles2ReceiveSPH[i];
    }

    integer particles2SendOffset = 0;
    for (int i=0; i<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; i++) {
        SPH::Kernel::Launch::collectSendIndices(&helperHandler->d_integerBuffer[i*sphInsertOffset], &buffer->d_integerBuffer[particles2SendOffset], particles2SendSPH[i]);
        //KernelHandler.collectSendIndicesSPH(&d_sortArray[i*sphInsertOffset], &d_sortArrayOut[particles2SendOffset], particles2SendSPH[i], false);
        particles2SendOffset += particles2SendSPH[i];
    }

    Logger(INFO) << "SPH: totalReceiveLength: " << totalReceiveLength;

    treeHandler->h_toDeleteLeaf[0] = numParticlesLocal;
    treeHandler->h_toDeleteLeaf[1] = numParticlesLocal + totalReceiveLength;
    gpuErrorcheck(cudaMemcpy(treeHandler->d_toDeleteLeaf, treeHandler->h_toDeleteLeaf, 2*sizeof(integer),
                             cudaMemcpyHostToDevice));

    // x
    SPH::Kernel::Launch::collectSendEntries(subDomainKeyTreeHandler->d_subDomainKeyTree, particleHandler->d_x,
                                            buffer->d_realBuffer, buffer->d_integerBuffer, d_sphSendCount,
                                            totalSendCount, sphInsertOffset);

    SPH::exchangeParticleEntry(subDomainKeyTreeHandler->h_subDomainKeyTree, particleHandler->d_x, buffer->d_realBuffer,
                               particles2SendSPH, particles2ReceiveSPH, numParticlesLocal);

    // y
    SPH::Kernel::Launch::collectSendEntries(subDomainKeyTreeHandler->d_subDomainKeyTree, particleHandler->d_y,
                                            buffer->d_realBuffer, buffer->d_integerBuffer, d_sphSendCount,
                                            totalSendCount, sphInsertOffset);
    SPH::exchangeParticleEntry(subDomainKeyTreeHandler->h_subDomainKeyTree, particleHandler->d_y, buffer->d_realBuffer,
                               particles2SendSPH, particles2ReceiveSPH, numParticlesLocal);

    // z
    SPH::Kernel::Launch::collectSendEntries(subDomainKeyTreeHandler->d_subDomainKeyTree, particleHandler->d_z,
                                            buffer->d_realBuffer, buffer->d_integerBuffer, d_sphSendCount,
                                            totalSendCount, sphInsertOffset);
    SPH::exchangeParticleEntry(subDomainKeyTreeHandler->h_subDomainKeyTree, particleHandler->d_z, buffer->d_realBuffer,
                               particles2SendSPH, particles2ReceiveSPH, numParticlesLocal);

    // mass
    SPH::Kernel::Launch::collectSendEntries(subDomainKeyTreeHandler->d_subDomainKeyTree, particleHandler->d_mass,
                                            buffer->d_realBuffer, buffer->d_integerBuffer, d_sphSendCount,
                                            totalSendCount, sphInsertOffset);
    SPH::exchangeParticleEntry(subDomainKeyTreeHandler->h_subDomainKeyTree, particleHandler->d_mass, buffer->d_realBuffer,
                               particles2SendSPH, particles2ReceiveSPH, numParticlesLocal);
    //TODO: exchange particle entry via MPI for all entries!
    // density, pressure, ...
    //SPH::Kernel::Launch::collectSendEntries(subDomainKeyTreeHandler->d_subDomainKeyTree, particleHandler->d_x,
    //                                        buffer->d_realBuffer, helperHandler->d_integerBuffer, d_sphSendCount,
    //                                        totalSendCount, sphInsertOffset);
    //SPH::exchangeParticleEntry(subDomainKeyTreeHandler->h_subDomainKeyTree, particleHandler->d_x, buffer->d_realBuffer,
    //                           particles2SendSPH, particles2ReceiveSPH, numParticlesLocal);

    //TODO: need to update numParticlesLocal?

    delete [] particles2SendSPH;
    delete [] particles2ReceiveSPH;

    //buffer->reset();
    //helperHandler->reset();

    //gpuErrorcheck(cudaMemcpy(treeHandler->d_index, &numParticlesLocal, sizeof(integer), cudaMemcpyHostToDevice));

    gpuErrorcheck(cudaMemcpy(&treeHandler->h_toDeleteNode[0], treeHandler->d_index, sizeof(integer),
                             cudaMemcpyDeviceToHost));

    Logger(INFO) << "SPH: Starting inserting particles...";
    Logger(INFO) << "SPH: treeHandler->h_toDeleteLeaf[0]: " << treeHandler->h_toDeleteLeaf[0];
    Logger(INFO) << "SPH: treeHandler->h_toDeleteLeaf[1]: " << treeHandler->h_toDeleteLeaf[1];
    ParticlesNS::Kernel::Launch::info(particleHandler->d_particles, 0, treeHandler->h_toDeleteLeaf[0], treeHandler->h_toDeleteLeaf[1]);
    //if (treeHandler->h_toDeleteLeaf[1] > treeHandler->h_toDeleteLeaf[0]) {
    Gravity::Kernel::Launch::insertReceivedParticles(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                     treeHandler->d_tree,
                                                     particleHandler->d_particles, domainListHandler->d_domainList,
                                                     lowestDomainListHandler->d_domainList,
                                                     treeHandler->h_toDeleteLeaf[1],
                                                     numParticles);
    //}


    //int indexAfterInserting;
    gpuErrorcheck(cudaMemcpy(&treeHandler->h_toDeleteNode[1], treeHandler->d_index, sizeof(integer), cudaMemcpyDeviceToHost));

    gpuErrorcheck(cudaMemcpy(treeHandler->d_toDeleteNode, treeHandler->h_toDeleteNode, 2*sizeof(integer),
                             cudaMemcpyHostToDevice));

    Logger(INFO) << "treeHandler->h_toDeleteNode[0]: " << treeHandler->h_toDeleteNode[0];
    Logger(INFO) << "treeHandler->h_toDeleteNode[1]: " << treeHandler->h_toDeleteNode[1];



    //real time = SPH::Kernel::Launch::fixedRadiusNN(treeHandler->d_tree, particleHandler->d_particles, particleHandler->d_nnl,
    //                                   numParticlesLocal, numParticles, numNodes);

    real time = SPH::Kernel::Launch::fixedRadiusNN_Test(treeHandler->d_tree, particleHandler->d_particles, particleHandler->d_nnl,
                                                   numParticlesLocal, numParticles, numNodes);

    Logger(TIME) << "SPH: fixedRadiusNN: " << time << " ms";



    SPH::Kernel::Launch::info(treeHandler->d_tree, particleHandler->d_particles, helperHandler->d_helper,
                              numParticlesLocal, numParticles, numNodes);

    cuda::free(d_sphSendCount);
    //gpuErrorcheck(cudaFree(d_sphSendCount));
    gpuErrorcheck(cudaFree(d_alreadyInserted));

}

void Miluphpc::fixedLoadDistribution() {
    for (int i=0; i<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; i++) {
        subDomainKeyTreeHandler->h_subDomainKeyTree->range[i] = i * (1UL << 63)/(subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses);
    }
    subDomainKeyTreeHandler->h_subDomainKeyTree->range[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses] = KEY_MAX;

    for (int i=0; i<=subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; i++) {
        printf("range[%i] = %lu\n", i, subDomainKeyTreeHandler->h_subDomainKeyTree->range[i]);
    }

    subDomainKeyTreeHandler->copy(To::device, true, true);
}

void Miluphpc::dynamicLoadDistribution() {

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
                                            bins, numParticlesLocal, curveType);

    all_reduce(comm, boost::mpi::inplace_t<integer*>(helperHandler->d_integerBuffer), bins - 1, std::plus<integer>());

    Gravity::Kernel::Launch::calculateNewRange(subDomainKeyTreeHandler->d_subDomainKeyTree, helperHandler->d_helper,
                                               bins, aimedParticlesPerProcess, curveType);
    gpuErrorcheck(cudaMemset(&subDomainKeyTreeHandler->d_range[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses],
                             KEY_MAX, sizeof(keyType)));
    subDomainKeyTreeHandler->copy(To::host, true, true);

    Logger(INFO) << "numProcesses: " << subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses;
    for(int i=0; i<=subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; i++) {
        Logger(INFO) << "range[" << i << "] = " << subDomainKeyTreeHandler->h_subDomainKeyTree->range[i];
    }

    int h_sum;
    cudaMemcpy(&h_sum, helperHandler->d_integerVal, sizeof(integer), cudaMemcpyDeviceToHost);
    Logger(INFO) << "h_sum = " << h_sum;

    // debug
    /*for (int i=0; i<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; i++) {
        subDomainKeyTreeHandler->h_subDomainKeyTree->range[i] = i * (1UL << 63)/(subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses);
    }
    subDomainKeyTreeHandler->h_subDomainKeyTree->range[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses] = KEY_MAX;

    subDomainKeyTreeHandler->toDevice();*/
    // end: debug

}


void Miluphpc::exchangeParticleEntry(integer *sendLengths, integer *receiveLengths, real *entry) {

    boost::mpi::communicator comm;

    std::vector<boost::mpi::request> reqParticles;
    std::vector<boost::mpi::status> statParticles;

    integer reqCounter = 0;
    integer receiveOffset = 0;

    for (integer proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            reqParticles.push_back(comm.isend(proc, 17, &helperHandler->d_realBuffer[0], sendLengths[proc])); //TODO: buffer or helperHandler
            statParticles.push_back(comm.recv(proc, 17, &entry[numParticlesLocal] + receiveOffset,
                                              receiveLengths[proc]));
            receiveOffset += receiveLengths[proc];
        }
    }

    boost::mpi::wait_all(reqParticles.begin(), reqParticles.end());

    //int offset = 0;
    //for (int i=0; i < h_subDomainHandler->rank; i++) {
    //    offset += h_procCounter[h_subDomainHandler->rank];
    //}

}

//TODO: combine and generalize sendParticlesEntry and exchangeParticlesEntry
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
                reqParticles.push_back(comm.isend(proc, 17,
                                                  &entry[subDomainKeyTreeHandler->h_procParticleCounter[proc-1]],
                                                  sendLengths[proc]));
            }
            statParticles.push_back(comm.recv(proc, 17, &helperHandler->d_realBuffer[0] + receiveOffset,
                                              receiveLengths[proc]));
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

    //Logger(INFO) << "numParticlesLocal = " << receiveOffset << " + " << subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank];
    return receiveOffset + subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank];
}

void Miluphpc::compPseudoParticlesParallel() {

    DomainListNS::Kernel::Launch::lowestDomainList(subDomainKeyTreeHandler->d_subDomainKeyTree, treeHandler->d_tree,
                                                   domainListHandler->d_domainList,
                                                   lowestDomainListHandler->d_domainList, numParticles, numNodes);

    //TODO: needed to zero domain list nodes?
    //Gravity::Kernel::Launch::zeroDomainListNodes(particleHandler->d_particles, domainListHandler->d_domainList,
    //                                             lowestDomainListHandler->d_domainList);

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
    MPI_Allreduce(MPI_IN_PLACE, &helperHandler->d_realBuffer[DOMAIN_LIST_SIZE], domainListIndex, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    //all_reduce(comm, boost::mpi::inplace_t<real*>(&helperHandler->d_realBuffer[DOMAIN_LIST_SIZE]), domainListIndex,
    //           std::plus<real>());

    gpuErrorcheck(cudaMemset(lowestDomainListHandler->d_domainListCounter, 0, sizeof(integer)));

    Gravity::Kernel::Launch::updateLowestDomainListNodes(particleHandler->d_particles, lowestDomainListHandler->d_domainList,
                                                         helperHandler->d_helper, Entry::x);

    //DomainListNS::Kernel::Launch::info(particleHandler->d_particles, domainListHandler->d_domainList);

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
               std::plus<real>());

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
               std::plus<real>());

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
               std::plus<real>());

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

    //DomainListNS::Kernel::Launch::info(particleHandler->d_particles, domainListHandler->d_domainList);

    // compute for the rest of the domain list nodes the values
    //KernelHandler.compDomainListPseudoParticlesPar(d_x, d_y, d_z, d_mass, d_child, d_index, numParticles, d_domainListIndices,
    //                                               d_domainListIndex, d_domainListLevels, d_lowestDomainListIndices, d_lowestDomainListIndex,
    //                                               false);

    Logger(INFO) << "Finished: compPseudoParticlesParallel()";


}

void Miluphpc::parallelForce() {

    //debugging
    HelperNS::Kernel::Launch::resetArray(helperHandler->d_realBuffer, (real)0, numParticles);
    //KernelHandler.resetFloatArray(d_tempArray, 0.f, 2*numParticles, false);

    gpuErrorcheck(cudaMemset(domainListHandler->d_domainListCounter, 0, sizeof(integer)));

    //compTheta
    Gravity::Kernel::Launch::compTheta(subDomainKeyTreeHandler->d_subDomainKeyTree, treeHandler->d_tree,
                                       particleHandler->d_particles, domainListHandler->d_domainList,
                                       helperHandler->d_helper, curveType);

    integer relevantIndicesCounter;
    gpuErrorcheck(cudaMemcpy(&relevantIndicesCounter, domainListHandler->d_domainListCounter, sizeof(integer),
                             cudaMemcpyDeviceToHost));

    Logger(INFO) << "relevantIndicesCounter: " << relevantIndicesCounter;

    //cudaMemcpy(&domainListIndex, d_relevantDomainListIndices, relevantIndicesCounter*sizeof(int), cudaMemcpyDeviceToHost);

    treeHandler->copy(To::host); //treeHandler->toHost();

    real diam_x = std::abs(*treeHandler->h_maxX) + std::abs(*treeHandler->h_minX);
#if DIM > 1
    real diam_y = std::abs(*treeHandler->h_maxY) + std::abs(*treeHandler->h_minY);
#if DIM == 3
    real diam_z = std::abs(*treeHandler->h_maxZ) + std::abs(*treeHandler->h_minZ);
#endif
#endif

#if DIM == 1
    real diam = diam_x;
#elif DIM == 2
    real diam = std::max({diam_x, diam_y});
#else
    real diam = std::max({diam_x, diam_y, diam_z});
    Logger(INFO) << "diam: " << diam << "  (x = " << diam_x << ", y = " << diam_y << ", z = " << diam_z << ")";
#endif

    real theta_ = theta; //0.5f;

    gpuErrorcheck(cudaMemset(domainListHandler->d_domainListCounter, 0, sizeof(integer)));
    integer currentDomainListCounter;
    real massOfDomainListNode;
    for (integer relevantIndex=0; relevantIndex</*1*/relevantIndicesCounter; relevantIndex++) {
        gpuErrorcheck(cudaMemcpy(&currentDomainListCounter, domainListHandler->d_domainListCounter, sizeof(integer),
                                 cudaMemcpyDeviceToHost));
        //gpuErrorcheck(cudaMemset(d_mutex, 0, sizeof(int)));
        //Logger(INFO) << "current value of domain list counter: " << currentDomainListCounter;

        integer treeIndex;
        gpuErrorcheck(cudaMemcpy(&treeIndex, treeHandler->d_index, sizeof(integer),
                                 cudaMemcpyDeviceToHost));
        //Logger(INFO) << "treeIndex = " << treeIndex;
        //Logger(INFO) << "symbolicForce ...";
        Gravity::Kernel::Launch::symbolicForce(subDomainKeyTreeHandler->d_subDomainKeyTree, treeHandler->d_tree,
                                               particleHandler->d_particles, domainListHandler->d_domainList,
                                               helperHandler->d_helper, diam, theta_, numParticles, numNodes,
                                               relevantIndex, curveType);

        // removing duplicates
        // TODO: remove duplicates by overwriting same array with index of to send and afterwards remove empty entries
        int sendCountTemp;
        gpuErrorcheck(cudaMemcpy(&sendCountTemp, domainListHandler->d_domainListCounter, sizeof(integer),
                                 cudaMemcpyDeviceToHost));

        gpuErrorcheck(cudaMemset(domainListHandler->d_domainListCounter, 0, sizeof(integer)));

        // CHECK for indices
        CudaUtils::Kernel::Launch::markDuplicates(helperHandler->d_integerBuffer, domainListHandler->d_domainListCounter,
                                                  sendCountTemp);
        // CHECK for indices and entries!
        //CudaUtils::Kernel::Launch::markDuplicates(helperHandler->d_integerBuffer, particleHandler->d_x,
        //                                          particleHandler->d_y, domainListHandler->d_domainListCounter,
        //                                          sendCountTemp);


        integer duplicatesCounter;
        gpuErrorcheck(cudaMemcpy(&duplicatesCounter, domainListHandler->d_domainListCounter, sizeof(integer),
                                 cudaMemcpyDeviceToHost));
        //Logger(INFO) << "duplicatesCounter: " << duplicatesCounter;
        //Logger(INFO) << "now resetting d_domainListCounter..";
        gpuErrorcheck(cudaMemset(domainListHandler->d_domainListCounter, 0, sizeof(integer)));
        //Logger(INFO) << "now removing duplicates..";
        CudaUtils::Kernel::Launch::removeDuplicates(helperHandler->d_integerBuffer, buffer->d_integerBuffer,
                                                    domainListHandler->d_domainListCounter, sendCountTemp);
        integer sendCount;
        gpuErrorcheck(cudaMemcpy(&sendCount, domainListHandler->d_domainListCounter, sizeof(integer),
                                 cudaMemcpyDeviceToHost));
        //Logger(INFO) << "sendCount: " << sendCount;
        // end: removing duplicates
    }

    subDomainKeyTreeHandler->copy(To::host, true, true);
    //gpuErrorcheck(cudaMemcpy(h_procCounter, d_procCounter, h_subDomainHandler->numProcesses*sizeof(int), cudaMemcpyDeviceToHost));

    int sendCountTemp;
    gpuErrorcheck(cudaMemcpy(&sendCountTemp, domainListHandler->d_domainListCounter, sizeof(integer),
                             cudaMemcpyDeviceToHost));
    Logger(INFO) << "sendCountTemp: " << sendCountTemp;

    int newSendCount;
    gpuErrorcheck(cudaMemset(domainListHandler->d_domainListCounter, 0, sizeof(integer)));

    DomainListNS::Kernel::Launch::info(particleHandler->d_particles, domainListHandler->d_domainList);

    // CHECK indices
    //CudaUtils::Kernel::Launch::markDuplicates(helperHandler->d_integerBuffer, domainListHandler->d_domainListCounter,
    //                                          sendCountTemp);
    // CHECK indices and entries
    //CudaUtils::Kernel::Launch::markDuplicates(helperHandler->d_integerBuffer, particleHandler->d_x,
    //                                          particleHandler->d_y, domainListHandler->d_domainListCounter,
    //                                          sendCountTemp);
    // CHECK indices and entries and DEBUG!
    CudaUtils::Kernel::Launch::markDuplicates(helperHandler->d_integerBuffer, particleHandler->d_x,
                                              particleHandler->d_y, domainListHandler->d_domainListCounter,
                                              treeHandler->d_child, sendCountTemp);

    int duplicatesCounter;
    gpuErrorcheck(cudaMemcpy(&duplicatesCounter, domainListHandler->d_domainListCounter, sizeof(integer),
                             cudaMemcpyDeviceToHost));
    Logger(INFO) << "duplicatesCounter: " << duplicatesCounter;
    Logger(INFO) << "now resetting d_domainListCounter..";
    gpuErrorcheck(cudaMemset(domainListHandler->d_domainListCounter, 0, sizeof(integer)));
    Logger(INFO) << "now removing duplicates..";
    CudaUtils::Kernel::Launch::removeDuplicates(helperHandler->d_integerBuffer, buffer->d_integerBuffer,
                                                domainListHandler->d_domainListCounter, sendCountTemp);
    //KernelHandler.removeDuplicates(d_sendIndicesTemp, d_sendIndices, d_domainListCounter, sendCountTemp, false);
    int sendCount;
    gpuErrorcheck(cudaMemcpy(&sendCount, domainListHandler->d_domainListCounter, sizeof(integer), cudaMemcpyDeviceToHost));
    Logger(INFO) << "sendCount: " << sendCount;

    // DEBUG
    //gpuErrorcheck(cudaMemset(buffer->d_integerVal, 0, sizeof(integer)));
    //CudaUtils::Kernel::Launch::findDuplicates(buffer->d_integerBuffer, particleHandler->d_x,
    //                                          particleHandler->d_y, buffer->d_integerVal, sendCount);
    //CudaUtils::Kernel::Launch::findDuplicates(buffer->d_integerBuffer, particleHandler->d_x,
    //                                          particleHandler->d_y, buffer->d_integerVal, sendCount);
    //integer duplicates;
    //gpuErrorcheck(cudaMemcpy(&duplicates, buffer->d_integerVal, sizeof(integer), cudaMemcpyDeviceToHost));
    //Logger(INFO) << "duplicates: " << duplicates;
    // end: DEBUG

    integer *sendLengths;
    sendLengths = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    sendLengths[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] = 0;
    integer *receiveLengths;
    receiveLengths = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    receiveLengths[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] = 0;

    for (int proc=0; proc < subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            sendLengths[proc] = sendCount;
        }
    }

    mpi::messageLengths(subDomainKeyTreeHandler->h_subDomainKeyTree, sendLengths, receiveLengths);

    int totalReceiveLength = 0;
    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            totalReceiveLength += receiveLengths[proc];
        }
    }

    Logger(INFO) << "totalReceiveLength = " << totalReceiveLength;

    treeHandler->h_toDeleteLeaf[0] = numParticlesLocal;
    treeHandler->h_toDeleteLeaf[1] = numParticlesLocal + totalReceiveLength; //+ sendCount;
    //cudaMemcpy(&d_to_delete_leaf[0], &h_procCounter[h_subDomainHandler->rank], sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(&d_to_delete_leaf[1], &to_delete_leaf_1, sizeof(int),
    //         cudaMemcpyHostToDevice);
    gpuErrorcheck(cudaMemcpy(treeHandler->d_toDeleteLeaf, treeHandler->h_toDeleteLeaf, 2*sizeof(integer),
                             cudaMemcpyHostToDevice));
    //gpuErrorcheck(cudaMemcpy(&treeHandler->d_toDeleteLeaf[1] &treeHandler->h_toDeleteLeaf[1], sizeof(integer),
    //                         cudaMemcpyHostToDevice));

    //copy values[indices] into d_tempArray (float)


    //CudaUtils::Kernel::Launch::checkValues(buffer->d_integerBuffer, particleHandler->d_x, particleHandler->d_y,
    //                                       particleHandler->d_z, sendCount);

    // x
    CudaUtils::Kernel::Launch::collectValues(buffer->d_integerBuffer, particleHandler->d_x, helperHandler->d_realBuffer,
                                             sendCount);
    exchangeParticleEntry(sendLengths, receiveLengths, particleHandler->d_x);
#if DIM > 1
    // y
    CudaUtils::Kernel::Launch::collectValues(buffer->d_integerBuffer, particleHandler->d_y, helperHandler->d_realBuffer,
                                             sendCount);
    exchangeParticleEntry(sendLengths, receiveLengths, particleHandler->d_y);
#if DIM == 3
    // z
    CudaUtils::Kernel::Launch::collectValues(buffer->d_integerBuffer, particleHandler->d_z, helperHandler->d_realBuffer,
                                             sendCount);
    exchangeParticleEntry(sendLengths, receiveLengths, particleHandler->d_z);
#endif
#endif

    // vx
    CudaUtils::Kernel::Launch::collectValues(buffer->d_integerBuffer, particleHandler->d_vx, helperHandler->d_realBuffer,
                                             sendCount);
    exchangeParticleEntry(sendLengths, receiveLengths, particleHandler->d_vx);
#if DIM > 1
    // vy
    CudaUtils::Kernel::Launch::collectValues(buffer->d_integerBuffer, particleHandler->d_vy, helperHandler->d_realBuffer,
                                             sendCount);
    exchangeParticleEntry(sendLengths, receiveLengths, particleHandler->d_vy);
#if DIM == 3
    // vz
    CudaUtils::Kernel::Launch::collectValues(buffer->d_integerBuffer, particleHandler->d_vz, helperHandler->d_realBuffer,
                                             sendCount);
    exchangeParticleEntry(sendLengths, receiveLengths, particleHandler->d_vz);
#endif
#endif

    // x
    CudaUtils::Kernel::Launch::collectValues(buffer->d_integerBuffer, particleHandler->d_ax, helperHandler->d_realBuffer,
                                             sendCount);
    exchangeParticleEntry(sendLengths, receiveLengths, particleHandler->d_ax);
#if DIM > 1
    // y
    CudaUtils::Kernel::Launch::collectValues(buffer->d_integerBuffer, particleHandler->d_ay, helperHandler->d_realBuffer,
                                             sendCount);
    exchangeParticleEntry(sendLengths, receiveLengths, particleHandler->d_ay);
#if DIM == 3
    // z
    CudaUtils::Kernel::Launch::collectValues(buffer->d_integerBuffer, particleHandler->d_az, helperHandler->d_realBuffer,
                                             sendCount);
    exchangeParticleEntry(sendLengths, receiveLengths, particleHandler->d_az);
#endif
#endif

    // mass
    CudaUtils::Kernel::Launch::collectValues(buffer->d_integerBuffer, particleHandler->d_mass, helperHandler->d_realBuffer,
                                             sendCount);
    exchangeParticleEntry(sendLengths, receiveLengths, particleHandler->d_mass);

    //insert into tree // remember within to_delete_cell
    //remember index
    //int indexBeforeInserting;
    gpuErrorcheck(cudaMemcpy(&treeHandler->h_toDeleteNode[0], treeHandler->d_index, sizeof(integer),
                             cudaMemcpyDeviceToHost));
    //gpuErrorcheck(cudaMemcpy(h_min_x, d_min_x, sizeof(float), cudaMemcpyDeviceToHost));


    //Logger(INFO) << "duplicateCounterCounter = " << duplicateCounterCounter;

    //KernelHandler.treeInfo(d_x, d_y, d_z, d_mass, d_count, d_start, d_child, d_index, d_min_x, d_max_x, d_min_y, d_max_y,
    //                       d_min_z, d_max_z, numParticlesLocal, numNodes, d_procCounter, d_subDomainHandler, d_sortArray,
    //                       d_sortArrayOut);

    Logger(INFO) << "Starting inserting particles...";
    Logger(INFO) << "treeHandler->h_toDeleteLeaf[0]: " << treeHandler->h_toDeleteLeaf[0];
    Logger(INFO) << "treeHandler->h_toDeleteLeaf[1]: " << treeHandler->h_toDeleteLeaf[1];
    Gravity::Kernel::Launch::insertReceivedParticles(subDomainKeyTreeHandler->d_subDomainKeyTree, treeHandler->d_tree,
                                                     particleHandler->d_particles, domainListHandler->d_domainList,
                                                     lowestDomainListHandler->d_domainList, treeHandler->h_toDeleteLeaf[1],
                                                     numParticles);



    //KernelHandler.insertReceivedParticles(d_x, d_y, d_z, d_mass, d_count, d_start, d_child, d_index, d_min_x, d_max_x,
    //                                      d_min_y, d_max_y, d_min_z, d_max_z, d_to_delete_leaf, d_domainListIndices,
    //                                      d_domainListIndex, d_lowestDomainListIndices, d_lowestDomainListIndex,
    //        to_delete_leaf_1, numParticles, false);


    //KernelHandler.treeInfo(d_x, d_y, d_z, d_mass, d_count, d_start, d_child, d_index, d_min_x, d_max_x, d_min_y, d_max_y,
    //                       d_min_z, d_max_z, numParticlesLocal, numNodes, d_procCounter, d_subDomainHandler, d_sortArray,
    //                       d_sortArrayOut);

    //int indexAfterInserting;
    gpuErrorcheck(cudaMemcpy(&treeHandler->h_toDeleteNode[1], treeHandler->d_index, sizeof(integer), cudaMemcpyDeviceToHost));

    //Logger(INFO) << "to_delete_leaf[0] = " << to_delete_leaf_0
    //             << " | " << "to_delete_leaf[1] = " << to_delete_leaf_1;

    //Logger(INFO) << "to_delete_cell[0] = " << indexBeforeInserting << " | " << "to_delete_cell[1] = "
    //             << indexAfterInserting;

    //gpuErrorcheck(cudaMemcpy(&treeHandler->d_toDeleteNode[0], &indexBeforeInserting, sizeof(integer),
    //                         cudaMemcpyHostToDevice));
    //gpuErrorcheck(cudaMemcpy(&treeHandler->d_toDeleteNode[1], &indexAfterInserting, sizeof(integer),
    //                         cudaMemcpyHostToDevice));

    gpuErrorcheck(cudaMemcpy(treeHandler->d_toDeleteNode, treeHandler->h_toDeleteNode, 2*sizeof(integer),
                                                   cudaMemcpyHostToDevice));

    Logger(INFO) << "treeHandler->h_toDeleteNode[0]: " << treeHandler->h_toDeleteNode[0];
    Logger(INFO) << "treeHandler->h_toDeleteNode[1]: " << treeHandler->h_toDeleteNode[1];

    Gravity::Kernel::Launch::centreOfMassReceivedParticles(particleHandler->d_particles, &treeHandler->d_toDeleteNode[0],
                                                           &treeHandler->d_toDeleteNode[1], numParticlesLocal);
    //KernelHandler.centreOfMassReceivedParticles(d_x, d_y, d_z, d_mass, &d_to_delete_cell[0], &d_to_delete_cell[1],
    //                                            numParticlesLocal, false);

    Logger(INFO) << "Finished inserting received particles!";

    //debug

    //TODO: reset index on device? -> no, not working anymore
    //gpuErrorcheck(cudaMemset(d_index, indexBeforeInserting, sizeof(int)));

    float elapsedTime = 0.f;


    TreeNS::Kernel::Launch::sort(treeHandler->d_tree, /*treeHandler->h_toDeleteNode[1]*/numParticlesLocal, numParticles, false);
    //KernelHandler.sort(d_count, d_start, d_sorted, d_child, d_index, numParticles, numParticles, false); //TODO: numParticlesLocal or numParticles?

    //actual (local) force
    integer warp = 32;
    integer stackSize = 64;
    integer blockSize = 256;
    Gravity::Kernel::Launch::computeForces(treeHandler->d_tree, particleHandler->d_particles, treeHandler->h_toDeleteNode[1]/*numParticlesLocal*/, numParticles,
                                           blockSize, warp, stackSize);

    //elapsedTime = KernelHandler.computeForces(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_ax, d_ay, d_az, d_mass, d_sorted, d_child,
    //                                          d_min_x, d_max_x, d_min_y, d_max_y, d_min_z, d_max_z, numParticlesLocal,
    //                                          numParticles, parameters.gravity, d_subDomainHandler, true); //TODO: numParticlesLocal or numParticles?


    // repairTree
    //TODO: necessary? Tree is build for every iteration

    Gravity::Kernel::Launch::repairTree(treeHandler->d_tree, particleHandler->d_particles,
                                        domainListHandler->d_domainList, numParticlesLocal, numNodes);


    //KernelHandler.repairTree(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_ax, d_ay, d_az, d_mass, d_count, d_start, d_child,
    //                         d_index, d_min_x, d_max_x, d_min_y, d_max_y, d_min_z, d_max_z, d_to_delete_cell, d_to_delete_leaf,
    //                         d_domainListIndices, numParticlesLocal, numNodes, false);


    //gpuErrorcheck(cudaMemcpy(d_index, &indexBeforeInserting, sizeof(int), cudaMemcpyHostToDevice));
    //gpuErrorcheck(cudaMemcpy(&d_to_delete_leaf[0], &numParticlesLocal, sizeof(int), cudaMemcpyHostToDevice));

    //return elapsedTime;

}

void Miluphpc::particles2file(HighFive::DataSet *pos, HighFive::DataSet *vel, HighFive::DataSet *key) {

    std::vector<std::vector<real>> x, v; // two dimensional vector for 3D vector data
    std::vector<keyType> k; // one dimensional vector holding particle keys

    particleHandler->copyDistribution(To::host, true, false);

    keyType *d_keys;
    cuda::malloc(d_keys, numParticlesLocal);

    //TreeNS::Kernel::Launch::getParticleKeys(treeHandler->d_tree, particleHandler->d_particles,
    //                                        d_keys, 21, numParticlesLocal, curveType);
    SubDomainKeyTreeNS::Kernel::Launch::getParticleKeys(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                        treeHandler->d_tree, particleHandler->d_particles,
                                                        d_keys, 21, numParticlesLocal, curveType);
    //SubDomainKeyTreeNS::Kernel::Launch::getParticleKeys(subDomainKeyTreeHandler->d_subDomainKeyTree,
    //                                                    treeHandler->d_tree, particleHandler->d_particles,
    //                                                    d_keys, 21, numParticlesLocal, curveType);


    keyType *h_keys;
    h_keys = new keyType[numParticlesLocal];
    gpuErrorcheck(cudaMemcpy(h_keys, d_keys, numParticlesLocal * sizeof(keyType), cudaMemcpyDeviceToHost));

    integer keyProc;

    for (int i=0; i<numParticlesLocal; i++) {
        x.push_back({particleHandler->h_x[i], particleHandler->h_y[i], particleHandler->h_z[i]});
        v.push_back({particleHandler->h_vx[i], particleHandler->h_vy[i], particleHandler->h_vz[i]});
        k.push_back(h_keys[i]);
    }

    gpuErrorcheck(cudaFree(d_keys));
    delete [] h_keys;

    // receive buffer
    int procN[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];

    // send buffer
    int sendProcN[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++){
        sendProcN[proc] = subDomainKeyTreeHandler->h_subDomainKeyTree->rank == proc ? numParticlesLocal : 0;
    }

    MPI_Allreduce(sendProcN, procN, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses, MPI_INT,
                  MPI_MAX, MPI_COMM_WORLD);

    std::size_t nOffset = 0;
    // count total particles on other processes
    for (int proc = 0; proc < subDomainKeyTreeHandler->h_subDomainKeyTree->rank; proc++){
        nOffset += procN[proc];
    }
    Logger(DEBUG) << "Offset to write to datasets: " << std::to_string(nOffset);

    // write to asscoiated datasets in h5 file
    // only working when load balancing has been completed and even number of particles
    pos->select({nOffset, 0},
                {std::size_t(numParticlesLocal), std::size_t(3)}).write(x);
    vel->select({nOffset, 0},
                {std::size_t(numParticlesLocal), std::size_t(3)}).write(v);
    key->select({nOffset}, {std::size_t(numParticlesLocal)}).write(k);

}
