#include "../include/miluphpc.h"

Miluphpc::Miluphpc(integer numParticles, integer numNodes) : numParticles(numParticles), numNodes(numNodes) {


    curveType = Curve::lebesgue;
    //curveType = Curve::hilbert;


    //TODO: how to distinguish/intialize numParticlesLocal vs numParticles
    //numParticlesLocal = numParticles/2;

    cuda::malloc(d_mutex, 1);
    helperHandler = new HelperHandler(numParticles);
    buffer = new HelperHandler(numNodes);
    particleHandler = new ParticleHandler(numParticles, numNodes);
    subDomainKeyTreeHandler = new SubDomainKeyTreeHandler();
    treeHandler = new TreeHandler(numParticles, numNodes);
    domainListHandler = new DomainListHandler(DOMAIN_LIST_SIZE);
    lowestDomainListHandler = new DomainListHandler(DOMAIN_LIST_SIZE);

    // testing
    cuda::malloc(d_particles2SendIndices, numParticles);
    cuda::malloc(d_pseudoParticles2SendIndices, numParticles);
    cuda::malloc(d_pseudoParticles2SendLevels, numParticles);
    cuda::malloc(d_pseudoParticles2ReceiveLevels, numParticles);

    cuda::malloc(d_particles2SendCount, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses);
    cuda::malloc(d_pseudoParticles2SendCount, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses);
    // end: testing

    numParticlesLocal = numParticles/2; //+ subDomainKeyTreeHandler->h_subDomainKeyTree->rank * 5000;
    Logger(INFO) << "numParticlesLocal = " << numParticlesLocal;

}

Miluphpc::~Miluphpc() {

    delete helperHandler;
    delete buffer;
    delete particleHandler;
    delete subDomainKeyTreeHandler;
    delete treeHandler;

    // testing
    cuda::free(d_particles2SendIndices);
    cuda::free(d_pseudoParticles2SendIndices);
    cuda::free(d_pseudoParticles2SendLevels);
    cuda::free(d_pseudoParticles2ReceiveLevels);

    cuda::free(d_particles2SendCount);
    cuda::free(d_pseudoParticles2SendCount);
    // end: testing

}

void Miluphpc::distributionFromFile() {

    HighFive::File file("N100000seed1885245432.h5", HighFive::File::ReadOnly);

    // containers to be filled
    real m;
    std::vector<std::vector<real>> x, v;

    // read datasets from file
    HighFive::DataSet mass = file.getDataSet("/m");
    HighFive::DataSet pos = file.getDataSet("/x");
    HighFive::DataSet vel = file.getDataSet("/v");

    // read data
    mass.read(m);
    pos.read(x);
    vel.read(v);

    integer ppp = 100000/subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses;

    // each process reads only a portion of the init file
    // j denotes global particle list index, i denotes local index
    for (int j=subDomainKeyTreeHandler->h_subDomainKeyTree->rank * ppp; j < (subDomainKeyTreeHandler->h_subDomainKeyTree->rank+1)*ppp; j++) {
        int i = j - subDomainKeyTreeHandler->h_subDomainKeyTree->rank * ppp;

        particleHandler->h_particles->mass[i] = m;
        particleHandler->h_particles->x[i] = x[j][0];
        particleHandler->h_particles->y[i] = x[j][1];
        particleHandler->h_particles->z[i] = x[j][2];
        particleHandler->h_particles->vx[i] = v[j][0];
        particleHandler->h_particles->vy[i] = v[j][1];
        particleHandler->h_particles->vz[i] = v[j][2];

    }

}

void Miluphpc::initDistribution(ParticleDistribution::Type particleDistribution) {

    /*switch(particleDistribution) {
        case ParticleDistribution::disk:
            diskModel();
            break;
        case ParticleDistribution::plummer:
            //
            break;
        default:
            diskModel();
    }*/

    distributionFromFile();


    //cuda::copy(particleHandler->h_sml, particleHandler->d_sml, numParticlesLocal, To::device);


    cuda::copy(particleHandler->h_sml, particleHandler->d_sml, numParticlesLocal, To::device);
    particleHandler->copyDistribution(To::device, true, true);

    // TESTING
    //Logger(INFO) << "reduction: max:";
    //HelperNS::reduceAndGlobalize(particleHandler->d_sml, helperHandler->d_realVal, numParticlesLocal, Reduction::max);
    //Logger(INFO) << "reduction: min:";
    //HelperNS::reduceAndGlobalize(particleHandler->d_sml, helperHandler->d_realVal, numParticlesLocal, Reduction::min);
    //Logger(INFO) << "reduction: sum:";
    //HelperNS::reduceAndGlobalize(particleHandler->d_sml, helperHandler->d_realVal, numParticlesLocal, Reduction::sum);
    // end: TESTING

    // DEBUG
    //gpuErrorcheck(cudaMemset(buffer->d_integerVal, 0, sizeof(integer)));
    //CudaUtils::Kernel::Launch::findDuplicateEntries(particleHandler->d_x, particleHandler->d_y, buffer->d_integerVal,
    //                                          numParticlesLocal);
    //integer duplicates;
    //gpuErrorcheck(cudaMemcpy(&duplicates, buffer->d_integerVal, sizeof(integer), cudaMemcpyDeviceToHost));
    //Logger(INFO) << "duplicates: " << duplicates;
    // end: DEBUG
}

void Miluphpc::diskModel() {

    real a = 1.0;
    real pi = 3.14159265;
    std::default_random_engine generator;
    std::uniform_real_distribution<real> distribution(4.0, 10.0);
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
                            5.0e-2 * subDomainKeyTreeHandler->h_subDomainKeyTree->rank;
                    //y[i] = (r + h_subDomainHandler->rank * 1.3e-1) * sin(theta) + 1.1e-2*h_subDomainHandler->rank;
                    particleHandler->h_particles->y[i] =
                            (r + subDomainKeyTreeHandler->h_subDomainKeyTree->rank * 1.3e-1) *
                            sin(theta_angle) +
                            5.1e-2 * subDomainKeyTreeHandler->h_subDomainKeyTree->rank;

                    if (i % 2 == 0) {
                        //z[i] = i * 1e-7 * h_subDomainHandler->rank + 0.5e-7*h_subDomainHandler->rank;
                        particleHandler->h_particles->z[i] =
                                i * 1e-3 * subDomainKeyTreeHandler->h_subDomainKeyTree->rank +
                                0.5e-5 * subDomainKeyTreeHandler->h_subDomainKeyTree->rank;
                    } else {
                        //z[i] = i * -1e-7 * h_subDomainHandler->rank + 0.4e-7*h_subDomainHandler->rank;
                        particleHandler->h_particles->z[i] =
                                i * -1e-3 * subDomainKeyTreeHandler->h_subDomainKeyTree->rank
                                + 0.4e-5 * subDomainKeyTreeHandler->h_subDomainKeyTree->rank;
                    }
                }


                // set velocity of particle
                real rotation = 1;  // 1: clockwise   -1: counter-clockwise
                real v = sqrt(solarMass / (r));

                if (i == 0) {
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
                        particleHandler->h_particles->mass[i] = solarMass; //2 * solarMass / numParticlesLocal; //solarMass; //100000; 2 * solarMass / numParticles;
                        particleHandler->h_particles->x[i] = 0;
                        particleHandler->h_particles->y[i] = 0;
                        particleHandler->h_particles->z[i] = 0;
                    } else {
                        particleHandler->h_particles->mass[i] = 0.01 * solarMass / numParticlesLocal;
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
                    particleHandler->h_particles->mass[i] = 0.01 * solarMass / numParticlesLocal;
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

                if (i == 0) {
                    particleHandler->h_particles->vx[0] = 0.0;
                    particleHandler->h_particles->vy[0] = 0.0;
                    particleHandler->h_particles->vz[0] = 0.0;
                } else {
                    particleHandler->h_particles->vx[i] = rotation * v * sin(theta_angle); //v * sin(theta_angle);
                    //y_vel[i] = -rotation*v*cos(theta);
                    particleHandler->h_particles->vz[i] = -rotation * v * cos(theta_angle); //v * cos(theta_angle);
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


/*MaterialHandler materialHandler("config/material.cfg");

    for (int i=0; i<materialHandler.numMaterials; i++) {
        materialHandler.h_materials[i].info();
}*/

void Miluphpc::rhs() {

    Logger(INFO) << "Miluphpc::rhs()";

    Logger(INFO) << "rhs:: reset()";
    reset();
    Logger(INFO) << "rhs: boundingBox()";
    boundingBox();
    Logger(INFO) << "rhs: assignParticles()";
    assignParticles();
    Logger(INFO) << "rhs: tree()";
    tree();
    Logger(INFO) << "rhs: pseudoParticles()";
    pseudoParticles();
    Logger(INFO) << "rhs: gravity()";
    gravity();

    // TODO: move integration to integrator
    //Gravity::Kernel::Launch::update(particleHandler->d_particles, numParticlesLocal, 0.001, 1.);

}

void Miluphpc::loadDistribution() {

    real time;

    Logger(INFO) << "Starting ...";

    Logger(INFO) << "initialize particle distribution ...";
    initDistribution();

    time = TreeNS::Kernel::Launch::computeBoundingBox(treeHandler->d_tree, particleHandler->d_particles, d_mutex,
                                                      numParticlesLocal, 256, true);

    treeHandler->globalizeBoundingBox(Execution::device);
    treeHandler->copy(To::host);

    if (false/*parameters.loadBalancing*/) {
        dynamicLoadDistribution();
    }
    else {
        fixedLoadDistribution();
    }
}

real Miluphpc::reset() {
    real time;
    // START: resetting arrays, variables, buffers, ...
    Logger(INFO) << "resetting (device) arrays ...";
    time = Kernel::Launch::resetArrays(treeHandler->d_tree, particleHandler->d_particles, d_mutex, numParticles,
                                       numNodes, true);
    helperHandler->reset();
    buffer->reset();
    domainListHandler->reset();
    lowestDomainListHandler->reset();
    subDomainKeyTreeHandler->reset();

    cuda::set(treeHandler->d_child, -1, POW_DIM * numNodes);
    Logger(TIME) << "resetArrays: " << time << " ms";
    // END: resetting arrays, variables, buffers, ...
    return time;
}

real Miluphpc::boundingBox() {
    real time;
    // START: computing bounding box/borders
    Logger(INFO) << "computing bounding box ...";
    time = TreeNS::Kernel::Launch::computeBoundingBox(treeHandler->d_tree, particleHandler->d_particles, d_mutex,
                                                      numParticlesLocal, 256, true);
    treeHandler->globalizeBoundingBox(Execution::device);
    treeHandler->copy(To::host);
    Logger(TIME) << "computeBoundingBox: " << time << " ms";
    printf("bounding box: x = (%f, %f), y = (%f, %f), z = (%f, %f)\n", *treeHandler->h_minX, *treeHandler->h_maxX,
           *treeHandler->h_minY, *treeHandler->h_maxY, *treeHandler->h_minZ, *treeHandler->h_maxZ);

    // END: computing bounding box/borders
    return time;
}

template <typename T>
real Miluphpc::arrangeParticleEntries(T *entry, T *temp) {
    real time;
    time = HelperNS::sortArray(entry, temp, helperHandler->d_integerBuffer, buffer->d_integerBuffer, numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(entry, temp, numParticlesLocal);
    return time;
}

real Miluphpc::assignParticles() {
    // START: assigning particles to correct MPI process
    SubDomainKeyTreeNS::Kernel::Launch::particlesPerProcess(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                            treeHandler->d_tree, particleHandler->d_particles,
                                                            numParticlesLocal, numNodes, curveType);


    SubDomainKeyTreeNS::Kernel::Launch::markParticlesProcess(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                             treeHandler->d_tree, particleHandler->d_particles,
                                                             numParticlesLocal, numNodes,
                                                             helperHandler->d_integerBuffer, curveType);

    arrangeParticleEntries(particleHandler->d_x, helperHandler->d_realBuffer);
    arrangeParticleEntries(particleHandler->d_vx, helperHandler->d_realBuffer);
    arrangeParticleEntries(particleHandler->d_ax, helperHandler->d_realBuffer);
#if DIM > 1
    arrangeParticleEntries(particleHandler->d_y, helperHandler->d_realBuffer);
    arrangeParticleEntries(particleHandler->d_vy, helperHandler->d_realBuffer);
    arrangeParticleEntries(particleHandler->d_ay, helperHandler->d_realBuffer);
#if DIM == 3
    arrangeParticleEntries(particleHandler->d_z, helperHandler->d_realBuffer);
    arrangeParticleEntries(particleHandler->d_vz, helperHandler->d_realBuffer);
    arrangeParticleEntries(particleHandler->d_az, helperHandler->d_realBuffer);
#endif
#endif
    arrangeParticleEntries(particleHandler->d_mass, helperHandler->d_realBuffer);

    //TODO: for all entries...

    subDomainKeyTreeHandler->copy(To::host, true, true); //TODO: needed?

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
    //TODO for all entries
    // END: assigning particles to correct MPI process
    return (real)0; //TODO: time function
}


real Miluphpc::tree() {
    real time;
    // START: creating domain list
    Logger(INFO) << "building domain list ...";
    time = DomainListNS::Kernel::Launch::createDomainList(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                          domainListHandler->d_domainList, MAX_LEVEL,
                                                          curveType);
    Logger(TIME) << "createDomainList: " << time << " ms";

    integer domainListLength;
    cuda::copy(&domainListLength, domainListHandler->d_domainListIndex, 1, To::host);
    Logger(INFO) << "domainListLength = " << domainListLength;
    // END: creating domain list

    // START: tree construction (including common coarse tree)
    integer treeIndexBeforeBuildingTree;

    cuda::copy(&treeIndexBeforeBuildingTree, treeHandler->d_index, 1, To::host);
    Logger(INFO) << "treeIndexBeforeBuildingTree: " << treeIndexBeforeBuildingTree;

    Logger(INFO) << "building tree ...";
    time = TreeNS::Kernel::Launch::buildTree(treeHandler->d_tree, particleHandler->d_particles, numParticlesLocal,
                                             numParticles, true);
    Logger(TIME) << "buildTree: " << time << " ms";

    integer treeIndex;

    cuda::copy(&treeIndex, treeHandler->d_index, 1, To::host);

    // DEBUGGING
    Logger(INFO) << "numParticlesLocal: " << numParticlesLocal;
    Logger(INFO) << "numParticles: " << numParticles;
    Logger(INFO) << "numNodes: " << numNodes;
    Logger(INFO) << "treeIndex: " << treeIndex;
    integer numParticlesSum = numParticlesLocal;
    boost::mpi::communicator comm;
    all_reduce(comm, boost::mpi::inplace_t<integer*>(&numParticlesSum), 1, std::plus<integer>());
    Logger(INFO) << "numParticlesSum: " << numParticlesSum;
    //ParticlesNS::Kernel::Launch::info(particleHandler->d_particles, numParticlesLocal, numParticles, treeIndex);
    // end: DEBUGGING

    Logger(INFO) << "building domain tree ...";
    time = SubDomainKeyTreeNS::Kernel::Launch::buildDomainTree(treeHandler->d_tree, particleHandler->d_particles,
                                                               domainListHandler->d_domainList, numParticlesLocal,
                                                               numNodes);
    Logger(TIME) << "build(Domain)Tree: " << time << " ms";
    // END: creating domain list
    return time;
}

real Miluphpc::pseudoParticles() {
    compPseudoParticlesParallel();
    return (real)0; // TODO: time function
}

real Miluphpc::gravity() {
    parallelForce();
    return (real)0; // TODO: time function
}

real Miluphpc::sph() {

    int sphInsertOffset = 50000;

    integer *d_sphSendCount;
    integer *d_alreadyInserted;
    cuda::malloc(d_alreadyInserted, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses);
    cuda::malloc(d_sphSendCount, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses);

    cuda::set(d_sphSendCount, 0, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses);

    cuda::set(helperHandler->d_integerBuffer, -1, numParticles);

    SPH::Kernel::Launch::particles2Send(subDomainKeyTreeHandler->d_subDomainKeyTree, treeHandler->d_tree,
                                        particleHandler->d_particles, domainListHandler->d_domainList,
                                        lowestDomainListHandler->d_domainList, 21, helperHandler->d_integerBuffer,
                                        d_sphSendCount, d_alreadyInserted, sphInsertOffset,
                                        numParticlesLocal, numParticles, numNodes, 1e-1, curveType);

    integer totalSendCount = 0;

    integer *particles2SendSPH;
    particles2SendSPH = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    cuda::copy(particles2SendSPH, d_sphSendCount, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses, To::host);
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
    cuda::copy(treeHandler->h_toDeleteLeaf, treeHandler->d_toDeleteLeaf, 2, To::device);


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
    cuda::copy(&treeHandler->h_toDeleteNode[0], treeHandler->d_index, 1, To::host);


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
    cuda::copy(&treeHandler->h_toDeleteNode[1], treeHandler->d_index, 1, To::host);
    cuda::copy(treeHandler->d_toDeleteNode, treeHandler->h_toDeleteNode, 2, To::device);


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
    cuda::free(d_alreadyInserted);

}

void Miluphpc::fixedLoadDistribution() {
    //for (int i=0; i<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; i++) {
    //    subDomainKeyTreeHandler->h_subDomainKeyTree->range[i] = i * (1UL << 63)/(subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses);
    //}
    //subDomainKeyTreeHandler->h_subDomainKeyTree->range[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses] = KEY_MAX;

    subDomainKeyTreeHandler->h_subDomainKeyTree->range[0] = 0UL;
    subDomainKeyTreeHandler->h_subDomainKeyTree->range[1] = 3458764513820540928;
    subDomainKeyTreeHandler->h_subDomainKeyTree->range[2] = 9223372036854775808;

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
    keyType keyMax = (keyType)KEY_MAX;
    cuda::set(&subDomainKeyTreeHandler->d_range[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses], keyMax, 1);
    subDomainKeyTreeHandler->copy(To::host, true, true);

    Logger(INFO) << "numProcesses: " << subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses;
    for(int i=0; i<=subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; i++) {
        Logger(INFO) << "range[" << i << "] = " << subDomainKeyTreeHandler->h_subDomainKeyTree->range[i];
    }

    int h_sum;
    cuda::copy(&h_sum, helperHandler->d_integerVal, 1, To::host);
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

template <typename T>
integer Miluphpc::sendParticles(T *sendBuffer, T *receiveBuffer, integer *sendLengths, integer *receiveLengths) {
    boost::mpi::communicator comm;

    std::vector<boost::mpi::request> reqParticles;
    std::vector<boost::mpi::status> statParticles;

    integer reqCounter = 0;
    integer receiveOffset = 0;
    integer sendOffset = 0;

    for (integer proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            reqParticles.push_back(comm.isend(proc, 17, &sendBuffer[sendOffset], sendLengths[proc]));
            statParticles.push_back(comm.recv(proc, 17, &receiveBuffer[receiveOffset], receiveLengths[proc]));

            receiveOffset += receiveLengths[proc];
            sendOffset += sendLengths[proc];
        }
    }

    boost::mpi::wait_all(reqParticles.begin(), reqParticles.end());

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

    //Gravity::Kernel::Launch::zeroDomainListNodes(particleHandler->d_particles, domainListHandler->d_domainList,
    //                                             lowestDomainListHandler->d_domainList);

    // compute local pseudo particles (not for domain list nodes, at least not for the upper domain list nodes)
    Gravity::Kernel::Launch::compLocalPseudoParticles(treeHandler->d_tree, particleHandler->d_particles,
                                                      domainListHandler->d_domainList, numParticles);

    integer domainListIndex;
    integer lowestDomainListIndex;
    // x ----------------------------------------------------------------------------------------------
    cuda::copy(&domainListIndex, domainListHandler->d_domainListIndex, 1, To::host);
    cuda::copy(&lowestDomainListIndex, lowestDomainListHandler->d_domainListIndex, 1, To::host);

    Logger(INFO) << "domainListIndex: " << domainListIndex << " | lowestDomainListIndex: " << lowestDomainListIndex;

    //KernelHandler.prepareLowestDomainExchange(d_x, d_mass, d_tempArray, d_lowestDomainListIndices,
    //                                          d_lowestDomainListIndex, d_lowestDomainListKeys,
    //                                          d_lowestDomainListCounter, false);

    boost::mpi::communicator comm;

    cuda::set(lowestDomainListHandler->d_domainListCounter, 0);
    Gravity::Kernel::Launch::prepareLowestDomainExchange(particleHandler->d_particles, lowestDomainListHandler->d_domainList,
                                                         helperHandler->d_helper, Entry::x);

    HelperNS::sortArray(helperHandler->d_realBuffer, &helperHandler->d_realBuffer[DOMAIN_LIST_SIZE],
                        lowestDomainListHandler->d_domainListKeys, lowestDomainListHandler->d_sortedDomainListKeys,
                        domainListIndex);


    // share among processes
    //TODO: domainListIndex or lowestDomainListIndex?
    //MPI_Allreduce(MPI_IN_PLACE, &helperHandler->d_realBuffer[DOMAIN_LIST_SIZE], domainListIndex, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    all_reduce(comm, boost::mpi::inplace_t<real*>(&helperHandler->d_realBuffer[DOMAIN_LIST_SIZE]), domainListIndex,
               std::plus<real>());

    cuda::set(lowestDomainListHandler->d_domainListCounter, 0);

    Gravity::Kernel::Launch::updateLowestDomainListNodes(particleHandler->d_particles, lowestDomainListHandler->d_domainList,
                                                         helperHandler->d_helper, Entry::x);

    //DomainListNS::Kernel::Launch::info(particleHandler->d_particles, domainListHandler->d_domainList);

    // y ----------------------------------------------------------------------------------------------

    //cudaMemcpy(&domainListIndex, d_domainListIndex, sizeof(int), cudaMemcpyDeviceToHost);
    //Logger(INFO) << "domainListIndex: " << domainListIndex;
    cuda::set(lowestDomainListHandler->d_domainListCounter, 0);
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

    cuda::set(lowestDomainListHandler->d_domainListCounter, 0);

    Gravity::Kernel::Launch::updateLowestDomainListNodes(particleHandler->d_particles, lowestDomainListHandler->d_domainList,
                                                         helperHandler->d_helper, Entry::y);

    // z ----------------------------------------------------------------------------------------------

    //cudaMemcpy(&domainListIndex, d_domainListIndex, sizeof(int), cudaMemcpyDeviceToHost);
    //Logger(INFO) << "domainListIndex: " << domainListIndex;
    cuda::set(lowestDomainListHandler->d_domainListCounter, 0);
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

    cuda::set(lowestDomainListHandler->d_domainListCounter, 0);

    Gravity::Kernel::Launch::updateLowestDomainListNodes(particleHandler->d_particles, lowestDomainListHandler->d_domainList,
                                                         helperHandler->d_helper, Entry::z);

    // m ----------------------------------------------------------------------------------------------

    //cudaMemcpy(&domainListIndex, d_domainListIndex, sizeof(int), cudaMemcpyDeviceToHost);
    //Logger(INFO) << "domainListIndex: " << domainListIndex;
    cuda::set(lowestDomainListHandler->d_domainListCounter, 0);
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

    cuda::set(lowestDomainListHandler->d_domainListCounter, 0);

    Gravity::Kernel::Launch::updateLowestDomainListNodes(particleHandler->d_particles, lowestDomainListHandler->d_domainList,
                                                         helperHandler->d_helper, Entry::mass);

    // ------------------------------------------------------------------------------------------------

    Gravity::Kernel::Launch::compLowestDomainListNodes(particleHandler->d_particles,
                                                       lowestDomainListHandler->d_domainList);
    //end: for all entries!


    Gravity::Kernel::Launch::compDomainListPseudoParticles(treeHandler->d_tree, particleHandler->d_particles,
                                                           domainListHandler->d_domainList,
                                                           lowestDomainListHandler->d_domainList,
                                                           numParticles);

    Logger(INFO) << "Finished: compPseudoParticlesParallel()";


}

void Miluphpc::parallelForce() {

    //debugging
    HelperNS::Kernel::Launch::resetArray(helperHandler->d_realBuffer, (real)0, numParticles);
    //KernelHandler.resetFloatArray(d_tempArray, 0.f, 2*numParticles, false);

    cuda::set(domainListHandler->d_domainListCounter, 0);

    Logger(INFO) << "compTheta()";

    //compTheta
    Gravity::Kernel::Launch::compTheta(subDomainKeyTreeHandler->d_subDomainKeyTree, treeHandler->d_tree,
                                       particleHandler->d_particles, domainListHandler->d_domainList,
                                       helperHandler->d_helper, curveType);

    integer relevantIndicesCounter;
    cuda::copy(&relevantIndicesCounter, domainListHandler->d_domainListCounter, 1, To::host);
    Logger(INFO) << "relevantIndicesCounter: " << relevantIndicesCounter;

    integer *h_relevantDomainListProcess;
    h_relevantDomainListProcess = new integer[relevantIndicesCounter]; //TODO: delete [] h_relevantDomainListProcess;
    cuda::copy(h_relevantDomainListProcess, domainListHandler->d_relevantDomainListProcess, relevantIndicesCounter, To::host);

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

    cuda::set(domainListHandler->d_domainListCounter, 0);
    // integer currentDomainListCounter;
    // real massOfDomainListNode;
    integer *d_markedSendIndices = buffer->d_integerBuffer;
    real *d_collectedEntries = buffer->d_realBuffer;

    //integer *h_relevantIndicesCounter;
    //h_relevantIndicesCounter = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];

    cuda::set(d_particles2SendIndices, -1, numParticles);
    cuda::set(d_pseudoParticles2SendIndices, -1, numParticles);
    cuda::set(d_pseudoParticles2SendLevels, -1, numParticles);
    cuda::set(d_pseudoParticles2ReceiveLevels, -1, numParticles);

    cuda::set(d_particles2SendCount, 0, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses);
    cuda::set(d_pseudoParticles2SendCount, 0, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses);

    //integer *d_particles2SendCount;
    //integer *d_pseudoParticles2SendCount;

    integer particlesOffset = 0;
    integer pseudoParticlesOffset = 0;

    integer particlesOffsetBuffer;
    integer pseudoParticlesOffsetBuffer;

    integer *h_particles2SendCount = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    integer *h_pseudoParticles2SendCount = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];

    for (integer proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        cuda::set(d_markedSendIndices, -1, numNodes);
        for (integer relevantIndex = 0; relevantIndex < relevantIndicesCounter; relevantIndex++) {
            if (h_relevantDomainListProcess[relevantIndex] == proc) {
                Logger(INFO) << "h_relevantDomainListProcess[" << relevantIndex << "] = " << h_relevantDomainListProcess[relevantIndex];
                for (integer level=0; level<MAX_LEVEL; level++) {
                    Gravity::Kernel::Launch::symbolicForce(subDomainKeyTreeHandler->d_subDomainKeyTree, treeHandler->d_tree,
                                                           particleHandler->d_particles, domainListHandler->d_domainList,
                                                           d_markedSendIndices, diam, theta_, numParticlesLocal, numParticles,
                                                           relevantIndex, level, curveType);
                }
                //Gravity::Kernel::Launch::symbolicForce(subDomainKeyTreeHandler->d_subDomainKeyTree, treeHandler->d_tree,
                //                                       particleHandler->d_particles, domainListHandler->d_domainList,
                //                                       d_markedSendIndices, diam, theta_, numParticlesLocal, numParticles,
                //                                       relevantIndex, curveType);
                Logger(INFO) << "Finished symbolicForce() for proc = " << proc << " and relevantIndex = " << relevantIndex;
            }
        }
        Gravity::Kernel::Launch::collectSendIndices(treeHandler->d_tree, particleHandler->d_particles,
                                                    d_markedSendIndices, &d_particles2SendIndices[particlesOffset],
                                                    &d_pseudoParticles2SendIndices[pseudoParticlesOffset],
                                                    &d_pseudoParticles2SendLevels[pseudoParticlesOffset],
                                                    &d_particles2SendCount[proc],
                                                    &d_pseudoParticles2SendCount[proc],
                                                    numParticles, numNodes, curveType);

        cuda::copy(&particlesOffsetBuffer, &d_particles2SendCount[proc], 1, To::host);
        cuda::copy(&pseudoParticlesOffsetBuffer, &d_pseudoParticles2SendCount[proc], 1, To::host);
    }

    cuda::copy(h_particles2SendCount, d_particles2SendCount, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses, To::host);
    cuda::copy(h_pseudoParticles2SendCount, d_pseudoParticles2SendCount, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses, To::host);

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
        Logger(INFO) << "h_particles2SendCount[" << proc << "] = " << h_particles2SendCount[proc];
        Logger(INFO) << "h_pseudoParticles2SendCount[" << proc << "] = " << h_pseudoParticles2SendCount[proc];
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            particleSendLengths[proc] = h_particles2SendCount[proc];
            pseudoParticleSendLengths[proc] = h_pseudoParticles2SendCount[proc];
        }
    }

    delete [] h_particles2SendCount;
    delete [] h_pseudoParticles2SendCount;

    mpi::messageLengths(subDomainKeyTreeHandler->h_subDomainKeyTree, particleSendLengths, particleReceiveLengths);

    integer particleTotalReceiveLength = 0;
    integer particleTotalSendLength = 0;
    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            particleTotalReceiveLength += particleReceiveLengths[proc];
            particleTotalSendLength += particleSendLengths[proc];
        }
    }
    Logger(INFO) << "particleTotalReceiveLength: " << particleTotalReceiveLength;
    Logger(INFO) << "particleTotalSendLength: " << particleTotalSendLength;

    mpi::messageLengths(subDomainKeyTreeHandler->h_subDomainKeyTree, pseudoParticleSendLengths, pseudoParticleReceiveLengths);

    integer pseudoParticleTotalReceiveLength = 0;
    integer pseudoParticleTotalSendLength = 0;
    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            pseudoParticleTotalReceiveLength += pseudoParticleReceiveLengths[proc];
            pseudoParticleTotalSendLength += pseudoParticleSendLengths[proc];
        }
    }
    Logger(INFO) << "pseudoParticleTotalReceiveLength: " << pseudoParticleTotalReceiveLength;
    Logger(INFO) << "pseudoParticleTotalSendLength: " << pseudoParticleTotalSendLength;

    //integer *d_particles2SendIndices;
    //integer *d_pseudoParticles2SendIndices;
    //integer *d_pseudoParticles2SendLevels;
    //integer *d_pseudoParticles2ReceiveLevels;

    integer treeIndex;
    cuda::copy(&treeIndex, treeHandler->d_index, 1, To::host);

    // debugging
    //TreeNS::Kernel::Launch::info(treeHandler->d_tree, particleHandler->d_particles, treeIndex, treeIndex + pseudoParticleTotalReceiveLength);
    //TreeNS::Kernel::Launch::info(treeHandler->d_tree, particleHandler->d_particles, numParticlesLocal, numParticlesLocal + particleTotalReceiveLength);
    // end: debugging

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
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_vx, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_vx[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
    // ax-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_ax, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_ax[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
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
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_vy, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_vy[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
    // ay-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_ay, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_ay[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
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
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_vz, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_vz[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
    // az-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_az, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_az[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);

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
    //integer *d_pseudoParticles2SendLevels;
    //integer *d_pseudoParticles2ReceiveLevels;
    //CudaUtils::Kernel::Launch::collectValues(d_pseudoParticles2SendIndices, particleHandler->d_mass, d_collectedEntries,
    //                                         pseudoParticleTotalSendLength);
    // TODO: check values for pseudo-particles levels!
    sendParticles(d_pseudoParticles2SendLevels, d_pseudoParticles2ReceiveLevels, pseudoParticleSendLengths,
                  pseudoParticleReceiveLengths);


    // debugging
    //Logger(INFO) << "exchanged particle entry: x";
    //TreeNS::Kernel::Launch::info(treeHandler->d_tree, particleHandler->d_particles, treeIndex, treeIndex + pseudoParticleTotalReceiveLength);
    //TreeNS::Kernel::Launch::info(treeHandler->d_tree, particleHandler->d_particles, numParticlesLocal, numParticlesLocal + particleTotalReceiveLength);
    // end: debugging

    treeHandler->h_toDeleteLeaf[0] = numParticlesLocal;
    treeHandler->h_toDeleteLeaf[1] = numParticlesLocal + particleTotalReceiveLength;
    cuda::copy(treeHandler->h_toDeleteLeaf, treeHandler->d_toDeleteLeaf, 2, To::device);

    // DEBUG
    //gpuErrorcheck(cudaMemset(buffer->d_integerVal, 0, sizeof(integer)));
    //CudaUtils::Kernel::Launch::findDuplicateEntries(&particleHandler->d_x[treeHandler->h_toDeleteLeaf[0]],
    //                                                &particleHandler->d_y[treeHandler->h_toDeleteLeaf[0]],
    //                                                buffer->d_integerVal,
    //                                                particleTotalReceiveLength);
    //integer duplicates;
    //gpuErrorcheck(cudaMemcpy(&duplicates, buffer->d_integerVal, sizeof(integer), cudaMemcpyDeviceToHost));
    //Logger(INFO) << "duplicates: " << duplicates << " between: " << treeHandler->h_toDeleteLeaf[0] << " and " << treeHandler->h_toDeleteLeaf[0] + particleTotalReceiveLength;
    // end: DEBUG

    treeHandler->h_toDeleteNode[0] = treeIndex;
    treeHandler->h_toDeleteNode[1] = treeIndex + pseudoParticleTotalReceiveLength;
    cuda::copy(treeHandler->h_toDeleteNode, treeHandler->d_toDeleteNode, 2, To::device);

    for (int level=0; level<MAX_LEVEL; level++) {
        Gravity::Kernel::Launch::insertReceivedPseudoParticles(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                               treeHandler->d_tree, particleHandler->d_particles,
                                                               d_pseudoParticles2ReceiveLevels, level, numParticles,
                                                               numParticles);
    }
    //Gravity::Kernel::Launch::insertReceivedPseudoParticles(subDomainKeyTreeHandler->d_subDomainKeyTree,
    //                                                       treeHandler->d_tree, particleHandler->d_particles,
    //                                                       d_pseudoParticles2ReceiveLevels, numParticles,
    //                                                       numParticles);

    Gravity::Kernel::Launch::insertReceivedParticles(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                     treeHandler->d_tree, particleHandler->d_particles,
                                                     domainListHandler->d_domainList, lowestDomainListHandler->d_domainList,
                                                     numParticles, numParticles);

    //TreeNS::Kernel::Launch::info(treeHandler->d_tree, particleHandler->d_particles);

    Logger(INFO) << "Finished inserting received particles!";


    float elapsedTime = 0.f;


    TreeNS::Kernel::Launch::sort(treeHandler->d_tree, /*treeHandler->h_toDeleteNode[1]*/numParticlesLocal, numParticles, false);
    //KernelHandler.sort(d_count, d_start, d_sorted, d_child, d_index, numParticles, numParticles, false); //TODO: numParticlesLocal or numParticles?

    //Logger(INFO) << "FINISHED!";
    //exit(0);

    Logger(INFO) << "treeHandler->h_toDeleteLeaf[1] = " << treeHandler->h_toDeleteLeaf[1];
    //actual (local) force
    integer warp = 32;
    integer stackSize = 64;
    integer blockSize = 256;
    Gravity::Kernel::Launch::computeForces(treeHandler->d_tree, particleHandler->d_particles, numParticles/*treeHandler->h_toDeleteNode[1]numParticlesLocal*/, /*treeHandler->h_toDeleteNode[1]*/ numParticles,
                                           blockSize, warp, stackSize);

    //elapsedTime = KernelHandler.computeForces(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_ax, d_ay, d_az, d_mass, d_sorted, d_child,
    //                                          d_min_x, d_max_x, d_min_y, d_max_y, d_min_z, d_max_z, numParticlesLocal,
    //                                          numParticles, parameters.gravity, d_subDomainHandler, true); //TODO: numParticlesLocal or numParticles?


    // repairTree
    //TODO: necessary? Tree is build for every iteration

    Gravity::Kernel::Launch::repairTree(treeHandler->d_tree, particleHandler->d_particles,
                                        domainListHandler->d_domainList, numParticlesLocal, numNodes);

    //TreeNS::Kernel::Launch::info(treeHandler->d_tree, particleHandler->d_particles, treeIndex, treeIndex + pseudoParticleTotalReceiveLength);
    //TreeNS::Kernel::Launch::info(treeHandler->d_tree, particleHandler->d_particles, numParticlesLocal, numParticlesLocal + particleTotalReceiveLength);


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
    cuda::copy(h_keys, d_keys, numParticlesLocal, To::host);

    integer keyProc;

    for (int i=0; i<numParticlesLocal; i++) {
        x.push_back({particleHandler->h_x[i], particleHandler->h_y[i], particleHandler->h_z[i]});
        v.push_back({particleHandler->h_vx[i], particleHandler->h_vy[i], particleHandler->h_vz[i]});
        k.push_back(h_keys[i]);
    }

    cuda::free(d_keys);
    delete [] h_keys;

    // receive buffer
    int procN[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];

    // send buffer
    int sendProcN[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++){
        sendProcN[proc] = subDomainKeyTreeHandler->h_subDomainKeyTree->rank == proc ? numParticlesLocal : 0;
    }

    boost::mpi::communicator comm;
    all_reduce(comm, sendProcN, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses, procN,
               boost::mpi::maximum<integer>());

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
