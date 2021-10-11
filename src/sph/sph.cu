#include "../../include/sph/sph.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"
//#include <cub/cub.cuh>

namespace SPH {

    void exchangeParticleEntry(SubDomainKeyTree *subDomainKeyTree, real *entry, real *toSend, integer *sendLengths,
                               integer *receiveLengths, integer numParticlesLocal) {

        boost::mpi::communicator comm;

        std::vector<boost::mpi::request> reqParticles;
        std::vector<boost::mpi::status> statParticles;

        integer reqCounter = 0;
        integer receiveOffset = 0;

        for (integer proc=0; proc<subDomainKeyTree->numProcesses; proc++) {
            if (proc != subDomainKeyTree->rank) {
                reqParticles.push_back(comm.isend(proc, 17, toSend, sendLengths[proc]));
                statParticles.push_back(comm.recv(proc, 17, &entry[numParticlesLocal] + receiveOffset, receiveLengths[proc]));
                receiveOffset += receiveLengths[proc];
            }
        }
        boost::mpi::wait_all(reqParticles.begin(), reqParticles.end());
    }

    namespace Kernel {

        __global__ void nearNeighbourSearch(Tree *tree, Particles *particles, integer *interactions, integer numParticlesLocal,
                                            integer numParticles, integer numNodes) {

            integer i, inc, nodeIndex, depth, childNumber, child, radius;
            real x, interactionDistance, dx, r, d, x_radius;
#if DIM > 1
            real y, dy, y_radius;
#endif
            integer currentNodeIndex[MAX_DEPTH];
            integer currentChildNumber[MAX_DEPTH];
            integer numberOfInteractions;
#if DIM == 3
            real z, dz, z_radius, r_temp;
#endif
            inc = blockDim.x * gridDim.x;
            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
//                x = p.x[i];
//#if DIM > 1
//                y = p.y[i];
//#if DIM == 3
//                z = p.z[i];
//#endif
//#endif
                //double sml; /* smoothing length of particle */
                //double smlj; /* smoothing length of potential interaction partner */

                x_radius = 0.5 * (*tree->maxX - (*tree->minX));
#if DIM > 1
                y_radius = 0.5 * (*tree->maxY - (*tree->minY));
#if DIM == 3
                z_radius = 0.5 * (*tree->maxZ - (*tree->minZ));
#endif
#endif

#if DIM == 1
                r = x_radius;
#elif DIM == 2
                r = fmaxf(x_radius, y_radius);
#else
                r_temp = fmaxf(x_radius, y_radius);
                r = fmaxf(r_temp, z_radius); //TODO: (0.5 * r) or (1.0 * r)
#endif

            // start at root
                depth = 0;
                currentNodeIndex[depth] = 0; //numNodes - 1;
                currentChildNumber[depth] = 0;
                numberOfInteractions = 0;
                r = radius * 0.5; // because we start with root children
                //sml = p.h[i];
                particles->noi[i] = 0;
                interactionDistance = (r + particles->sml[i]);

                do {

                    childNumber = currentChildNumber[depth];
                    nodeIndex = currentNodeIndex[depth];

                    while (childNumber < POW_DIM) {

                        child = tree->child[POW_DIM * nodeIndex + childNumber];
                        childNumber++;

                        if (child != -1 && child != i) {

                            dx = particles->x[i] - particles->x[child];
#if DIM > 1
                            dy = particles->y[i] - particles->y[child];
#if DIM == 3
                            dz = particles->z[i] - particles->z[child];
#endif
#endif

                            if (child < numParticles) {
                                //if (p_rhs.materialId[child] == EOS_TYPE_IGNORE) {
                                //    continue;
                                //}
                                d = dx*dx;
#if DIM > 1
                                d += dy*dy;
#if DIM == 3
                                d += dz*dz;
#endif
#endif

                                //smlj = p.h[child];

                                if (d < particles->sml[i]*particles->sml[i] && d < particles->sml[child]*particles->sml[child]) {
                                    interactions[i * MAX_NUM_INTERACTIONS + numberOfInteractions] = child;
                                    numberOfInteractions++;
//#if TOO_MANY_INTERACTIONS_KILL_PARTICLE
//                                    if (numberOfInteractions >= MAX_NUM_INTERACTIONS) {
//                                printf("setting the smoothing length for particle %d to 0!\n", i);
//                                p.h[i] = 0.0;
//                                p.noi[i] = 0;
//                                sml = 0.0;
//                                interactionDistance = 0.0;
//                                p_rhs.materialId[i] = EOS_TYPE_IGNORE;
//                                // continue with next particle by setting depth to -1
//                                // cms 2018-01-19
//                                depth = -1;
//                                break;
//                            }
//#endif
                                }
                            } else if (fabs(dx) < interactionDistance
                                       #if DIM > 1
                                       && fabs(dy) < interactionDistance
                                       #if DIM == 3
                                       && fabs(dz) < interactionDistance
#endif
#endif
                                    ) {
                                // put child on stack
                                currentChildNumber[depth] = childNumber;
                                currentNodeIndex[depth] = nodeIndex;
                                depth++;
                                r *= 0.5;
                                interactionDistance = (r + particles->sml[i]); //sml);
                                if (depth >= MAX_DEPTH) {
                                    printf("Error, maxdepth reached depth = %i < MAX_DEPTH = %i!\n", depth, MAX_DEPTH);
                                    assert(depth < MAX_DEPTH);
                                }
                                childNumber = 0;
                                nodeIndex = child;
                            }
                        }
                    }

                    depth--;
                    r *= 2.0;
                    interactionDistance = (r + particles->sml[i]);
                } while (depth >= 0);

                if (numberOfInteractions >= MAX_NUM_INTERACTIONS) {
                    printf("ERROR: Maximum number of interactions exceeded: %d / %d\n", numberOfInteractions, MAX_NUM_INTERACTIONS);
//#if !TOO_MANY_INTERACTIONS_KILL_PARTICLE
//                    assert(numberOfInteractions < MAX_NUM_INTERACTIONS);
//#endif
                    /*
                    for (child = 0; child < MAX_NUM_INTERACTIONS; child++) {
                        printf("(thread %d): %d - %d\n", threadIdx.x, i, interactions[i*MAX_NUM_INTERACTIONS+child]);
                    } */
                }
                particles->noi[i] = numberOfInteractions;
            }
        }

        __global__ void
        fixedRadiusNN(Tree *tree, Particles *particles, integer *interactions, integer numParticlesLocal,
                      integer numParticles, integer numNodes) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            integer childNumber, nodeIndex, depth, childIndex;

            real dx, x_radius;
#if DIM > 1
            real dy, y_radius;
#if DIM == 3
            real dz, z_radius;
            real r_temp;
#endif
#endif

            real d, r, interactionDistance;

            integer noOfInteractions;

            integer currentNodeIndex[MAX_DEPTH];
            integer currentChildNumber[MAX_DEPTH];

            while ((bodyIndex + offset) < numParticlesLocal) {

                // resetting
                for (integer i = 0; i < MAX_NUM_INTERACTIONS; i++) {
                    interactions[(bodyIndex + offset) * MAX_NUM_INTERACTIONS + i] = -1;
                }
                //numberOfInteractions[bodyIndex + offset] = 0;
                // end: resetting

                depth = 0;
                currentNodeIndex[depth] = 0; //numNodes - 1;
                currentChildNumber[depth] = 0;
                noOfInteractions = 0;

                x_radius = 0.5 * (*tree->maxX - (*tree->minX));
#if DIM > 1
                y_radius = 0.5 * (*tree->maxY - (*tree->minY));
#if DIM == 3
                z_radius = 0.5 * (*tree->maxZ - (*tree->minZ));
#endif
#endif

#if DIM == 1
                r = x_radius;
#elif DIM == 2
                r = fmaxf(x_radius, y_radius);
#else
                r_temp = fmaxf(x_radius, y_radius);
                r = fmaxf(r_temp, z_radius); //TODO: (0.5 * r) or (1.0 * r)
#endif

                interactionDistance = (r + particles->sml[bodyIndex + offset]);

                do {
                    childNumber = currentChildNumber[depth];
                    nodeIndex = currentNodeIndex[depth];

                    while (childNumber < POW_DIM) {
                        childIndex = tree->child[POW_DIM * nodeIndex + childNumber];
                        childNumber++;

                        if (childIndex != -1 && childIndex != (bodyIndex + offset)) {
                            dx = particles->x[bodyIndex + offset] - particles->x[childIndex];
#if DIM > 1
                            dy = particles->y[bodyIndex + offset] - particles->y[childIndex];
#if DIM == 3
                            dz = particles->z[bodyIndex + offset] - particles->z[childIndex];
#endif
#endif
                            // its a leaf
                            if (childIndex < numParticles) {
#if DIM == 1
                                d = dx*dx;
#elif DIM == 2
                                d = dx*dx + dy*dy;
#else
                                d = dx * dx + dy * dy + dz * dz;
#endif

                                if ((bodyIndex + offset) % 1000 == 0) {
                                    //printf("sph: index = %i, d = %i\n", bodyIndex+offset, d);
                                }

                                if (d < (particles->sml[bodyIndex + offset] * particles->sml[bodyIndex + offset])) {
                                    //printf("Adding interaction partner!\n");
                                    interactions[(bodyIndex + offset) * MAX_NUM_INTERACTIONS +
                                                 noOfInteractions] = childIndex;
                                    noOfInteractions++;
                                }
                            }
#if DIM == 1
                                else if (fabs(dx) < interactionDistance) {
#elif DIM == 2
                                else if (fabs(dx) < interactionDistance &&
                                     fabs(dy) < interactionDistance) {
#else
                            else if (fabs(dx) < interactionDistance &&
                                     fabs(dy) < interactionDistance &&
                                     fabs(dz) < interactionDistance) {
#endif
                                currentChildNumber[depth] = childNumber;
                                currentNodeIndex[depth] = nodeIndex;
                                //if (depth < MAX_DEPTH) { //TODO: REMOVE!!! just to keep kernel from crashing as long as sml is not dynamic!
                                //    // put child on stack
                                //    depth++;
                                //}
                                depth++;
                                r *= 0.5;
                                interactionDistance = (r + particles->sml[bodyIndex + offset]);
                                if (depth > MAX_DEPTH) {
                                    printf("ERROR: maximal depth reached! depth = %i > MAX_DEPTH = %i\n", depth, MAX_DEPTH);
                                    assert(depth < MAX_DEPTH);
                                }
                                childNumber = 0;
                                nodeIndex = childIndex;
                            }
                        }
                    }

                    depth--;
                    r *= 2.0;
                    interactionDistance = (r + particles->sml[bodyIndex + offset]);

                } while (depth >= 0);

                particles->noi[bodyIndex + offset] = noOfInteractions;
                offset += stride;
            }
        }

        __global__ void
        fixedRadiusNN_Test(Tree *tree, Particles *particles, integer *interactions, integer numParticlesLocal,
                           integer numParticles, integer numNodes) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            integer childNumber, nodeIndex, depth, childIndex;

            real dx, x_radius;
#if DIM > 1
            real dy, y_radius;
#if DIM == 3
            real dz, z_radius;
            real r_temp;
#endif
#endif

            real d, r, interactionDistance;

            integer noOfInteractions;

            //integer currentNodeIndex[MAX_DEPTH];
            //integer currentChildNumber[MAX_DEPTH];

            extern __shared__ integer buffer[];
            integer *currentNodeIndex = (integer*)buffer;
            integer *currentChildNumber = (integer*)&currentNodeIndex[MAX_DEPTH];

            //__shared__ integer currentNodeIndex[MAX_DEPTH];
            //__shared__ integer currentChildNumber[MAX_DEPTH];

            while ((bodyIndex + offset) < numParticlesLocal) {

                // resetting
                for (integer i = 0; i < MAX_NUM_INTERACTIONS; i++) {
                    interactions[(bodyIndex + offset) * MAX_NUM_INTERACTIONS + i] = -1;
                }
                //numberOfInteractions[bodyIndex + offset] = 0;
                // end: resetting

                depth = 0;
                currentNodeIndex[depth] = 0; //numNodes - 1;
                currentChildNumber[depth] = 0;
                noOfInteractions = 0;

                x_radius = 0.5 * (*tree->maxX - (*tree->minX));
#if DIM > 1
                y_radius = 0.5 * (*tree->maxY - (*tree->minY));
#if DIM == 3
                z_radius = 0.5 * (*tree->maxZ - (*tree->minZ));
#endif
#endif

#if DIM == 1
                r = x_radius;
#elif DIM == 2
                r = fmaxf(x_radius, y_radius);
#else
                r_temp = fmaxf(x_radius, y_radius);
                r = fmaxf(r_temp, z_radius); //TODO: (0.5 * r) or (1.0 * r)
#endif

                interactionDistance = (r + particles->sml[bodyIndex + offset]);

                do {
                    childNumber = currentChildNumber[depth];
                    nodeIndex = currentNodeIndex[depth];

                    while (childNumber < POW_DIM) {
                        childIndex = tree->child[POW_DIM * nodeIndex + childNumber];
                        childNumber++;

                        if (childIndex != -1 && childIndex != (bodyIndex + offset)) {
                            dx = particles->x[bodyIndex + offset] - particles->x[childIndex];
#if DIM > 1
                            dy = particles->y[bodyIndex + offset] - particles->y[childIndex];
#if DIM == 3
                            dz = particles->z[bodyIndex + offset] - particles->z[childIndex];
#endif
#endif
                            // its a leaf
                            if (childIndex < numParticles) {
#if DIM == 1
                                d = dx*dx;
#elif DIM == 2
                                d = dx*dx + dy*dy;
#else
                                d = dx * dx + dy * dy + dz * dz;
#endif

                                if ((bodyIndex + offset) % 1000 == 0) {
                                    //printf("sph: index = %i, d = %i\n", bodyIndex+offset, d);
                                }

                                if (d < (particles->sml[bodyIndex + offset] * particles->sml[bodyIndex + offset])) {
                                    //printf("Adding interaction partner!\n");
                                    interactions[(bodyIndex + offset) * MAX_NUM_INTERACTIONS +
                                                 noOfInteractions] = childIndex;
                                    noOfInteractions++;
                                }
                            }
#if DIM == 1
                                else if (fabs(dx) < interactionDistance) {
#elif DIM == 2
                                else if (fabs(dx) < interactionDistance &&
                                     fabs(dy) < interactionDistance) {
#else
                            else if (fabs(dx) < interactionDistance &&
                                     fabs(dy) < interactionDistance &&
                                     fabs(dz) < interactionDistance) {
#endif
                                // put child on stack
                                currentChildNumber[depth] = childNumber;
                                currentNodeIndex[depth] = nodeIndex;
                                depth++;
                                r *= 0.5;
                                interactionDistance = (r + particles->sml[bodyIndex + offset]);

                                if (depth > MAX_DEPTH) {
                                    printf("ERROR: maximal depth reached! MAX_DEPTH = %i\n", MAX_DEPTH);
                                    assert(depth < MAX_DEPTH);
                                }
                                childNumber = 0;
                                nodeIndex = childIndex;
                            }
                        }
                    }

                    depth--;
                    r *= 2.0;
                    interactionDistance = (r + particles->sml[bodyIndex + offset]);

                } while (depth >= 0);

                particles->noi[bodyIndex + offset] = noOfInteractions;
                offset += stride;
            }
        }

        __global__ void compTheta(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                  DomainList *lowestDomainList, Curve::Type curveType) {

            integer index = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;
            integer bodyIndex;
            keyType key;
            integer domainIndex;
            integer proc;

            //"loop" over domain list nodes
            while ((index + offset) < *lowestDomainList->domainListIndex) {

                bodyIndex = lowestDomainList->domainListIndices[index + offset];
                //calculate key
                key = tree->getParticleKey(particles, bodyIndex, MAX_LEVEL, curveType);

                //if domain list node belongs to other process: add to relevant domain list indices
                proc = subDomainKeyTree->key2proc(key);
                if (proc != subDomainKeyTree->rank) {
                    domainIndex = atomicAdd(lowestDomainList->domainListCounter, 1);
                    lowestDomainList->relevantDomainListIndices[domainIndex] = bodyIndex;
                    lowestDomainList->relevantDomainListLevels[domainIndex] = lowestDomainList->domainListLevels[index +
                                                                                                                 offset];
                    lowestDomainList->relevantDomainListProcess[domainIndex] = proc;

                    //printf("[rank %i] Adding relevant domain list node: %i (%f, %f, %f)\n", subDomainKeyTree->rank,
                    //       bodyIndex, particles->x[bodyIndex],
                    //       particles->y[bodyIndex], particles->z[bodyIndex]);
                }
                offset += stride;
            }
        }

        __global__ void symbolicForce(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                      DomainList *lowestDomainList, integer *sendIndices,
                                      integer n, integer m, integer relevantIndex,
                                      Curve::Type curveType) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            integer insertIndex;
            integer insertIndexOffset;

            integer proc, currentChild;
            integer childPath;

            real dx, min_x, max_x;
#if DIM > 1
            real dy, min_y, max_y;
#if DIM == 3
            real dz, min_z, max_z;
#endif
#endif
            real d;

            while ((bodyIndex + offset) < n) {


                min_x = *tree->minX;
                max_x = *tree->maxX;
#if DIM > 1
                min_y = *tree->minY;
                max_y = *tree->maxY;
#if DIM == 3
                min_z = *tree->minZ;
                max_z = *tree->maxZ;
#endif
#endif

                //if (bodyIndex + offset == 0) {
                //    printf("lowestDomainList->domainListLevels[%i] = %i\n", relevantIndex, lowestDomainList->relevantDomainListLevels[relevantIndex]);
                //}

                // determine domain list node's bounding box (in order to determine the distance)
                for (int j = 0; j < lowestDomainList->relevantDomainListLevels[relevantIndex]; j++) {
                    //childPath = 0;
                    if (particles->x[lowestDomainList->relevantDomainListIndices[relevantIndex]] < 0.5 * (min_x + max_x)) {
                        //childPath += 1;
                        max_x = 0.5 * (min_x + max_x);
                    } else {
                        min_x = 0.5 * (min_x + max_x);
                    }
#if DIM > 1
                    if (particles->y[lowestDomainList->relevantDomainListIndices[relevantIndex]] < 0.5 * (min_y + max_y)) {
                        //childPath += 2;
                        max_y = 0.5 * (min_y + max_y);
                    } else {
                        min_y = 0.5 * (min_y + max_y);
                    }
#if DIM == 3
                    if (particles->z[lowestDomainList->relevantDomainListIndices[relevantIndex]] < 0.5 * (min_z + max_z)) {
                        //childPath += 4;
                        max_z = 0.5 * (min_z + max_z);
                    } else {
                        min_z = 0.5 * (min_z + max_z);
                    }
#endif
#endif
                }

                // x-direction
                if (particles->x[bodyIndex + offset] < min_x) {
                    // outside
                    dx = particles->x[bodyIndex + offset] - min_x;
                } else if (particles->x[bodyIndex + offset] > max_x) {
                    // outside
                    dx = particles->x[bodyIndex + offset] - max_x;
                } else {
                    // in between: do nothing
                    dx = 0;
                }
#if DIM > 1
                // y-direction
                if (particles->y[bodyIndex + offset] < min_y) {
                    // outside
                    dy = particles->y[bodyIndex + offset] - min_y;
                } else if (particles->y[bodyIndex + offset] > max_y) {
                    // outside
                    dy = particles->y[bodyIndex + offset] - max_y;
                } else {
                    // in between: do nothing
                    dy = 0;
                }
#if DIM == 3
                // z-direction
                if (particles->z[bodyIndex + offset] < min_z) {
                    // outside
                    dz = particles->z[bodyIndex + offset] - min_z;
                } else if (particles->z[bodyIndex + offset] > max_z) {
                    // outside
                    dz = particles->z[bodyIndex + offset] - max_z;
                } else {
                    // in between: do nothing
                    dz = 0;
                }
#endif
#endif

#if DIM == 1
                d = dx*dx;
#elif DIM == 2
                d = dx*dx + dy*dy;
#else
                d = dx * dx + dy * dy + dz * dz;
#endif

                if (d < (particles->sml[bodyIndex + offset] * particles->sml[bodyIndex + offset])) {
                    //printf("d = %f < (%f * %f = %f)\n", d, particles->sml[bodyIndex + offset], particles->sml[bodyIndex + offset],
                    //       particles->sml[bodyIndex + offset] * particles->sml[bodyIndex + offset]);
                    sendIndices[bodyIndex + offset] = 1;
                }
                //else {
                //    sendIndices[bodyIndex + offset] = -1;
                //}

                __threadfence();
                offset += stride;
            }


        }

        __global__ void collectSendIndices(Tree *tree, Particles *particles, integer *sendIndices,
                                           integer *particles2Send, integer *particlesCount,
                                           integer n, integer length, Curve::Type curveType) {

            integer bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
            integer stride = blockDim.x*gridDim.x;
            integer offset = 0;

            integer particleInsertIndex;

            while ((bodyIndex + offset) < length) {

                if (sendIndices[bodyIndex + offset] == 1) {
                    particleInsertIndex = atomicAdd(particlesCount, 1);
                    particles2Send[particleInsertIndex] = bodyIndex + offset;
                }

                __threadfence();
                offset += stride;
            }

        }


        __global__ void particles2Send(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                       DomainList *domainList, DomainList *lowestDomainList, integer maxLevel,
                                       integer *toSend, integer *sendCount, integer *alreadyInserted,
                                       integer insertOffset,
                                       integer numParticlesLocal, integer numParticles, integer numNodes, real radius,
                                       Curve::Type curveType) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            integer insertIndex;
            integer insertIndexOffset;

            integer proc, currentChild;
            integer childPath;

            real dx, min_x, max_x;
#if DIM > 1
            real dy, min_y, max_y;
#if DIM == 3
            real dz, min_z, max_z;
#endif
#endif
            real d;

            //int alreadyInserted[10];

            while ((bodyIndex + offset) < numParticlesLocal) {

                if ((bodyIndex + offset) == 0) {
                    printf("sphParticles2SendKernel: insertOffset = %i\n", insertOffset);
                }

                //toSend[bodyIndex + offset] = -1;

                for (int i = 0; i < subDomainKeyTree->numProcesses; i++) {
                    alreadyInserted[i] = 0;
                }

                // loop over (lowest?) domain list nodes
                for (int i = 0; i < *lowestDomainList->domainListIndex; i++) {

                    min_x = *tree->minX;
                    max_x = *tree->maxX;
#if DIM > 1
                    min_y = *tree->minY;
                    max_y = *tree->maxY;
#if DIM == 3
                    min_z = *tree->minZ;
                    max_z = *tree->maxZ;
#endif
#endif

                    // TODO: hilbert change!?
                    proc = KeyNS::key2proc(lowestDomainList->domainListKeys[i], subDomainKeyTree);
                    // check if (lowest?) domain list node belongs to other process

                    if (proc != subDomainKeyTree->rank && alreadyInserted[proc] != 1) {

                        /*int path[MAX_LEVEL];
                        for (integer j = 0; j <= lowestDomainList->domainListLevels[i]; j++) { //TODO: "<" or "<="
                            path[j] = (integer) (
                                    lowestDomainList->domainListKeys[i] >> (MAX_LEVEL * DIM - DIM * (j + 1))
                                    & (integer) (POW_DIM - 1));
                        }

                        for (integer j = 0; j <= lowestDomainList->domainListLevels[i]; j++) {

                            currentChild = path[j];

                            if (currentChild % 2 != 0) {
                                max_x = 0.5 * (min_x + max_x);
                                currentChild -= 1;
                            } else {
                                min_x = 0.5 * (min_x + max_x);
                            }
#if DIM > 1
                            if (currentChild % 2 == 0 && currentChild % 4 != 0) {
                                max_y = 0.5 * (min_y + max_y);
                                currentChild -= 2;
                            } else {
                                min_y = 0.5 * (min_y + max_y);
                            }
#if DIM == 3
                            if (currentChild == 4) {
                                max_z = 0.5 * (min_z + max_z);
                                currentChild -= 4;
                            } else {
                                min_z = 0.5 * (min_z + max_z);
                            }
#endif
#endif
                        }*/

                        // determine domain list node's bounding box (in order to determine the distance)
                        for (int j = 0; j < lowestDomainList->domainListLevels[i]; j++) {
                            childPath = 0;
                            if (particles->x[lowestDomainList->domainListIndices[i]] < 0.5 * (min_x + max_x)) {
                                //childPath += 1;
                                max_x = 0.5 * (min_x + max_x);
                            } else {
                                min_x = 0.5 * (min_x + max_x);
                            }
#if DIM > 1
                            if (particles->y[lowestDomainList->domainListIndices[i]] < 0.5 * (min_y + max_y)) {
                                //childPath += 2;
                                max_y = 0.5 * (min_y + max_y);
                            } else {
                                min_y = 0.5 * (min_y + max_y);
                            }
#if DIM == 3
                            if (particles->z[lowestDomainList->domainListIndices[i]] < 0.5 * (min_z + max_z)) {
                                //childPath += 4;
                                max_z = 0.5 * (min_z + max_z);
                            } else {
                                min_z = 0.5 * (min_z + max_z);
                            }
#endif
#endif
                        }

                        // x-direction
                        if (particles->x[bodyIndex + offset] < min_x) {
                            // outside
                            dx = particles->x[bodyIndex + offset] - min_x;
                        } else if (particles->x[bodyIndex + offset] > max_x) {
                            // outside
                            dx = particles->x[bodyIndex + offset] - max_x;
                        } else {
                            // in between: do nothing
                            dx = 0;
                        }
#if DIM > 1
                        // y-direction
                        if (particles->y[bodyIndex + offset] < min_y) {
                            // outside
                            dy = particles->y[bodyIndex + offset] - min_y;
                        } else if (particles->y[bodyIndex + offset] > max_y) {
                            // outside
                            dy = particles->y[bodyIndex + offset] - max_y;
                        } else {
                            // in between: do nothing
                            dy = 0;
                        }
#if DIM == 3
                        // z-direction
                        if (particles->z[bodyIndex + offset] < min_z) {
                            // outside
                            dz = particles->z[bodyIndex + offset] - min_z;
                        } else if (particles->z[bodyIndex + offset] > max_z) {
                            // outside
                            dz = particles->z[bodyIndex + offset] - max_z;
                        } else {
                            // in between: do nothing
                            dz = 0;
                        }
#endif
#endif

#if DIM == 1
                        d = dx*dx;
#elif DIM == 2
                        d = dx*dx + dy*dy;
#else
                        d = dx * dx + dy * dy + dz * dz;
#endif

                        if (d < (particles->sml[bodyIndex + offset] * particles->sml[bodyIndex + offset])) {

                            //printf("d = %f < %f * %f = %f\n", d, particles->sml[bodyIndex + offset], particles->sml[bodyIndex + offset],
                            //       particles->sml[bodyIndex + offset] * particles->sml[bodyIndex + offset]);

                            insertIndex = atomicAdd(&sendCount[proc], 1);
                            if (insertIndex > 100000) {
                                printf("Attention!!! insertIndex: %i\n", insertIndex);
                            }
                            insertIndexOffset = insertOffset * proc; //0;
                            toSend[insertIndexOffset + insertIndex] = bodyIndex + offset;
                            //printf("[rank %i] toSend[%i] = %i\n", subDomainKeyTree->rank, insertIndexOffset + insertIndex,
                            //       toSend[insertIndexOffset + insertIndex]);
                            //toSend[proc][insertIndex] = bodyIndex+offset;
                            /*if (insertIndex % 100 == 0) {
                                printf("[rank %i] Inserting %i into : %i + %i  toSend[%i] = %i\n", s->rank, bodyIndex+offset,
                                       (insertOffset * proc), insertIndex, (insertOffset * proc) + insertIndex,
                                       toSend[(insertOffset * proc) + insertIndex]);
                            }*/
                            alreadyInserted[proc] = 1;
                            //break;
                        } else {
                            // else: do nothing
                        }
                    }
                }

                __threadfence();
                offset += stride;
            }

        }

        __global__ void collectSendIndicesBackup(integer *toSend, integer *toSendCollected, integer count) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            while ((bodyIndex + offset) < count) {
                toSendCollected[bodyIndex + offset] = toSend[bodyIndex + offset];
                //printf("toSendCollected[%i] = %i\n", bodyIndex + offset, toSendCollected[bodyIndex + offset]);
                offset += stride;
            }
        }

        __global__ void collectSendEntriesBackup(SubDomainKeyTree *subDomainKeyTree, real *entry, real *toSend,
                                           integer *sendIndices, integer *sendCount, integer totalSendCount,
                                           integer insertOffset) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            integer proc = subDomainKeyTree->rank - 1;
            if (proc < 0) {
                proc = 0;
            }

            if ((bodyIndex + offset) == 0) {
                printf("[rank %i] sendCount(%i, %i)\n", subDomainKeyTree->rank, sendCount[0], sendCount[1]);
            }

            //bodyIndex += proc*insertOffset;

            while ((bodyIndex + offset) < totalSendCount) {
                toSend[bodyIndex + offset] = entry[sendIndices[bodyIndex + offset]];
                printf("toSend[%i] = %f sendIndices = %i (insertOffset = %i)\n", bodyIndex + offset, toSend[bodyIndex + offset],
                       sendIndices[bodyIndex + offset], insertOffset);
                offset += stride;
            }
        }

        __global__ void insertReceivedParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                DomainList *domainList, DomainList *lowestDomainList, int n, int m) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;

            //note: -1 used as "null pointer"
            //note: -2 used to lock a child (pointer)

            integer offset;
            bool newBody = true;

            real min_x;
            real max_x;
#if DIM > 1
            real min_y;
            real max_y;
#if DIM == 3
            real min_z;
            real max_z;
#endif
#endif

            integer childPath;
            integer temp;

            offset = 0;

            bodyIndex += tree->toDeleteLeaf[0];

            if (bodyIndex == 0) {
                printf("tree->toDeleteLeaf[0] = %i\n", tree->toDeleteLeaf[0]);
                printf("tree->toDeleteLeaf[1] = %i\n", tree->toDeleteLeaf[1]);
            }

            /*while ((bodyIndex + offset) < tree->toDeleteLeaf[1]) {

                // copy bounding box(es)
                min_x = *tree->minX;
                max_x = *tree->maxX;
#if DIM > 1
                min_y = *tree->minY;
                max_y = *tree->maxY;
#if DIM == 3
                min_z = *tree->minZ;
                max_z = *tree->maxZ;
#endif
#endif
                temp = 0;
                childPath = 0;

                // find insertion point for body
                if (particles->x[bodyIndex + offset] < 0.5 * (min_x + max_x)) { // x direction
                    childPath += 1;
                    max_x = 0.5 * (min_x + max_x);
                }
                else {
                    min_x = 0.5 * (min_x + max_x);
                }
#if DIM > 1
                if (particles->y[bodyIndex + offset] < 0.5 * (min_y + max_y)) { // y direction
                    childPath += 2;
                    max_y = 0.5 * (min_y + max_y);
                }
                else {
                    min_y = 0.5 * (min_y + max_y);
                }
#if DIM == 3
                if (particles->z[bodyIndex + offset] < 0.5 * (min_z + max_z)) {  // z direction
                    childPath += 4;
                    max_z = 0.5 * (min_z + max_z);
                }
                else {
                    min_z = 0.5 * (min_z + max_z);
                }
#endif
#endif


                integer childIndex = tree->child[temp*POW_DIM + childPath];

                printf("rank[%i] child = %i childIndex = %i inserting x = (%f, %f, %f) %f\n", subDomainKeyTree->rank,
                       childPath, childIndex, particles->x[bodyIndex + offset],
                       particles->y[bodyIndex + offset],
                       particles->z[bodyIndex + offset],
                       particles->mass[bodyIndex + offset]);

                offset += stride;

            } */

            while ((bodyIndex + offset) < tree->toDeleteLeaf[1]) {

                //printf("rank[%i] inserting x = (%f, %f, %f) %f\n", subDomainKeyTree->rank,
                //       particles->x[bodyIndex + offset],
                //       particles->y[bodyIndex + offset],
                //       particles->z[bodyIndex + offset],
                //       particles->mass[bodyIndex + offset]);

                if (newBody) {

                    newBody = false;

                    // copy bounding box(es)
                    min_x = *tree->minX;
                    max_x = *tree->maxX;
#if DIM > 1
                    min_y = *tree->minY;
                    max_y = *tree->maxY;
#if DIM == 3
                    min_z = *tree->minZ;
                    max_z = *tree->maxZ;
#endif
#endif
                    temp = 0;
                    childPath = 0;

                    // find insertion point for body
                    if (particles->x[bodyIndex + offset] < 0.5 * (min_x + max_x)) { // x direction
                        childPath += 1;
                        max_x = 0.5 * (min_x + max_x);
                    }
                    else {
                        min_x = 0.5 * (min_x + max_x);
                    }
#if DIM > 1
                    if (particles->y[bodyIndex + offset] < 0.5 * (min_y + max_y)) { // y direction
                        childPath += 2;
                        max_y = 0.5 * (min_y + max_y);
                    }
                    else {
                        min_y = 0.5 * (min_y + max_y);
                    }
#if DIM == 3
                    if (particles->z[bodyIndex + offset] < 0.5 * (min_z + max_z)) {  // z direction
                        childPath += 4;
                        max_z = 0.5 * (min_z + max_z);
                    }
                    else {
                        min_z = 0.5 * (min_z + max_z);
                    }
#endif
#endif
                }

                integer childIndex = tree->child[temp*POW_DIM + childPath];

                // traverse tree until hitting leaf node
                while (childIndex >= m) { //n

                    temp = childIndex;

                    childPath = 0;

                    // find insertion point for body
                    if (particles->x[bodyIndex + offset] < 0.5 * (min_x + max_x)) { // x direction
                        childPath += 1;
                        max_x = 0.5 * (min_x + max_x);
                    }
                    else {
                        min_x = 0.5 * (min_x + max_x);
                    }
#if DIM > 1
                    if (particles->y[bodyIndex + offset] < 0.5 * (min_y + max_y)) { // y direction
                        childPath += 2;
                        max_y = 0.5 * (min_y + max_y);
                    }
                    else {
                        min_y = 0.5 * (min_y + max_y);
                    }
#if DIM == 3
                    if (particles->z[bodyIndex + offset] < 0.5 * (min_z + max_z)) { // z direction
                        childPath += 4;
                        max_z = 0.5 * (min_z + max_z);
                    }
                    else {
                        min_z = 0.5 * (min_z + max_z);
                    }
#endif
#endif
                    if (particles->mass[bodyIndex + offset] != 0) {
                        atomicAdd(&particles->x[temp], particles->weightedEntry(bodyIndex + offset, Entry::x));
#if DIM > 1
                        atomicAdd(&particles->y[temp], particles->weightedEntry(bodyIndex + offset, Entry::y));
#if DIM == 3
                        atomicAdd(&particles->z[temp], particles->weightedEntry(bodyIndex + offset, Entry::z));
#endif
#endif
                    }

                    atomicAdd(&particles->mass[temp], particles->mass[bodyIndex + offset]);
                    atomicAdd(&tree->count[temp], 1);

                    childIndex = tree->child[POW_DIM * temp + childPath];
                }

                // if child is not locked
                if (childIndex != -2) {

                    integer locked = temp * POW_DIM + childPath;

                    if (atomicCAS(&tree->child[locked], childIndex, -2) == childIndex) {

                        // check whether a body is already stored at the location
                        if (childIndex == -1) {
                            //insert body and release lock
                            tree->child[locked] = bodyIndex + offset;
                        }
                        else {
                            //if (childIndex >= n) {
                            //    printf("ATTENTION! %i\n", childIndex);
                            //}
                            integer patch = POW_DIM * m; //8*n
                            while (childIndex >= 0 && childIndex < n) { // was n

                                //create a new cell (by atomically requesting the next unused array index)
                                integer cell = atomicAdd(tree->index, 1);
                                patch = min(patch, cell);

                                if (patch != cell) {
                                    tree->child[POW_DIM * temp + childPath] = cell;
                                }

                                // insert old/original particle
                                childPath = 0;
                                if (particles->x[childIndex] < 0.5 * (min_x + max_x)) { childPath += 1; }
#if DIM > 1
                                if (particles->y[childIndex] < 0.5 * (min_y + max_y)) { childPath += 2; }
#if DIM == 3
                                if (particles->z[childIndex] < 0.5 * (min_z + max_z)) { childPath += 4; }
#endif
#endif

                                particles->x[cell] += particles->weightedEntry(childIndex, Entry::x);
#if DIM > 1
                                particles->y[cell] += particles->weightedEntry(childIndex, Entry::y);
#if DIM == 3
                                particles->z[cell] += particles->weightedEntry(childIndex, Entry::z);
#endif
#endif

                                //if (cell % 1000 == 0) {
                                //    printf("buildTree: x[%i] = (%f, %f, %f) from x[%i] = (%f, %f, %f) m = %f\n", cell, particles->x[cell], particles->y[cell],
                                //           particles->z[cell], childIndex, particles->x[childIndex], particles->y[childIndex],
                                //           particles->z[childIndex], particles->mass[childIndex]);
                                //}

                                particles->mass[cell] += particles->mass[childIndex];
                                tree->count[cell] += tree->count[childIndex];

                                tree->child[POW_DIM * cell + childPath] = childIndex;
                                tree->start[cell] = -1;

                                // insert new particle
                                temp = cell;
                                childPath = 0;

                                // find insertion point for body
                                if (particles->x[bodyIndex + offset] < 0.5 * (min_x + max_x)) {
                                    childPath += 1;
                                    max_x = 0.5 * (min_x + max_x);
                                } else {
                                    min_x = 0.5 * (min_x + max_x);
                                }
#if DIM > 1
                                if (particles->y[bodyIndex + offset] < 0.5 * (min_y + max_y)) {
                                    childPath += 2;
                                    max_y = 0.5 * (min_y + max_y);
                                } else {
                                    min_y = 0.5 * (min_y + max_y);
                                }
#if DIM == 3
                                if (particles->z[bodyIndex + offset] < 0.5 * (min_z + max_z)) {
                                    childPath += 4;
                                    max_z = 0.5 * (min_z + max_z);
                                } else {
                                    min_z = 0.5 * (min_z + max_z);
                                }
#endif
#endif

                                // COM / preparing for calculation of COM
                                if (particles->mass[bodyIndex + offset] != 0) {
                                    particles->x[cell] += particles->weightedEntry(bodyIndex + offset, Entry::x);
#if DIM > 1
                                    particles->y[cell] += particles->weightedEntry(bodyIndex + offset, Entry::y);
#if DIM == 3
                                    particles->z[cell] += particles->weightedEntry(bodyIndex + offset, Entry::z);
#endif
#endif
                                    particles->mass[cell] += particles->mass[bodyIndex + offset];
                                }
                                tree->count[cell] += tree->count[bodyIndex + offset];
                                childIndex = tree->child[POW_DIM * temp + childPath];
                            }

                            tree->child[POW_DIM * temp + childPath] = bodyIndex + offset;

                            __threadfence();  // written to global memory arrays (child, x, y, mass) thus need to fence
                            tree->child[locked] = patch;
                        }
                        offset += stride;
                        newBody = true;
                    }
                }
                __syncthreads();
            }

        }

        /*__global__ void insertReceivedParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                DomainList *domainList, DomainList *lowestDomainList, int n, int m) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;

            //note: -1 used as "null pointer"
            //note: -2 used to lock a child (pointer)

            integer offset;
            bool newBody = true;

            real min_x, max_x;
#if DIM > 1
            real min_y, max_y;
#if DIM == 3
            real min_z, max_z;
#endif
#endif

            integer childPath;
            integer temp;

            bool isDomainList = false;

            offset = 0;

            bodyIndex += tree->toDeleteLeaf[0];

            while ((bodyIndex + offset) < tree->toDeleteLeaf[1]) { // && (bodyIndex + offset) >= tree->toDeleteLeaf[0]) {

                if (newBody) {

                    newBody = false;
                    isDomainList = false;

                    min_x = *tree->minX;
                    max_x = *tree->maxX;
#if DIM > 1
                    min_y = *tree->minY;
                    max_y = *tree->maxY;
#if DIM == 3
                    min_z = *tree->minZ;
                    max_z = *tree->maxZ;
#endif
#endif

                    temp = 0;
                    childPath = 0;

                    // find insertion point for body
                    if (particles->x[bodyIndex + offset] < 0.5 * (min_x + max_x)) { // x direction
                        childPath += 1;
                        max_x = 0.5 * (min_x + max_x);
                    }
                    else {
                        min_x = 0.5 * (min_x + max_x);
                    }
#if DIM > 1
                    if (particles->y[bodyIndex + offset] < 0.5 * (min_y + max_y)) { // y direction
                        childPath += 2;
                        max_y = 0.5 * (min_y + max_y);
                    }
                    else {
                        min_y = 0.5 * (min_y + max_y);
                    }
#if DIM == 3
                    if (particles->z[bodyIndex + offset] < 0.5 * (min_z + max_z)) {  // z direction
                        childPath += 4;
                        max_z = 0.5 * (min_z + max_z);
                    }
                    else {
                        min_z = 0.5 * (min_z + max_z);
                    }
#endif
#endif
                }

                int childIndex = tree->child[temp*POW_DIM + childPath];

                // traverse tree until hitting leaf node
                while (childIndex >= m) { // && childIndex < (8*m)) { //formerly n

                    isDomainList = false;

                    temp = childIndex;

                    childPath = 0;

                    // find insertion point for body
                    if (particles->x[bodyIndex + offset] < 0.5 * (min_x + max_x)) { // x direction
                        childPath += 1;
                        max_x = 0.5 * (min_x + max_x);
                    }
                    else {
                        min_x = 0.5 * (min_x + max_x);
                    }
#if DIM > 1
                    if (particles->y[bodyIndex + offset] < 0.5 * (min_y + max_y)) { // y direction
                        childPath += 2;
                        max_y = 0.5 * (min_y + max_y);
                    }
                    else {
                        min_y = 0.5 * (min_y + max_y);
                    }
#if DIM == 3
                    if (particles->z[bodyIndex + offset] < 0.5 * (min_z + max_z)) { // z direction
                        childPath += 4;
                        max_z = 0.5 * (min_z + max_z);
                    }
                    else {
                        min_z = 0.5 * (min_z + max_z);
                    }
#endif
#endif

//                    for (int i=0; i<*domainList->domainListIndex; i++) {
//                        if (temp == domainList->domainListIndices[i]) {
//                            isDomainList = true;
//                            break;
//                        }
//                    }

                    //TODO: !!!
//                    if (!isDomainList) {
//                        if (particles->mass[bodyIndex + offset] != 0) {
//                            atomicAdd(&particles->x[temp], particles->mass[bodyIndex + offset] * particles->x[bodyIndex + offset]);
//#if DIM > 1
//                            atomicAdd(&particles->y[temp], particles->mass[bodyIndex + offset] * particles->y[bodyIndex + offset]);
//#if DIM == 3
//                            atomicAdd(&particles->z[temp], particles->mass[bodyIndex + offset] * particles->z[bodyIndex + offset]);
//#endif
//#endif
//                        }
//                        atomicAdd(&particles->mass[temp], particles->mass[bodyIndex + offset]);
//                        //atomicAdd(&count[temp], 1); // do not count, since particles are just temporarily saved on this process
//                    }
//                    atomicAdd(&tree->count[temp], 1); // do not count, since particles are just temporarily saved on this process
//                    childIndex = tree->child[POW_DIM*temp + childPath];
                }

                // if child is not locked
                if (childIndex != -2) {

                    int locked = temp * 8 + childPath;

                    //lock
                    if (atomicCAS(&tree->child[locked], childIndex, -2) == childIndex) {

                        // check whether a body is already stored at the location
                        if (childIndex == -1) {
                            //insert body and release lock
                            tree->child[locked] = bodyIndex + offset;
                        }
                        else {
                            int patch = POW_DIM * m; //8*n
                            while (childIndex >= 0 && childIndex < n) {

                                //debug
                                //if (x[childIndex] == x[bodyIndex + offset]) {
                                //    printf("ATTENTION (shouldn't happen...): x[%i] = (%f, %f, %f) vs. x[%i] = (%f, %f, %f) | to_delete_leaf = (%i, %i)\n",
                                //           childIndex, x[childIndex], y[childIndex], z[childIndex], bodyIndex + offset,  x[bodyIndex + offset],
                                //           y[bodyIndex + offset], z[bodyIndex + offset], to_delete_leaf[0], to_delete_leaf[1]);
                                //}

                                //create a new cell (by atomically requesting the next unused array index)
                                int cell = atomicAdd(tree->index, 1);

                                patch = min(patch, cell);

                                if (patch != cell) {
                                    tree->child[POW_DIM * temp + childPath] = cell;
                                }

//                                if (particles->x[childIndex] == particles->x[bodyIndex + offset] &&
//                                        particles->y[childIndex] == particles->y[bodyIndex + offset]) {
//                                    printf("[rank %i]ATTENTION!!! %i vs. %i\n", subDomainKeyTree->rank,
//                                           childIndex, bodyIndex + offset);
//                                    break;
//                                }

                                // insert old/original particle
                                childPath = 0;
                                if (particles->x[childIndex] < 0.5 * (min_x + max_x)) {
                                    childPath += 1;
                                }
#if DIM > 1
                                if (particles->y[childIndex] < 0.5 * (min_y + max_y)) {
                                    childPath += 2;
                                }
#if DIM == 3
                                if (particles->z[childIndex] < 0.5 * (min_z + max_z)) {
                                    childPath += 4;
                                }
#endif
#endif

//                                particles->x[cell] += particles->mass[childIndex] * particles->x[childIndex];
//#if DIM > 1
//                                particles->y[cell] += particles->mass[childIndex] * particles->y[childIndex];
//#if DIM == 3
//                                particles->z[cell] += particles->mass[childIndex] * particles->z[childIndex];
//#endif
//#endif
//
//                                particles->mass[cell] += particles->mass[childIndex];
//                                // do not count, since particles are just temporarily saved on this process
//                                tree->count[cell] += tree->count[childIndex];

                                tree->child[POW_DIM * cell + childPath] = childIndex;

                                //tree->start[cell] = -1; //TODO: resetting start needed in insertReceivedParticles()?

                                // insert new particle
                                temp = cell;
                                childPath = 0;

                                // find insertion point for body
                                if (particles->x[bodyIndex + offset] < 0.5 * (min_x + max_x)) {
                                    childPath += 1;
                                    max_x = 0.5 * (min_x + max_x);
                                } else {
                                    min_x = 0.5 * (min_x + max_x);
                                }
#if DIM > 1
                                if (particles->y[bodyIndex + offset] < 0.5 * (min_y + max_y)) {
                                    childPath += 2;
                                    max_y = 0.5 * (min_y + max_y);
                                } else {
                                    min_y = 0.5 * (min_y + max_y);
                                }
#if DIM == 3
                                if (particles->z[bodyIndex + offset] < 0.5 * (min_z + max_z)) {
                                    childPath += 4;
                                    max_z = 0.5 * (min_z + max_z);
                                } else {
                                    min_z = 0.5 * (min_z + max_z);
                                }
#endif
#endif

                               // /* // COM / preparing for calculation of COM
                               // if (particles->mass[bodyIndex + offset] != 0) {
                               //     particles->x[cell] += particles->mass[bodyIndex + offset] * particles->x[bodyIndex + offset];
//#if DIM > 1
                               //     particles->y[cell] += particles->mass[bodyIndex + offset] * particles->y[bodyIndex + offset];
//#if DIM == 3
                               //     particles->z[cell] += particles->mass[bodyIndex + offset] * particles->z[bodyIndex + offset];
//#endif
//#endif
                                //    particles->mass[cell] += particles->mass[bodyIndex + offset];
                                //}
                                // // do not count, since particles are just temporarily saved on this process
                                //tree->count[cell] += tree->count[bodyIndex + offset];

                                childIndex = tree->child[POW_DIM * temp + childPath];
                            }

                            tree->child[POW_DIM * temp + childPath] = bodyIndex + offset;

                            __threadfence();  // written to global memory arrays (child, x, y, mass) thus need to fence
                            tree->child[locked] = patch;
                        }
                        offset += stride;
                        newBody = true;
                    }
                    else {

                    }
                }
                else {

                }
                __syncthreads();
            }

        }*/

        __global__ void info(Tree *tree, Particles *particles, Helper *helper,
                             integer numParticlesLocal, integer numParticles, integer numNodes) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;
            integer i;

            while ((bodyIndex + offset) < numParticlesLocal) {
                if ((bodyIndex + offset) % 1000 == 0) {
                    i = 0;
                    printf("particles->noi[%i] = %i\n", bodyIndex + offset, particles->noi[bodyIndex + offset]);
                    //while (particles->nnl[(bodyIndex + offset) * MAX_NUM_INTERACTIONS + i] != -1 && i < MAX_NUM_INTERACTIONS) {
                    //    printf("particles->nnl[%i * %i + %i] = %i\n", bodyIndex + offset, MAX_NUM_INTERACTIONS, i,
                    //           particles->nnl[(bodyIndex + offset) * MAX_NUM_INTERACTIONS + i]);
                    //    i++;
                    //}
                }

                offset += stride;
            }
        }

        namespace Launch {
            real fixedRadiusNN(Tree *tree, Particles *particles, integer *interactions, integer numParticlesLocal,
                               integer numParticles, integer numNodes) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::SPH::Kernel::fixedRadiusNN, tree, particles, interactions,
                                    numParticlesLocal, numParticles, numNodes);
            }

            real fixedRadiusNN_Test(Tree *tree, Particles *particles, integer *interactions, integer numParticlesLocal,
                               integer numParticles, integer numNodes) {
                size_t sharedMemory = 2*sizeof(integer)*MAX_DEPTH;
                ExecutionPolicy executionPolicy(512, 256, sharedMemory);
                return cuda::launch(true, executionPolicy, ::SPH::Kernel::fixedRadiusNN, tree, particles, interactions,
                                    numParticlesLocal, numParticles, numNodes);
            }

            real compTheta(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                           DomainList *lowestDomainList, Curve::Type curveType) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::SPH::Kernel::compTheta, subDomainKeyTree, tree,
                                    particles, lowestDomainList, curveType);
            }

            real symbolicForce(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                               DomainList *lowestDomainList, integer *sendIndices,
                               integer n, integer m, integer relevantIndex,
                               Curve::Type curveType) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::SPH::Kernel::symbolicForce, subDomainKeyTree, tree, particles,
                                    lowestDomainList, sendIndices, n, m, relevantIndex, curveType);
            }

            real collectSendIndices(Tree *tree, Particles *particles, integer *sendIndices,
                                    integer *particles2Send, integer *particlesCount,
                                    integer n, integer length, Curve::Type curveType) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::SPH::Kernel::collectSendIndices, tree, particles,
                                    sendIndices, particles2Send, particlesCount, n, length, curveType);
            }

            real particles2Send(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                DomainList *domainList, DomainList *lowestDomainList, integer maxLevel,
                                integer *toSend, integer *sendCount, integer *alreadyInserted,
                                integer insertOffset,
                                integer numParticlesLocal, integer numParticles, integer numNodes, real radius,
                                Curve::Type curveType) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::SPH::Kernel::particles2Send, subDomainKeyTree, tree,
                                    particles, domainList, lowestDomainList, maxLevel, toSend, sendCount,
                                    alreadyInserted, insertOffset, numParticlesLocal, numParticles, numNodes,
                                    radius, curveType);
            }

            real collectSendIndicesBackup(integer *toSend, integer *toSendCollected, integer count) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::SPH::Kernel::collectSendIndicesBackup, toSend, toSendCollected,
                                    count);
            }

            real collectSendEntriesBackup(SubDomainKeyTree *subDomainKeyTree, real *entry, real *toSend, integer *sendIndices,
                                    integer *sendCount, integer totalSendCount, integer insertOffset) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::SPH::Kernel::collectSendEntriesBackup, subDomainKeyTree,
                                    entry, toSend, sendIndices, sendCount, totalSendCount, insertOffset);
            }

            real insertReceivedParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                 DomainList *domainList, DomainList *lowestDomainList, int n, int m) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::SPH::Kernel::insertReceivedParticles, subDomainKeyTree,
                                    tree, particles, domainList, lowestDomainList, n, m);
            }

            real info(Tree *tree, Particles *particles, Helper *helper,
                      integer numParticlesLocal, integer numParticles, integer numNodes) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::SPH::Kernel::info, tree, particles, helper,
                                    numParticlesLocal, numParticles, numNodes);
            }
        }
    }

}
