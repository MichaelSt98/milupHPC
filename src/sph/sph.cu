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
                r = fmaxf(x_temp, y_radius);
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
                                else if (fabs(dx) < interactionDistance &&) {
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
                r = fmaxf(x_temp, y_radius);
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
                                else if (fabs(dx) < interactionDistance &&) {
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

                        int path[MAX_LEVEL];
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

                //__threadfence();
                offset += stride;
            }

        }

        __global__ void collectSendIndices(integer *toSend, integer *toSendCollected, integer count) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            while ((bodyIndex + offset) < count) {
                toSendCollected[bodyIndex + offset] = toSend[bodyIndex + offset];
                //printf("toSendCollected[%i] = %i\n", bodyIndex + offset, toSendCollected[bodyIndex + offset]);
                offset += stride;
            }
        }

        __global__ void collectSendEntries(SubDomainKeyTree *subDomainKeyTree, real *entry, real *toSend,
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
                //printf("toSend[%i] = %f sendIndices = %i (insertOffset = %i)\n", bodyIndex + offset, toSend[bodyIndex + offset],
                //       sendIndices[bodyIndex + offset], insertOffset);
                offset += stride;
            }
        }

        __global__ void info(Tree *tree, Particles *particles, Helper *helper,
                             integer numParticlesLocal, integer numParticles, integer numNodes) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            while ((bodyIndex + offset) < numParticlesLocal) {
                if ((bodyIndex + offset) % 1000 == 0) {
                    printf("particles->nnl[%i * %i + %i] = %i\n", bodyIndex + offset, MAX_NUM_INTERACTIONS, 0,
                           particles->nnl[(bodyIndex + offset) * MAX_NUM_INTERACTIONS]);
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
                ExecutionPolicy executionPolicy(256, 256, sharedMemory);
                return cuda::launch(true, executionPolicy, ::SPH::Kernel::fixedRadiusNN, tree, particles, interactions,
                                    numParticlesLocal, numParticles, numNodes);
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

            real collectSendIndices(integer *toSend, integer *toSendCollected, integer count) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::SPH::Kernel::collectSendIndices, toSend, toSendCollected,
                                    count);
            }

            real collectSendEntries(SubDomainKeyTree *subDomainKeyTree, real *entry, real *toSend, integer *sendIndices,
                                    integer *sendCount, integer totalSendCount, integer insertOffset) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::SPH::Kernel::collectSendEntries, subDomainKeyTree,
                                    entry, toSend, sendIndices, sendCount, totalSendCount, insertOffset);
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
