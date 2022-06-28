#include "../../include/sph/sph.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"
//#include <cub/cub.cuh>

#define MAX_VARIABLE_SML_ITERATIONS 5
// tolerance value. if found number of interactions is as close as TOLERANCE_WANTED_NUMBER_OF_INTERACTIONS, we stop iterating
#define TOLERANCE_WANTED_NUMBER_OF_INTERACTIONS 5

namespace SPH {

    // deprecated
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

        // Brute-force method (you don't want to use this!)
        __global__ void fixedRadiusNN_bruteForce(Tree *tree, Particles *particles, integer *interactions, integer numParticlesLocal,
                                            integer numParticles, integer numNodes) {


            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            real dx, dy, dz;
            real distance;
            int numInteractions;

            while ((bodyIndex + offset) < numParticlesLocal) {

                numInteractions = 0;
                for (int i=0; i<tree->toDeleteLeaf[1]; ++i) {
                    if ((bodyIndex + offset) != i) {

                        dx = particles->x[bodyIndex + offset] - particles->x[i];
#if DIM > 1
                        dy = particles->y[bodyIndex + offset] - particles->y[i];
#if DIM == 3
                        dz = particles->z[bodyIndex + offset] - particles->z[i];
#endif
#endif

#if DIM == 1

#elif DIM == 2

#else //DIM == 3
                        distance = dx * dx + dy * dy + dz * dz;
#endif

                        if (distance < (particles->sml[bodyIndex + offset] * particles->sml[bodyIndex + offset]) &&
                            distance < (particles->sml[i] * particles->sml[i])) {
                            interactions[(bodyIndex + offset) * MAX_NUM_INTERACTIONS + numInteractions] = i;
                            numInteractions++;
                            //if ((bodyIndex + offset) % 1000 == 0) {
                            //    printf("distance: %e, %i vs. %i\n", distance, bodyIndex + offset, i);
                            //}
                            if (numInteractions > MAX_NUM_INTERACTIONS) {
                                //printf("numInteractions = %i > MAX_NUM_INTERACTIONS = %i\n", numInteractions,
                                //       MAX_NUM_INTERACTIONS);
                                cudaTerminate("numInteractions = %i > MAX_NUM_INTERACTIONS = %i\n", numInteractions,
                                              MAX_NUM_INTERACTIONS);
                            }
                        }
                    }
                }
                particles->noi[bodyIndex + offset] = numInteractions;
                offset += stride;
            }


        }

        __global__ void fixedRadiusNN(Tree *tree, Particles *particles, integer *interactions, real radius,
                                      integer numParticlesLocal, integer numParticles, integer numNodes) {

            register int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            register int stride = blockDim.x * gridDim.x;
            register int offset = 0;
            register int index;

            register integer childNumber, nodeIndex, depth, childIndex;

            register real dx, x;
            //real x_child;
#if DIM > 1
            register real dy, y;
            //real y_child;
#if DIM == 3
            register real dz, z;
            //real z_child;
#endif
#endif

            register real d, r, interactionDistance;

            register int noOfInteractions;

            register int currentNodeIndex[MAX_DEPTH];
            register int currentChildNumber[MAX_DEPTH];

            real sml, smlj;

            while ((bodyIndex + offset) < numParticlesLocal) {

                //index = tree->sorted[bodyIndex + offset];
                index = bodyIndex + offset;

                x = particles->x[index];
#if DIM > 1
                y = particles->y[index];
#if DIM == 3
                z = particles->z[index];
#endif
#endif
                sml = particles->sml[index];

                // resetting
                //for (integer i = 0; i < MAX_NUM_INTERACTIONS; i++) {
                //    interactions[(bodyIndex + offset) * MAX_NUM_INTERACTIONS + i] = -1;
                //}
                //numberOfInteractions[bodyIndex + offset] = 0;
                // end: resetting

                depth = 0;
                currentNodeIndex[depth] = 0; //numNodes - 1;
                currentChildNumber[depth] = 0;
                noOfInteractions = 0;

                r = radius * 0.5;

                interactionDistance = (r + sml);

                //bool remember;
                do {

                    childNumber = currentChildNumber[depth];
                    nodeIndex = currentNodeIndex[depth];

                    //remember = false;
                    while (childNumber < POW_DIM) {

                        childIndex = tree->child[POW_DIM * nodeIndex + childNumber];
                        childNumber++;

                        if (childIndex != -1 && childIndex != index) {

                            //x_child = particles->x[childIndex];
                            //#if DIM > 1
                            //y_child = particles->y[childIndex];
                            //#if DIM == 3
                            //z_child = particles->z[childIndex];
                            //#endif
                            //#endif

                            smlj = particles->sml[childIndex];

                            dx = x - particles->x[childIndex]; //x_child;
#if DIM > 1
                            dy = y - particles->y[childIndex]; //y_child;
#if DIM == 3
                            dz = z - particles->z[childIndex]; //z_child;
#endif
#endif
                            // its a leaf
                            if (childIndex < numParticles) {
#if DIM == 1
                                d = dx*dx;
#elif DIM == 2
                                d = dx*dx + dy*dy;
#else
                                d = dx*dx + dy*dy + dz*dz;
#endif

                                //if ((bodyIndex + offset) % 1000 == 0) {
                                    //printf("sph: index = %i, d = %i\n", bodyIndex+offset, d);
                                //}

                                if (d < (sml * sml) &&
                                    d < (smlj * smlj)
                                    /*&& noOfInteractions < MAX_NUM_INTERACTIONS*/) { //TODO: remove, just for testing purposes
                                    //printf("Adding interaction partner!\n");
                                    interactions[index * MAX_NUM_INTERACTIONS + noOfInteractions] = childIndex;
                                    // debug
                                    //if (noOfInteractions > MAX_NUM_INTERACTIONS) {
                                    //    printf("%i: noOfInteractions = %i > MAX_NUM_INTERACTIONS = %i (sml = %e) (%e, %e, %e)\n",
                                    //           bodyIndex + offset, noOfInteractions, MAX_NUM_INTERACTIONS, particles->sml[bodyIndex + offset],
                                    //           particles->x[bodyIndex + offset], particles->y[bodyIndex + offset],
                                    //           particles->y[bodyIndex + offset]);
                                    //    assert(0);
                                    //}
                                    // end: debug
                                    // debug: should not happen
                                    //if ((bodyIndex + offset) == childIndex) {
                                    //    printf("fixedRadiusNN: index = %i == childIndex = %i\n",
                                    //           bodyIndex + offset, childIndex);
                                    //    assert(0);
                                    //}
                                    // end: debug
                                    noOfInteractions++;
                                }
                            }
#if DIM == 1
                                else if (cuda::math::abs(dx) < interactionDistance) {
#elif DIM == 2
                                else if (cuda::math::abs(dx) < interactionDistance &&
                                     cuda::math::abs(dy) < interactionDistance) {
#else
                            else if (/*tree->child[POW_DIM * nodeIndex + childNumber] != -1 && */  // need to check, since there are nodes without any leaves as children
                                     /*(childNumber == currentChildNumber[depth-1] && nodeIndex == currentNodeIndex[depth-1]) &&*/  //TODO: just a fix, why is this happening at all?
                                    (cuda::math::abs(dx) < interactionDistance &&
                                     cuda::math::abs(dy) < interactionDistance &&
                                     cuda::math::abs(dz) < interactionDistance) || particles->nodeType[childIndex] >= 1) {
#endif

                                currentChildNumber[depth] = childNumber;
                                currentNodeIndex[depth] = nodeIndex;

                                //if (childNumber == currentChildNumber[depth-1] && nodeIndex == currentNodeIndex[depth-1]) {
                                //    printf("ATTENTION[%i]: current = before child = %i node = %i depth = %i tree->child = %i\n", bodyIndex + offset,
                                //           childNumber, nodeIndex, depth, tree->child[POW_DIM * nodeIndex + childNumber]);
                                    //assert(0);
                                //}
                                //if (depth < MAX_DEPTH) { //TODO: REMOVE!!! just to keep kernel from crashing as long as sml is not dynamic!
                                //    // put child on stack
                                //    depth++;
                                //}
                                depth++;
                                //if (particles->nodeType[childIndex] == 0) {
                                r *= 0.5;
                                //remember = true;
                                //}
                                interactionDistance = (r + sml);
                                if (depth > MAX_DEPTH) { //MAX_DEPTH) {
                                    //for (int i_depth=0; i_depth<depth; i_depth++) {
                                    //    printf("%i node[%i] = %i child[%i] = %i tree->child = %i\n", bodyIndex + offset, i_depth,
                                    //           currentNodeIndex[i_depth], i_depth, currentChildNumber[i_depth], tree->child[POW_DIM * currentNodeIndex[i_depth] + currentChildNumber[i_depth]]);
                                    //}
                                    // TODO: why not here redoNeighborSearch() ???
                                    cudaTerminate("depth = %i > MAX_DEPTH = %i\n", depth, MAX_DEPTH);
                                }
                                childNumber = 0;
                                nodeIndex = childIndex;
                            }
                        }
                    }

                    depth--;
                    //if (!remember) {
                    r *= 2.0;
                    //}
                    interactionDistance = (r + sml);

                } while (depth >= 0);

                particles->noi[index] = noOfInteractions;
                //printf("%i: noOfInteractions = %i\n", bodyIndex + offset, noOfInteractions);

                offset += stride;
                __syncthreads();
            }
        }

        __global__ void fixedRadiusNN_withinBox(Tree *tree, Particles *particles, integer *interactions, integer numParticlesLocal,
                                      integer numParticles, integer numNodes) {

            int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;
            int offset = 0;
            int index;

            //if ((bodyIndex + offset) == 0) {
            //    printf("new fixed radius...\n");
            //}

            integer childNumber, nodeIndex, childIndex;

            real tmp1, tmp2;
            register real dx, min_x, max_x, x_temp, x_child, min_dx, max_dx, x;
#if DIM > 1
            register real dy, min_y, max_y, y_temp, y_child, min_dy, max_dy, y;
#if DIM == 3
            register real dz, min_z, max_z, z_temp, z_child, min_dz, max_dz, z;
#endif
#endif
            register real sml;
            real d;
            real min_distance;
            real max_distance;

            // outer stack
            register int outer_currentNodeIndex[MAX_DEPTH];
            register int outer_currentChildNumber[MAX_DEPTH];
            register int outer_currentNodeLevel[MAX_DEPTH];

            // inner stack
            //integer inner_currentNodeIndex[MAX_DEPTH];
            //integer inner_currentChildNumber[MAX_DEPTH];
            // but reuse (outer stack!)
            int *inner_currentNodeIndex;
            int *inner_currentChildNumber;

            int depth, inner_depth;
            int noOfInteractions;

            int level;

            while ((bodyIndex + offset) < numParticlesLocal) {

                index = bodyIndex + offset;
                //index = tree->sorted[bodyIndex + offset];
                //if (tree->sorted[bodyIndex + offset] < 0 || tree->sorted[bodyIndex + offset] > numParticlesLocal) {
                    //printf("sorted[%i] = %i\n", index, tree->sorted[bodyIndex + offset]);
                //}

                x = particles->x[index];
#if DIM > 1
                y = particles->y[index];
#if DIM == 3
                z = particles->z[index];
#endif
#endif
                sml = particles->sml[index];

                depth = 0;
                outer_currentNodeIndex[depth] = 0;
                outer_currentChildNumber[depth] = 0;
                outer_currentNodeLevel[depth] = 1;
                noOfInteractions = 0;
                level = 1;

                do {
                    childNumber = outer_currentChildNumber[depth];
                    nodeIndex = outer_currentNodeIndex[depth];
                    level = outer_currentNodeLevel[depth];

                    while (childNumber < POW_DIM) {

                        childIndex = tree->child[POW_DIM * nodeIndex + childNumber];
                        childNumber++;

                        if (childIndex != -1 && childIndex != (index)) {

                            x_child = particles->x[childIndex];
#if DIM > 1
                            y_child = particles->y[childIndex];
#if DIM == 3
                            z_child = particles->z[childIndex];
#endif
#endif

                            if (childIndex >= numParticles) {

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
                                for (int _level=0; _level < level; ++_level) {
                                    if (x_child < 0.5 * (min_x + max_x)) {
                                        max_x = 0.5 * (min_x + max_x);
                                    }
                                    else {
                                        min_x = 0.5 * (min_x + max_x);
                                    }
#if DIM > 1
                                    if (y_child < 0.5 * (min_y + max_y)) {
                                        max_y = 0.5 * (min_y + max_y);
                                    }
                                    else {
                                        min_y = 0.5 * (min_y + max_y);
                                    }
#if DIM == 3
                                    if (z_child < 0.5 * (min_z + max_z)) {
                                        max_z = 0.5 * (min_z + max_z);
                                    }
                                    else {
                                        min_z = 0.5 * (min_z + max_z);
                                    }
#endif
#endif
                                }

                                if (x < min_x) {
                                    min_dx = x - min_x;
                                    max_dx = x - max_x;
                                } else if (x > max_x) {
                                    min_dx = x - max_x;
                                    max_dx = x - min_x;
                                } else {
                                    min_dx = 0.f;
                                    //max_dx = (cuda::math::abs(x-min_x) > cuda::math::abs(x-max_x)) ? cuda::math::abs(x-min_x) : cuda::math::abs(x-max_x);
                                    tmp1 = cuda::math::abs(x-min_x);
                                    tmp2 = cuda::math::abs(x-max_x);
                                    max_dx = (tmp1 > tmp2) ? tmp1 : tmp2;
                                }
#if DIM > 1
                                if (y < min_y) {
                                    min_dy = y - min_y;
                                    max_dy = y - max_y;
                                } else if (y > max_y) {
                                    min_dy = y - max_y;
                                    max_dy = y - min_y;
                                } else {
                                    min_dy = 0.f;
                                    //max_dy = (cuda::math::abs(y-min_y) > cuda::math::abs(y-max_y)) ? cuda::math::abs(y-min_y) : cuda::math::abs(y-max_y);
                                    tmp1 = cuda::math::abs(y-min_y);
                                    tmp2 = cuda::math::abs(y-max_y);
                                    max_dy = (tmp1 > tmp2) ? tmp1 : tmp2;
                                }
#if DIM == 3
                                if (z < min_z) {
                                    min_dz = z - min_z;
                                    max_dz = z - max_z;
                                } else if (z > max_z) {
                                    min_dz = z - max_z;
                                    max_dz = z - min_z;
                                } else {
                                    min_dz = 0.f;
                                    //max_dz = (cuda::math::abs(z-min_z) > cuda::math::abs(z-max_z)) ? cuda::math::abs(z-min_z) : cuda::math::abs(z-max_z);
                                    tmp1 = cuda::math::abs(z-min_z);
                                    tmp2 = cuda::math::abs(z-max_z);
                                    max_dz = (tmp1 > tmp2) ? tmp1 : tmp2;
                                }

#endif
#endif
#if DIM == 1
                                //min_distance = cuda::math::sqrt(dx*dx);
                                min_distance = min_dx*min_dx;
                                max_distance = max_dx*max_dx;
#elif DIM == 2
                                //r = cuda::math::sqrt(dx*dx + dy*dy);
                                min_distance = min_dx*min_dx + min_dy*min_dy;
                                max_distance = max_dx*max_dx + max_dy*max_dy;
#else
                                min_distance = min_dx*min_dx + min_dy*min_dy + min_dz*min_dz;
                                max_distance = max_dx*max_dx + max_dy*max_dy + max_dz*max_dz;
#endif
                            }
                            else {
                                min_distance = 0;
                                max_distance = 0;
                            }

                            //if (index % 1000 == 0) {
                            //    printf("level: %i | d: %e, min_distance: %e, max_distance: %e\n", level, d, min_distance, max_distance);
                            //}
                            // its a leaf

                            //if (index % 1000 == 0) { printf("child on stack? %e > %e\n", particles->sml[index], min_distance); }

                            if (childIndex < numParticles) {

                                dx = x - x_child;
#if DIM > 1
                                dy = y - y_child;
#if DIM == 3
                                dz = z - z_child;
#endif
#endif

#if DIM == 1
                                d = dx*dx;
#elif DIM == 2
                                d = dx*dx + dy*dy;
#else
                                d = dx*dx + dy*dy + dz*dz;
#endif

                                if (d < (sml * sml) &&
                                    d < (particles->sml[childIndex] * particles->sml[childIndex])) {
                                    interactions[index * MAX_NUM_INTERACTIONS + noOfInteractions] = childIndex;
                                    //if (index % 1000 == 0) { printf("%i adding interaction #%i...\n", index, noOfInteractions); }
                                    noOfInteractions++;
                                    if (noOfInteractions > MAX_NUM_INTERACTIONS) {
                                        cudaTerminate("noOfInteractions = %i > MAX_NUM_INTERACTIONS = %i\n",
                                                   noOfInteractions, MAX_NUM_INTERACTIONS);
                                    }
                                }
                            }
                            // box at least partly within sml, thus add to exlicity stack
                            else if ((sml*sml) > min_distance) {
                                // box completely within sml
                                if ((sml*sml) >= max_distance) {

                                    inner_currentNodeIndex = &outer_currentNodeIndex[depth];
                                    inner_currentChildNumber = &outer_currentChildNumber[depth];

                                    inner_depth = 0;
                                    inner_currentNodeIndex[inner_depth] = childIndex; //nodeIndex;
                                    inner_currentChildNumber[inner_depth] = 0; //childNumber;
                                    int inner_childNumber; //= childNumber;
                                    int inner_nodeIndex; // = nodeIndex;
                                    int inner_childIndex;
                                    do {
                                        //inner_currentNodeIndex[depth] = nodeIndex;
                                        //inner_currentChildNumber[depth] = childNumber;
                                        inner_childNumber = inner_currentChildNumber[inner_depth];
                                        inner_nodeIndex = inner_currentNodeIndex[inner_depth];

                                        while (inner_childNumber < POW_DIM) {

                                            inner_childIndex = tree->child[POW_DIM * inner_nodeIndex + inner_childNumber];
                                            inner_childNumber++;

                                            if (inner_childIndex != -1 && inner_childIndex != (index)) {
                                                if (inner_childIndex < numParticles) {
                                                    interactions[index * MAX_NUM_INTERACTIONS +
                                                                 noOfInteractions] = inner_childIndex;
                                                    noOfInteractions++;
                                                    //printf("%i: adding directly: %i, depth: %i, child: %i\n", index, inner_childIndex, inner_depth, inner_childNumber);
                                                    if (noOfInteractions > MAX_NUM_INTERACTIONS) {
                                                        cudaTerminate("noOfInteractions = %i > MAX_NUM_INTERACTIONS = %i\n",
                                                                      noOfInteractions, MAX_NUM_INTERACTIONS);
                                                    }
                                                } else {
                                                    //printf("%i: directly on stack: %i, depth: %i, child: %i\n", index, inner_childIndex, inner_depth, inner_childNumber);
                                                    inner_currentChildNumber[inner_depth] = inner_childNumber;
                                                    inner_currentNodeIndex[inner_depth] = inner_nodeIndex;
                                                    inner_depth++;

                                                    inner_childNumber = 0;
                                                    inner_nodeIndex = inner_childIndex;
                                                }
                                            }
                                        }
                                        inner_depth--;
                                        //printf("%i: directly depth--: %i, depth: %i\n", inner_childIndex, inner_depth);
                                    } while (inner_depth >= 0);
                                    //printf("%i: added directly: %i (counter: %i, maxDistance: %e, sml: %e)\n", index, numParticlesDirectly, counter, max_distance, particles->sml[index]);
                                }
                                // box only partly within sml, thus add to exlicity stack
                                else {
                                    // put child on stack
                                    outer_currentChildNumber[depth] = childNumber;
                                    outer_currentNodeIndex[depth] = nodeIndex;
                                    outer_currentNodeLevel[depth] = level; // + 1;
                                    level++;
                                    depth++;

                                    if (depth > MAX_DEPTH) {
                                        cudaTerminate("depth = %i > MAX_DEPTH = %i\n", depth, MAX_DEPTH);
                                    }
                                    childNumber = 0;
                                    nodeIndex = childIndex;
                                }
                            }
                        }
                    }

                    depth--;

                } while (depth >= 0);

                particles->noi[index] = noOfInteractions;
                //printf("%i: noOfInteractions = %i\n", bodyIndex + offset, noOfInteractions);
                //if (index % 1000 == 0) {
                //    printf("noi[%i]: %i\n", index, noOfInteractions);
                //}
                offset += stride;
            }

        }

        __global__ void
        fixedRadiusNN_sharedMemory(Tree *tree, Particles *particles, integer *interactions, integer numParticlesLocal,
                           integer numParticles, integer numNodes) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            integer childNumber, nodeIndex, depth, childIndex;

            register real dx, x_radius;
#if DIM > 1
            register real dy, y_radius;
#if DIM == 3
            register real dz, z_radius;
            real r_temp;
#endif
#endif

            real d, r, interactionDistance;

            integer noOfInteractions;

            extern __shared__ int buffer[];
            integer *currentNodeIndex = (int*)&buffer[threadIdx.x * MAX_DEPTH];
            integer *currentChildNumber = (int*)&currentNodeIndex[(10 + threadIdx.x) * MAX_DEPTH];

            //register int currentNodeIndex[MAX_DEPTH];
            //register int currentChildNumber[MAX_DEPTH];

            while ((bodyIndex + offset) < numParticlesLocal) {

                // resetting
                //#pragma unroll
                //for (integer i = 0; i < MAX_NUM_INTERACTIONS; i++) {
                //    interactions[(bodyIndex + offset) * MAX_NUM_INTERACTIONS + i] = -1;
                //}
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
                r = cuda::math::max(x_radius, y_radius);
#else
                r_temp = cuda::math::max(x_radius, y_radius);
                r = cuda::math::max(r_temp, z_radius) * 0.5; //TODO: (0.5 * r) or (1.0 * r)
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
                                d = dx*dx + dy*dy + dz*dz;
#endif

                                if ((bodyIndex + offset) % 1000 == 0) {
                                    //printf("sph: index = %i, d = %i\n", bodyIndex+offset, d);
                                }

                                if (d < (particles->sml[bodyIndex + offset] * particles->sml[bodyIndex + offset])
                                    && d < (particles->sml[childIndex] * particles->sml[childIndex])) {
                                    //printf("Adding interaction partner!\n");
                                    interactions[(bodyIndex + offset) * MAX_NUM_INTERACTIONS +
                                                 noOfInteractions] = childIndex;
                                    noOfInteractions++;
                                }
                            }
#if DIM == 1
                                else if (cuda::math::abs(dx) < interactionDistance ||
                                        particles->nodeType[childIndex] >= 1) {
#elif DIM == 2
                                else if ((cuda::math::abs(dx) < interactionDistance &&
                                     cuda::math::abs(dy) < interactionDistance) ||
                                     particles->nodeType[childIndex] >= 1) {
#else
                            else if ((cuda::math::abs(dx) < interactionDistance &&
                                     cuda::math::abs(dy) < interactionDistance &&
                                     cuda::math::abs(dz) < interactionDistance) ||
                                     particles->nodeType[childIndex] >= 1) {
#endif
                                // put child on stack
                                currentChildNumber[depth] = childNumber;
                                currentNodeIndex[depth] = nodeIndex;
                                depth++;
                                r *= 0.5;
                                interactionDistance = (r + particles->sml[bodyIndex + offset]);

                                if (depth > MAX_DEPTH) {
                                    cudaTerminate("depth = %i > MAX_DEPTH = %i\n", depth, MAX_DEPTH);
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

                //if (noOfInteractions < 30) {
                //    particles->sml[bodyIndex + offset] *= 1.5;
                //}

                __syncthreads();
                offset += stride;
            }
        }

        __global__ void fixedRadiusNN_variableSML(Material *materials, Tree *tree, Particles *particles, integer *interactions,
                                                  integer numParticlesLocal, integer numParticles,
                                                  integer numNodes) {

            /*register */int i, inc, nodeIndex, depth, childNumber, child;
            /*register */int currentNodeIndex[MAX_DEPTH];
            /*register */int currentChildNumber[MAX_DEPTH];
            /*register */int numberOfInteractions;

            /*register */real x, dx, interactionDistance, r, radius, d;
#if DIM > 1
            /*register */real y, dy;
#if DIM == 3
            /*register */real z, dz;
#endif
#endif

            real x_radius;
#if DIM > 1
            real y_radius;
#if DIM == 3
            real z_radius;
            real r_temp;
#endif
#endif

            x_radius = /*0.5 * */(*tree->maxX - (*tree->minX));
#if DIM > 1
            y_radius = /*0.5 * */(*tree->maxY - (*tree->minY));
#if DIM == 3
            z_radius = /*0.5 * */(*tree->maxZ - (*tree->minZ));
#endif
#endif
#if DIM == 1
            radius = x_radius;
#elif DIM == 2
            radius = cuda::math::max(x_radius, y_radius);
#else
            r_temp = cuda::math::max(x_radius, y_radius);
            radius = cuda::math::max(r_temp, z_radius); //TODO: (0.5 * r) or (1.0 * r)
#endif


            inc = blockDim.x * gridDim.x;
            /* loop over all particles */
            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticlesLocal; i += inc) {

                x = particles->x[i];
#if DIM > 1
                y = particles->y[i];
#if DIM == 3
                z = particles->z[i];
#endif
#endif

#if DIM == 1
                radius = x_radius;
#elif DIM == 2
                radius = cuda::math::max(x_radius, y_radius);
#else
                r_temp = cuda::math::max(x_radius, y_radius);
                radius = cuda::math::max(r_temp, z_radius); //TODO: (0.5 * r) or (1.0 * r)
#endif

                /*volatile */bool found = false;
                /*register */int nit = -1;

                real htmp, htmpold;
                /*volatile */real htmpj;

                htmp = particles->sml[i];

                // look for nice sml
                while (!found) {
                    numberOfInteractions = 0;
                    nit++;
                    depth = 0;
                    currentNodeIndex[depth] = 0; //numNodes - 1;
                    currentChildNumber[depth] = 0;
                    numberOfInteractions = 0;
                    r = radius; // * 0.5; // because we start with root children
                    interactionDistance = (r + htmp);
                    do {

                        childNumber = currentChildNumber[depth];
                        nodeIndex = currentNodeIndex[depth];

                        while (childNumber < POW_DIM) {
                            child = tree->child[POW_DIM * nodeIndex + childNumber];
                            childNumber++;

                            if (child != -1 && child != i) {
                                dx = x - particles->x[child];
#if DIM > 1
                                dy = y - particles->y[child];
#if DIM == 3
                                dz = z - particles->z[child];
#endif
#endif
                                if (child < numParticles) {
                                    d = dx*dx;
#if DIM > 1
                                    d += dy*dy;
#if DIM == 3
                                    d += dz*dz;
#endif
#endif
                                    htmpj = particles->sml[child];

                                    if (d < htmp*htmp && d < htmpj*htmpj) {
                                        numberOfInteractions++;
                                    }
                                } else if (/*tree->child[POW_DIM * nodeIndex + childNumber] != -1 &&*/  // need to check, since there are nodes without any leaves as children
                                           /*(childNumber == currentChildNumber[depth-1] && nodeIndex == currentNodeIndex[depth-1]) &&*/  //TODO: just a fix, why is this happening at all?
                                        (cuda::math::abs(dx) < interactionDistance
#if DIM > 1
                                            && cuda::math::abs(dy) < interactionDistance
#if DIM == 3
                                            && cuda::math::abs(dz) < interactionDistance
#endif
#endif
                                        ) || particles->nodeType[child] >= 1) {
                                    // put child on stack
                                    currentChildNumber[depth] = childNumber;
                                    currentNodeIndex[depth] = nodeIndex;
                                    //if (depth > 500) {
                                    //    printf("debug: i: depth: %i for %i (%e, %e, %e) node: %i child: %i numP = %i\n", depth, i,
                                    //           particles->x[i], particles->y[i], particles->z[i], nodeIndex, childNumber,
                                    //           numParticlesLocal);
                                    //}
                                    depth++;
                                    r *= 0.5;
                                    interactionDistance = (r + htmp);
                                    if (depth > MAX_DEPTH) {
                                        //for (int i_depth=0; i_depth<depth; i_depth++) {
                                        //    printf("%i node[%i] = %i child[%i] = %i tree->child = %i\n", i, i_depth,
                                        //           currentNodeIndex[i_depth], i_depth, currentChildNumber[i_depth], tree->child[POW_DIM * currentNodeIndex[i_depth] + currentChildNumber[i_depth]]);
                                        //}
                                        cudaTerminate("depth = %i > MAX_DEPTH = %i\n", depth, MAX_DEPTH);
                                    }
                                    /*if (depth >= MAX_DEPTH) {
                                        printf("Error, maxdepth reached! problem in tree during interaction search");
                                        printf("??? child: %i i: %i depth: %i child[8 * %i + %i] = %i (%e, %e, %e)\n", child, i, depth - 11, currentNodeIndex[depth - 11],
                                               currentChildNumber[depth - 11], tree->child[POW_DIM * currentNodeIndex[depth - 11] + currentChildNumber[depth - 11]],
                                               particles->x[tree->child[POW_DIM * currentNodeIndex[depth - 11] + currentChildNumber[depth - 11]]],
                                               particles->y[tree->child[POW_DIM * currentNodeIndex[depth - 11] + currentChildNumber[depth - 11]]],
                                               particles->z[tree->child[POW_DIM * currentNodeIndex[depth - 11] + currentChildNumber[depth - 11]]]);
                                        printf("??? child: %i i: %i depth: %i child[8 * %i + %i] = %i (%e, %e, %e)\n", child, i, depth - 10, currentNodeIndex[depth - 10],
                                               currentChildNumber[depth - 10], tree->child[POW_DIM * currentNodeIndex[depth - 10] + currentChildNumber[depth - 10]],
                                               particles->x[tree->child[POW_DIM * currentNodeIndex[depth - 10] + currentChildNumber[depth - 11]]],
                                               particles->y[tree->child[POW_DIM * currentNodeIndex[depth - 10] + currentChildNumber[depth - 11]]],
                                               particles->z[tree->child[POW_DIM * currentNodeIndex[depth - 10] + currentChildNumber[depth - 11]]]);
                                        //for (int m=0; m<MAX_DEPTH; m++) {
                                        //    printf("??? %i depth: %i currentDepth: %i node: %i path: %i\n", i, MAX_DEPTH, m,
                                        //           currentNodeIndex[m], currentChildNumber[m]);
                                        //}
                                        //assert(depth < MAX_DEPTH);
                                        assert(0);
                                    }*/
                                    childNumber = 0;
                                    nodeIndex = child;
                                }
                            }
                        }
                        depth--;
                        r *= 2.0;
                        interactionDistance = (r + htmp);
                    } while (depth >= 0);

                    htmpold = htmp;
                    //printf("%d %d %e\n", i, numberOfInteractions, htmp);
                    // stop if we have the desired number of interaction partners \pm TOLERANCE_WANTED_NUMBER_OF_INTERACTIONS
                    if ((nit > MAX_VARIABLE_SML_ITERATIONS ||
                        abs(numberOfInteractions - materials[particles->materialId[i]].interactions) < TOLERANCE_WANTED_NUMBER_OF_INTERACTIONS )
                        && numberOfInteractions < MAX_NUM_INTERACTIONS) {

                        found = true;
                        particles->sml[i] = htmp;

                    } else if (numberOfInteractions >= MAX_NUM_INTERACTIONS) {
                        htmpold = htmp;
                        if (numberOfInteractions < 1)
                            numberOfInteractions = 1;
                        htmp *= 0.5 *  ( 1.0 + pow( (real) materials[particles->materialId[i]].interactions/ (real) numberOfInteractions, 1./DIM));
                        //printf("htmp *= 0.5 * %f\n", ( 1.0 + pow( (real) materials[particles->materialId[i]].interactions/ (real) numberOfInteractions, 1./DIM)));

                    } else {
                        // lower or raise htmp accordingly
                        if (numberOfInteractions < 1)
                            numberOfInteractions = 1;

                        htmpold = htmp;
                        htmp *= 0.5 *  ( 1.0 + pow( (real) materials[particles->materialId[i]].interactions/ (real) numberOfInteractions, 1./DIM));
                        //printf("htmp *= 0.5 * %f\n", ( 1.0 + pow( (real) materials[particles->materialId[i]].interactions/ (real) numberOfInteractions, 1./DIM)));
                    }

                    if (htmp < 1e-20) {
#if DIM == 3
                        printf("+++ particle: %d it: %d htmp: %e htmpold: %e wanted: %d current: %d mId: %d uid: %i (%e, %e, %e) n = %i\n", i, nit,
                                htmp, htmpold, materials[particles->materialId[i]].interactions, numberOfInteractions, particles->materialId[i],
                                particles->uid[i], particles->x[i], particles->y[i], particles->z[i], numParticlesLocal);
#endif
                    }

                }
                if (numberOfInteractions > MAX_NUM_INTERACTIONS || numberOfInteractions == 0) {
#if DIM == 3
                    printf("+++ particle: %d it: %d htmp: %e htmpold: %e wanted: %d current: %d mId: %d uid: %i (%e, %e, %e) n = %i\n",
                           i, nit,
                           htmp, htmpold, materials[particles->materialId[i]].interactions, numberOfInteractions,
                           particles->materialId[i],
                           particles->uid[i], particles->x[i], particles->y[i], particles->z[i], numParticlesLocal);
#endif
                    cudaTerminate("numberOfInteractions = %i > MAX_NUM_INTERACTIONS = %i\n", numberOfInteractions,
                                  MAX_NUM_INTERACTIONS);
                }
                //if (numberOfInteractions == 0) {
                //if (i % 1000 == 0) {
                //    printf("noi: %d: %d %e (nit: %i)\n", i, numberOfInteractions, particles->sml[i], nit);
                //}
            }

        }

        __device__ void redoNeighborSearch(Tree *tree, Particles *particles, int particleId,
                                             int *interactions, real radius, integer numParticles, integer numNodes) {

            register int i, inc, nodeIndex, depth, childNumber, child;
            i = particleId;
            register real x, dx, interactionDistance, r, d;
            x = particles->x[i];
#if DIM > 1
            register real y, dy;
            y = particles->y[i];
#if DIM == 3
            register real z, dz;
            z = particles->z[i];
#endif
#endif
            register int currentNodeIndex[MAX_DEPTH];
            register int currentChildNumber[MAX_DEPTH];
            register int numberOfInteractions;

            //printf("1) sml_new > h: noi: %d\n", p.noi[i]);

            real sml; // smoothing length of particle
            real smlj; // smoothing length of potential interaction partner

            // start at root
            depth = 0;
            currentNodeIndex[depth] = 0; //numNodes - 1;
            currentChildNumber[depth] = 0;
            numberOfInteractions = 0;
            r = radius * 0.5; // because we start with root children
            sml = particles->sml[i];
            particles->noi[i] = 0;
            interactionDistance = (r + sml);

            do {
                childNumber = currentChildNumber[depth];
                nodeIndex = currentNodeIndex[depth];
                while (childNumber < POW_DIM) {
                    child = tree->child[POW_DIM * nodeIndex + childNumber];
                    childNumber++;
                    if (child != -1 && child != i) {
                        dx = x - particles->x[child];
#if DIM > 1
                        dy = y - particles->y[child];
#if DIM == 3
                        dz = z - particles->z[child];
#endif
#endif

                        if (child < numParticles) {
                            //if (p_rhs.materialId[child] == EOS_TYPE_IGNORE) {
                            //    continue;
                            //}
                            d = dx * dx;
#if DIM > 1
                            d += dy * dy;
#if DIM == 3
                            d += dz * dz;
#endif
#endif
                            smlj = particles->sml[child];

                            if (d < sml*sml && d < smlj*smlj) {
                                interactions[i * MAX_NUM_INTERACTIONS + numberOfInteractions] = child;
                                numberOfInteractions++;
//#if TOO_MANY_INTERACTIONS_KILL_PARTICLE
//                                if (numberOfInteractions >= MAX_NUM_INTERACTIONS) {
//                            printf("setting the smoothing length for particle %d to 0!\n", i);
//                            p.h[i] = 0.0;
//                            p.noi[i] = 0;
//                            sml = 0.0;
//                            interactionDistance = 0.0;
//                            p_rhs.materialId[i] = EOS_TYPE_IGNORE;
//                            // continue with next particle by setting depth to -1
//                            // cms 2018-01-19
//                            depth = -1;
//                            break;
//                        }
//#endif
                            }
                        } else if (cuda::math::abs(dx) < interactionDistance
#if DIM > 1
                                   && cuda::math::abs(dy) < interactionDistance
#if DIM == 3
                                   && cuda::math::abs(dz) < interactionDistance
#endif
#endif
                                ) {
                            // put child on stack
                            currentChildNumber[depth] = childNumber;
                            currentNodeIndex[depth] = nodeIndex;
                            depth++;
                            r *= 0.5;
                            interactionDistance = (r + sml);
                            if (depth > MAX_DEPTH) {
                                cudaTerminate("depth = %i > MAX_DEPTH = %i\n", depth, MAX_DEPTH);
                            }
                            childNumber = 0;
                            nodeIndex = child;
                        }
                    }
                }

                depth--;
                r *= 2.0;
                interactionDistance = (r + sml);
            } while (depth >= 0);

            if (numberOfInteractions >= MAX_NUM_INTERACTIONS) {
                printf("ERROR: Maximum number of interactions exceeded: %d / %d\n", numberOfInteractions, MAX_NUM_INTERACTIONS);
//#if !TOO_MANY_INTERACTIONS_KILL_PARTICLE
//                // assert(numberOfInteractions < MAX_NUM_INTERACTIONS);
//#endif
            }
            particles->noi[i] = numberOfInteractions;
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
                //key = tree->getParticleKey(particles, bodyIndex, MAX_LEVEL, curveType);
                //printf("x = %e, %e, %e\n", particles->x[bodyIndex], particles->y[bodyIndex], particles->y[bodyIndex]);
                //if domain list node belongs to other process: add to relevant domain list indices
                // TODO: is this the problem?
                //proc = subDomainKeyTree->key2proc(key);
                if (curveType == Curve::Type::lebesgue) {
                    proc = subDomainKeyTree->key2proc(lowestDomainList->domainListKeys[index + offset]);
                }
                else {
                    proc = subDomainKeyTree->key2proc(KeyNS::lebesgue2hilbert(lowestDomainList->domainListKeys[index + offset], MAX_LEVEL, lowestDomainList->domainListLevels[index + offset]));
                }
                //printf("[rank %i] sph: proc = %i, bodyIndex = %i\n", subDomainKeyTree->rank, proc, bodyIndex);
                if (proc != subDomainKeyTree->rank) {
                    //printf("[rank %i] sph: proc = %i, bodyIndex = %i level = %i (%e, %e, %e)\n", subDomainKeyTree->rank,
                    //       proc, bodyIndex, lowestDomainList->domainListLevels[index + offset], particles->x[bodyIndex],
                    //       particles->y[bodyIndex], particles->z[bodyIndex]);
                    domainIndex = atomicAdd(lowestDomainList->domainListCounter, 1);
                    //printf("[rank %i] sph: domainIndex = %i\n", subDomainKeyTree->rank, domainIndex);
                    lowestDomainList->relevantDomainListIndices[domainIndex] = bodyIndex;
                    lowestDomainList->relevantDomainListLevels[domainIndex] = lowestDomainList->domainListLevels[index + offset];
                    lowestDomainList->relevantDomainListProcess[domainIndex] = proc;
                    lowestDomainList->relevantDomainListOriginalIndex[domainIndex] = index + offset;

                    //printf("[rank %i] Adding relevant domain list node: %i (%f, %f, %f)\n", subDomainKeyTree->rank,
                    //       bodyIndex, particles->x[bodyIndex],
                    //       particles->y[bodyIndex], particles->z[bodyIndex]);
                }
                offset += stride;
            }
        }


        __global__ void symbolicForce(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                      DomainList *lowestDomainList, integer *sendIndices, real searchRadius,
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
                //printf("sph: lowestDomainList: (%e, %e, %e)\n", particles->x[lowestDomainList->relevantDomainListIndices[relevantIndex]],
                //       particles->y[lowestDomainList->relevantDomainListIndices[relevantIndex]], particles->z[lowestDomainList->relevantDomainListIndices[relevantIndex]]);

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
                d = dx*dx + dy*dy + dz*dz;
#endif

                //if ((bodyIndex + offset) % 500 == 0) {
                //    printf("d = %e < (%e * %e = %e) sml = %e\n", d, searchRadius, searchRadius,
                //           searchRadius * searchRadius, particles->sml[bodyIndex + offset]);
                //}
                //if (d < (particles->sml[bodyIndex + offset] * particles->sml[bodyIndex + offset])) {
                if (d < (searchRadius * searchRadius)) {
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

        // TODO: SPH::symbolicForce_test version
        //  - dispatch all (lowest) domain list nodes for one process directly
        //  - min, max via memory not via computing
        __global__ void symbolicForce_test(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                      DomainList *lowestDomainList, integer *sendIndices, real searchRadius,
                                      integer n, integer m, integer relevantProc, integer relevantIndicesCounter,
                                      integer relevantIndexOld, Curve::Type curveType) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            //integer insertIndex;
            //integer insertIndexOffset;

            //integer currentChild;
            //integer childPath;
            integer currentParticleIndex;

            real dx, min_x, max_x;
#if DIM > 1
            real dy, min_y, max_y;
#if DIM == 3
            real dz, min_z, max_z;
#endif
#endif
            real d;

            bool added;

            while ((bodyIndex + offset) < n) {

                added = false;

                for (int relevantIndex = 0; relevantIndex < relevantIndicesCounter; relevantIndex++) {

                    if (lowestDomainList->relevantDomainListProcess[relevantIndex] != relevantProc) {
                        continue;
                    }

                    if (added) {
                        break;
                    }

                    currentParticleIndex = bodyIndex + offset;

                    min_x = lowestDomainList->borders[lowestDomainList->relevantDomainListOriginalIndex[relevantIndex]*2*DIM];
                    max_x = lowestDomainList->borders[lowestDomainList->relevantDomainListOriginalIndex[relevantIndex]*2*DIM+1];
#if DIM > 1
                    min_y = lowestDomainList->borders[lowestDomainList->relevantDomainListOriginalIndex[relevantIndex]*2*DIM+2];
                    max_y = lowestDomainList->borders[lowestDomainList->relevantDomainListOriginalIndex[relevantIndex]*2*DIM+3];
#if DIM == 3
                    min_z = lowestDomainList->borders[lowestDomainList->relevantDomainListOriginalIndex[relevantIndex]*2*DIM+4];
                    max_z = lowestDomainList->borders[lowestDomainList->relevantDomainListOriginalIndex[relevantIndex]*2*DIM+5];
#endif
#endif

                    //printf("x -> (%e, %e), y -> (%e, %e), z -> (%e, %e)\n", min_x, max_x, min_y, max_y, min_z, max_z);

                    // x-direction
                    if (particles->x[currentParticleIndex] < min_x) {
                        // outside
                        dx = particles->x[currentParticleIndex] - min_x;
                    } else if (particles->x[currentParticleIndex] > max_x) {
                        // outside
                        dx = particles->x[currentParticleIndex] - max_x;
                    } else {
                        // in between: do nothing
                        dx = 0;
                    }
#if DIM > 1
                    // y-direction
                    if (particles->y[currentParticleIndex] < min_y) {
                        // outside
                        dy = particles->y[currentParticleIndex] - min_y;
                    } else if (particles->y[currentParticleIndex] > max_y) {
                        // outside
                        dy = particles->y[currentParticleIndex] - max_y;
                    } else {
                        // in between: do nothing
                        dy = 0;
                    }
#if DIM == 3
                    // z-direction
                    if (particles->z[currentParticleIndex] < min_z) {
                        // outside
                        dz = particles->z[currentParticleIndex] - min_z;
                    } else if (particles->z[currentParticleIndex] > max_z) {
                        // outside
                        dz = particles->z[currentParticleIndex] - max_z;
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
                    d = dx*dx + dy*dy + dz*dz;
#endif

                    //if ((bodyIndex + offset) % 500 == 0) {
                    //    printf("d = %e < (%e * %e = %e) sml = %e\n", d, searchRadius, searchRadius,
                    //           searchRadius * searchRadius, particles->sml[bodyIndex + offset]);
                    //}
                    //if (d < (particles->sml[bodyIndex + offset] * particles->sml[bodyIndex + offset])) {
                    //if (currentParticleIndex % 100 == 0) {
                    //    printf("d = %e < %e\n", d, searchRadius * searchRadius);
                    //}
                    if (d < (searchRadius * searchRadius)) {
                        //printf("i = %i: d = %e < %e\n", currentParticleIndex, d, searchRadius * searchRadius);
                        //printf("d = %f < (%f * %f = %f)\n", d, particles->sml[bodyIndex + offset], particles->sml[bodyIndex + offset],
                        //       particles->sml[bodyIndex + offset] * particles->sml[bodyIndex + offset]);
                        added = true;
                        sendIndices[currentParticleIndex] = 1;
                    }
                    //else {
                    //    sendIndices[bodyIndex + offset] = -1;
                    //}

                }

                //__threadfence();
                offset += stride;
            }


        }

        __global__ void symbolicForce_test2(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                            DomainList *domainList, integer *sendIndices, real searchRadius,
                                            integer n, integer m, integer relevantProc, integer relevantIndicesCounter,
                                            Curve::Type curveType) {

            int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;
            int offset = 0; // start with numParticles

            int currentParticleIndex, particleLevel, domainListLevel, currentDomainListIndex, childIndex, currentProc;

            real min_x, max_x;
            real dx;
#if DIM > 1
            real min_y, max_y;
            real dy;
#if DIM == 3
            real min_z, max_z;
            real dz;
#endif
#endif
            real d;

            //bool added;
            int added = 0;

            //printf("Hallo !!!!! %i \n", n);

            while ((bodyIndex + offset) < n) {

                //added = false;

                for (int relevantIndex = 0; relevantIndex < relevantIndicesCounter; relevantIndex++) {


                    currentProc = domainList->relevantDomainListProcess[relevantIndex];

                    //if ((added >> currentProc) & 1) {
                    //    continue;
                    //}


                    currentParticleIndex = bodyIndex + offset;
                    //domainListLevel = domainList->relevantDomainListLevels[relevantIndex];
                    //particleLevel = particles->level[currentParticleIndex];
                    currentDomainListIndex = domainList->relevantDomainListIndices[relevantIndex];



                    min_x = domainList->borders[domainList->relevantDomainListOriginalIndex[relevantIndex] * 2 *
                                                DIM];
                    max_x = domainList->borders[
                            domainList->relevantDomainListOriginalIndex[relevantIndex] * 2 * DIM +
                            1];
#if DIM > 1
                    min_y = domainList->borders[
                            domainList->relevantDomainListOriginalIndex[relevantIndex] * 2 * DIM +
                            2];
                    max_y = domainList->borders[
                            domainList->relevantDomainListOriginalIndex[relevantIndex] * 2 * DIM +
                            3];
#if DIM == 3
                    min_z = domainList->borders[
                            domainList->relevantDomainListOriginalIndex[relevantIndex] * 2 * DIM +
                            4];
                    max_z = domainList->borders[
                            domainList->relevantDomainListOriginalIndex[relevantIndex] * 2 * DIM +
                            5];
#endif
#endif

                    // determine (smallest) distance between domain list box and (pseudo-) particle
                    if (particles->x[currentParticleIndex] < min_x) {
                        dx = particles->x[currentParticleIndex] - min_x;
                    } else if (particles->x[currentParticleIndex] > max_x) {
                        dx = particles->x[currentParticleIndex] - max_x;
                    } else {
                        dx = 0.;
                    }
#if DIM > 1
                    if (particles->y[currentParticleIndex] < min_y) {
                        dy = particles->y[currentParticleIndex] - min_y;
                    } else if (particles->y[currentParticleIndex] > max_y) {
                        dy = particles->y[currentParticleIndex] - max_y;
                    } else {
                        dy = 0.;
                    }
#if DIM == 3
                    if (particles->z[currentParticleIndex] < min_z) {
                        dz = particles->z[currentParticleIndex] - min_z;
                    } else if (particles->z[currentParticleIndex] > max_z) {
                        dz = particles->z[currentParticleIndex] - max_z;
                    } else {
                        dz = 0.;
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

                    //printf("d = %e < %e\n", d, searchRadius * searchRadius);
                    if (d < (searchRadius * searchRadius)) {

                        //printf("adding ...\n");

                        added = added | (1 << currentProc);
                        sendIndices[currentParticleIndex] = sendIndices[currentParticleIndex] | (1 << currentProc);

                    }
                }

                added = 0;
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
                    //printf("check: sending: (%e, %e, %e) %e\n", particles->x[bodyIndex + offset], particles->y[bodyIndex + offset],
                    //       particles->z[bodyIndex + offset], particles->mass[bodyIndex + offset]);
                }

                __threadfence();
                offset += stride;
            }

        }

        __global__ void collectSendIndices_test2(Tree *tree, Particles *particles, integer *sendIndices,
                                                 integer *particles2Send, integer *particlesCount,
                                                 integer numParticlesLocal, integer numParticles,
                                                 integer treeIndex, int currentProc, Curve::Type curveType) {

            integer bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
            integer stride = blockDim.x*gridDim.x;
            integer offset = 0;

            integer particleInsertIndex;

            // it is a particle
            while ((bodyIndex + offset) < numParticlesLocal) {
                if ((sendIndices[bodyIndex + offset] >> currentProc) & 1) { // TODO: >= 1 or == 1 (was == 1)
                    if (bodyIndex + offset < numParticlesLocal) {
                        particleInsertIndex = atomicAdd(particlesCount, 1);
                        particles2Send[particleInsertIndex] = bodyIndex + offset;
                    }
                }
                __threadfence();
                offset += stride;
            }
        }

        // deprecated
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


#define COMPUTE_DIRECTLY 0

        __global__ void insertReceivedParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                DomainList *domainList, DomainList *lowestDomainList, int n, int m) {


            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;

            //note: -1 used as "null pointer"
            //note: -2 used to lock a child (pointer)

            volatile integer *childList = tree->child;

            integer offset;
            int level;
            bool newBody = true;

            real min_x;
            real max_x;
            real x;
#if DIM > 1
            real y;
            real min_y;
            real max_y;
#if DIM == 3
            real z;
            real min_z;
            real max_z;
#endif
#endif

            integer childPath;
            integer temp;

            offset = tree->toDeleteLeaf[0];

            while ((bodyIndex + offset) < tree->toDeleteLeaf[1]) {

                if (newBody) {

                    newBody = false;
                    level = 0;

                    //printf("check: inserting: (%e, %e, %e) %e\n", particles->x[bodyIndex + offset], particles->y[bodyIndex + offset],
                    //       particles->z[bodyIndex + offset], particles->mass[bodyIndex + offset]);

                    // copy bounding box(es)
                    min_x = *tree->minX;
                    max_x = *tree->maxX;
                    x = particles->x[bodyIndex + offset];
#if DIM > 1
                    y = particles->y[bodyIndex + offset];
                    min_y = *tree->minY;
                    max_y = *tree->maxY;
#if DIM == 3
                    z = particles->z[bodyIndex + offset];
                    min_z = *tree->minZ;
                    max_z = *tree->maxZ;
#endif
#endif
                    temp = 0;
                    childPath = 0;

                    // find insertion point for body
                    //if (particles->x[bodyIndex + offset] < 0.5 * (min_x + max_x)) { // x direction
                    if (x < 0.5 * (min_x + max_x)) { // x direction
                        childPath += 1;
                        max_x = 0.5 * (min_x + max_x);
                    }
                    else {
                        min_x = 0.5 * (min_x + max_x);
                    }
#if DIM > 1
                    //if (particles->y[bodyIndex + offset] < 0.5 * (min_y + max_y)) { // y direction
                    if (y < 0.5 * (min_y + max_y)) { // y direction
                        childPath += 2;
                        max_y = 0.5 * (min_y + max_y);
                    }
                    else {
                        min_y = 0.5 * (min_y + max_y);
                    }
#if DIM == 3
                    //if (particles->z[bodyIndex + offset] < 0.5 * (min_z + max_z)) {  // z direction
                    if (z < 0.5 * (min_z + max_z)) {  // z direction
                        childPath += 4;
                        max_z = 0.5 * (min_z + max_z);
                    }
                    else {
                        min_z = 0.5 * (min_z + max_z);
                    }
#endif
#endif
                }

                register integer childIndex = childList[temp*POW_DIM + childPath];

                // traverse tree until hitting leaf node
                while (childIndex >= m) { //n

                    temp = childIndex;
                    level++;

                    //if (particles->level[temp] == -1) {
                    //    printf("level[%i] = %i...\n", temp, particles->level[temp]);
                    //    assert(0);
                    //}

                    childPath = 0;

                    // find insertion point for body
                    if (x < 0.5 * (min_x + max_x)) { // x direction
                        childPath += 1;
                        max_x = 0.5 * (min_x + max_x);
                    }
                    else {
                        min_x = 0.5 * (min_x + max_x);
                    }
#if DIM > 1
                    if (y < 0.5 * (min_y + max_y)) { // y direction
                        childPath += 2;
                        max_y = 0.5 * (min_y + max_y);
                    }
                    else {
                        min_y = 0.5 * (min_y + max_y);
                    }
#if DIM == 3
                    if (z < 0.5 * (min_z + max_z)) { // z direction
                        childPath += 4;
                        max_z = 0.5 * (min_z + max_z);
                    }
                    else {
                        min_z = 0.5 * (min_z + max_z);
                    }
#endif
#endif

#if COMPUTE_DIRECTLY
                    if (particles->mass[bodyIndex + offset] != 0) {
                //particles->x[temp] += particles->weightedEntry(bodyIndex + offset, Entry::x);
                atomicAdd(&particles->x[temp], particles->weightedEntry(bodyIndex + offset, Entry::x));
#if DIM > 1
                //particles->y[temp] += particles->weightedEntry(bodyIndex + offset, Entry::y);
                atomicAdd(&particles->y[temp], particles->weightedEntry(bodyIndex + offset, Entry::y));
#if DIM == 3
                //particles->z[temp] += particles->weightedEntry(bodyIndex + offset, Entry::z);
                atomicAdd(&particles->z[temp], particles->weightedEntry(bodyIndex + offset, Entry::z));
#endif
#endif
            }

            //particles->mass[temp] += particles->mass[bodyIndex + offset];
            atomicAdd(&particles->mass[temp], particles->mass[bodyIndex + offset]);
#endif // COMPUTE_DIRECTLY

                    atomicAdd(&tree->count[temp], 1);
                    childIndex = childList[POW_DIM * temp + childPath];
                }

                __syncthreads();

                // if child is not locked
                if (childIndex != -2) {

                    integer locked = temp * POW_DIM + childPath;

                    if (atomicCAS((int *)&childList[locked], childIndex, -2) == childIndex) {

                        // check whether a body is already stored at the location
                        if (childIndex == -1) {
                            //insert body and release lock
                            childList[locked] = bodyIndex + offset;
                            particles->level[bodyIndex + offset] = level + 1;

                        }
                        else {
                            if (childIndex >= n) {
                                printf("ATTENTION!\n");
                            }
                            integer patch = POW_DIM * m; //8*n
                            while (childIndex >= 0 && childIndex < n) { // was n

                                //create a new cell (by atomically requesting the next unused array index)
                                integer cell = atomicAdd(tree->index, 1);
                                patch = min(patch, cell);

                                if (patch != cell) {
                                    childList[POW_DIM * temp + childPath] = cell;
                                }

                                level++;
                                // ATTENTION: most likely a problem with level counting (level = level - 1)
                                // but could be also a problem regarding maximum tree depth...
                                if (level > (MAX_LEVEL + 1)) {
#if DIM == 1
                                    cudaAssert("buildTree: level = %i for index %i (%e)", level,
                                       bodyIndex + offset, particles->x[bodyIndex + offset]);
#elif DIM == 2
                                    cudaAssert("buildTree: level = %i for index %i (%e, %e)", level,
                                       bodyIndex + offset, particles->x[bodyIndex + offset],
                                       particles->y[bodyIndex + offset]);
#else
                                    cudaAssert("buildTree: level = %i for index %i (%e, %e, %e)", level,
                                               bodyIndex + offset, particles->x[bodyIndex + offset],
                                               particles->y[bodyIndex + offset],
                                               particles->z[bodyIndex + offset]);
#endif
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
#if COMPUTE_DIRECTLY
                                particles->x[cell] += particles->weightedEntry(childIndex, Entry::x);
                        //particles->x[cell] += particles->weightedEntry(childIndex, Entry::x);
#if DIM > 1
                        particles->y[cell] += particles->weightedEntry(childIndex, Entry::y);
                        //particles->y[cell] += particles->weightedEntry(childIndex, Entry::y);
#if DIM == 3
                        particles->z[cell] += particles->weightedEntry(childIndex, Entry::z);
                        //particles->z[cell] += particles->weightedEntry(childIndex, Entry::z);
#endif
#endif

                        //if (cell % 1000 == 0) {
                        //    printf("buildTree: x[%i] = (%f, %f, %f) from x[%i] = (%f, %f, %f) m = %f\n", cell, particles->x[cell], particles->y[cell],
                        //           particles->z[cell], childIndex, particles->x[childIndex], particles->y[childIndex],
                        //           particles->z[childIndex], particles->mass[childIndex]);
                        //}

                        particles->mass[cell] += particles->mass[childIndex];
#else // COMPUTE_DIRECTLY
                                //particles->x[cell] = particles->x[childIndex];
                                particles->x[cell] = 0.5 * (min_x + max_x);
#if DIM > 1
                                //particles->y[cell] = particles->y[childIndex];
                                particles->y[cell] = 0.5 * (min_y + max_y);
#if DIM == 3
                                //particles->z[cell] = particles->z[childIndex];
                                particles->z[cell] = 0.5 * (min_z + max_z);
#endif
#endif

#endif // COMPUTE_DIRECTLY

                                tree->count[cell] += tree->count[childIndex];

                                childList[POW_DIM * cell + childPath] = childIndex;
                                particles->level[cell] = level;
                                particles->level[childIndex] += 1;
                                tree->start[cell] = -1;

#if DEBUGGING
                                if (particles->level[cell] >= particles->level[childIndex]) {
                            printf("lvl: %i vs. %i\n", particles->level[cell], particles->level[childIndex]);
                            assert(0);
                        }
#endif

                                // insert new particle
                                temp = cell;
                                childPath = 0;

                                // find insertion point for body
                                //if (particles->x[bodyIndex + offset] < 0.5 * (min_x + max_x)) {
                                if (x < 0.5 * (min_x + max_x)) {
                                    childPath += 1;
                                    max_x = 0.5 * (min_x + max_x);
                                } else {
                                    min_x = 0.5 * (min_x + max_x);
                                }
#if DIM > 1
                                //if (particles->y[bodyIndex + offset] < 0.5 * (min_y + max_y)) {
                                if (y < 0.5 * (min_y + max_y)) {
                                    childPath += 2;
                                    max_y = 0.5 * (min_y + max_y);
                                } else {
                                    min_y = 0.5 * (min_y + max_y);
                                }
#if DIM == 3
                                //if (particles->z[bodyIndex + offset] < 0.5 * (min_z + max_z)) {
                                if (z < 0.5 * (min_z + max_z)) {
                                    childPath += 4;
                                    max_z = 0.5 * (min_z + max_z);
                                } else {
                                    min_z = 0.5 * (min_z + max_z);
                                }
#endif
#endif
#if COMPUTE_DIRECTLY
                                // COM / preparing for calculation of COM
                        if (particles->mass[bodyIndex + offset] != 0) {
                            //particles->x[cell] += particles->weightedEntry(bodyIndex + offset, Entry::x);
                            particles->x[cell] += particles->weightedEntry(bodyIndex + offset, Entry::x);
#if DIM > 1
                            //particles->y[cell] += particles->weightedEntry(bodyIndex + offset, Entry::y);
                            particles->y[cell] += particles->weightedEntry(bodyIndex + offset, Entry::y);
#if DIM == 3
                            //particles->z[cell] += particles->weightedEntry(bodyIndex + offset, Entry::z);
                            particles->z[cell] += particles->weightedEntry(bodyIndex + offset, Entry::z);
#endif
#endif
                            particles->mass[cell] += particles->mass[bodyIndex + offset];
                        }
#endif // COMPUTE_DIRECTLY
                                tree->count[cell] += tree->count[bodyIndex + offset];
                                childIndex = childList[POW_DIM * temp + childPath];
                            }

                            childList[POW_DIM * temp + childPath] = bodyIndex + offset;
                            particles->level[bodyIndex + offset] = level + 1;

                            __threadfence();  // written to global memory arrays (child, x, y, mass) thus need to fence
                            childList[locked] = patch;
                        }
                        offset += stride;
                        newBody = true;
                    }
                }
                __syncthreads(); //TODO: __syncthreads() needed?
            }
        }

        /*__global__ void insertReceivedParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                DomainList *domainList, DomainList *lowestDomainList, int n, int m) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;

            //note: -1 used as "null pointer"
            //note: -2 used to lock a child (pointer)

            integer offset;
            int level;
            //printf("newBody = %d\n", newBody);

            volatile real min_x;
            volatile real max_x;
            volatile real x;
#if DIM > 1
            volatile real y;
            volatile real min_y;
            volatile real max_y;
#if DIM == 3
            volatile real z;
            volatile real min_z;
            volatile real max_z;
#endif
#endif


            integer childPath;
            integer temp;

            offset = tree->toDeleteLeaf[0];

            //bool newBody = true;
            //volatile int
            bool newBody;

            while ((bodyIndex + offset) < tree->toDeleteLeaf[1]) {

#if DEBUGGING
                printf("[rank %i] sph(%i) START newBody = %d offset = %i [rank %i] toDelete = %i - %i\n", subDomainKeyTree->rank, bodyIndex + offset, newBody, offset,
                       subDomainKeyTree->rank, tree->toDeleteLeaf[0], tree->toDeleteLeaf[1]);
#endif

                if (newBody) {

                    //printf("[rank %i] sph(%i) START newBody::newBody = %d\n", subDomainKeyTree->rank, bodyIndex + offset, newBody);
                    newBody = false;
                    level = 0;

                    // copy bounding box(es)
                    min_x = *tree->minX;
                    max_x = *tree->maxX;
                    x = particles->x[bodyIndex + offset];
#if DIM > 1
                    y = particles->y[bodyIndex + offset];
                    min_y = *tree->minY;
                    max_y = *tree->maxY;
#if DIM == 3
                    z = particles->z[bodyIndex + offset];
                    min_z = *tree->minZ;
                    max_z = *tree->maxZ;
#endif
#endif
                    temp = 0;
                    childPath = 0;

                    // find insertion point for body
                    //if (particles->x[bodyIndex + offset] < 0.5 * (min_x + max_x)) { // x direction
                    if (x < 0.5 * (min_x + max_x)) { // x direction
                        childPath += 1;
                        max_x = 0.5 * (min_x + max_x);
                    }
                    else {
                        min_x = 0.5 * (min_x + max_x);
                    }
#if DIM > 1
                    //if (particles->y[bodyIndex + offset] < 0.5 * (min_y + max_y)) { // y direction
                    if (y < 0.5 * (min_y + max_y)) { // y direction
                        childPath += 2;
                        max_y = 0.5 * (min_y + max_y);
                    }
                    else {
                        min_y = 0.5 * (min_y + max_y);
                    }
#if DIM == 3
                    //if (particles->z[bodyIndex + offset] < 0.5 * (min_z + max_z)) {  // z direction
                    if (z < 0.5 * (min_z + max_z)) {  // z direction
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

#if DIM == 3
                if (particles->x[bodyIndex + offset] > max_x || particles->x[bodyIndex + offset] < min_x ||
                        particles->y[bodyIndex + offset] > max_y || particles->y[bodyIndex + offset] < min_y ||
                        particles->z[bodyIndex + offset] > max_z || particles->z[bodyIndex + offset] < min_z) {
                    printf("[rank %i] sph(%i) out of box (%e, %e, %e) min/max = (%e - %e, %e - %e, %e - %e) level = %i\n",
                           subDomainKeyTree->rank, bodyIndex + offset,
                           particles->x[bodyIndex + offset], particles->y[bodyIndex + offset], particles->z[bodyIndex + offset],
                           min_x, max_x, min_y, max_y, min_z, max_z, level);
                    //assert(0);
                    cudaAssert("[rank %i] sph index: %i out of box!\n", subDomainKeyTree->rank, bodyIndex + offset);
                }

//#if DEBUGGING
                //if (childIndex != -2 && childIndex != -1 &&
                //    (particles->x[childIndex] > max_x || particles->x[childIndex] < min_x ||
                //    particles->y[childIndex] > max_y || particles->y[childIndex] < min_y ||
                //    particles->z[childIndex] > max_z || particles->z[childIndex] < min_z)) {
                //    printf("[rank %i] sph(%i) out of box for childIndex = %i (%e, %e, %e) min/max = (%e - %e, %e - %e, %e - %e) level = %i\n", subDomainKeyTree->rank, bodyIndex + offset,
                //           childIndex, particles->x[childIndex], particles->y[childIndex], particles->z[childIndex],
                //           min_x, max_x, min_y, max_y, min_z, max_z, level);
                //    //assert(0);
                //}
//#endif
#endif

#if DEBUGGING
                printf("[rank %i] sph(%i) childIndex = %i temp = %i childPath = %i (%e, %e, %e) vs (%e, %e, %e) min/max = (%e - %e, %e - %e, %e - %e\n",
                       subDomainKeyTree->rank, bodyIndex + offset, childIndex, temp, childPath,
                       particles->x[bodyIndex + offset], particles->y[bodyIndex + offset], particles->z[bodyIndex + offset],
                       particles->x[childIndex], particles->y[childIndex], particles->z[childIndex],
                       min_x, max_x, min_y, max_y, min_z, max_z);
#endif

                // traverse tree until hitting leaf node
                while (childIndex >= m) { //n

                    temp = childIndex;
                    level++;

                    childPath = 0;

                    // find insertion point for body
                    //if (particles->x[bodyIndex + offset] < 0.5 * (min_x + max_x)) { // x direction
                    if (x < 0.5 * (min_x + max_x)) { // x direction
                        childPath += 1;
                        max_x = 0.5 * (min_x + max_x);
                    }
                    else {
                        min_x = 0.5 * (min_x + max_x);
                    }
#if DIM > 1
                    //if (particles->y[bodyIndex + offset] < 0.5 * (min_y + max_y)) { // y direction
                    if (y < 0.5 * (min_y + max_y)) { // y direction
                        childPath += 2;
                        max_y = 0.5 * (min_y + max_y);
                    }
                    else {
                        min_y = 0.5 * (min_y + max_y);
                    }
#if DIM == 3
                    //if (particles->z[bodyIndex + offset] < 0.5 * (min_z + max_z)) { // z direction
                    if (z < 0.5 * (min_z + max_z)) { // z direction
                        childPath += 4;
                        max_z = 0.5 * (min_z + max_z);
                    }
                    else {
                        min_z = 0.5 * (min_z + max_z);
                    }
#endif
#endif

                    //atomicAdd(&tree->count[temp], 1);

                    childIndex = tree->child[POW_DIM * temp + childPath];
#if DEBUGGING
                    printf("[rank %i] sph(%i) childIndex = %i level = %i (temp = %i)\n", subDomainKeyTree->rank, bodyIndex + offset, childIndex, level, temp);
#endif
                }

                // if child is not locked
                if (childIndex != -2) {

                    integer locked = temp * POW_DIM + childPath;

                    if (atomicCAS(&tree->child[locked], childIndex, -2) == childIndex) {

                        // check whether a body is already stored at the location
                        if (childIndex == -1) {
#if DEBUGGING
                            printf("[rank %i] sph(%i) inserting for temp = %i and childPath %i... childIndex = %i level = %i\n", subDomainKeyTree->rank, bodyIndex + offset, temp, childPath, childIndex, level);
#endif
                            //insert body and release lock
                            tree->child[locked] = bodyIndex + offset;
                            particles->level[bodyIndex + offset] = level + 1;

                        }
                        else {
                            //if (childIndex >= n) {
                            //    printf("ATTENTION!\n");
                            //}
                            integer patch = POW_DIM * m; //8*n
                            while (childIndex >= 0 && childIndex < n) { // was n

                                //create a new cell (by atomically requesting the next unused array index)
                                integer cell = atomicAdd(tree->index, 1);
                                //printf("cell = %i\n", cell);
                                patch = min(patch, cell);

                                if (patch != cell) {
                                    tree->child[POW_DIM * temp + childPath] = cell;
                                }

                                level++;

                                if (level > (MAX_LEVEL + 1)) {
                                    printf("[rank %i] sph(%i) assert... childIndex = %i cell = %i (type: %i) temp = %i (type: %i) level = %i (%i, %i)\n",
                                           subDomainKeyTree->rank, bodyIndex + offset, childIndex, cell,
                                           particles->nodeType[cell], temp, particles->nodeType[temp], level,
                                           tree->toDeleteLeaf[0], tree->toDeleteLeaf[1]);

                                    printf("level = %i for index %i (%e, %e, %e)\n", level,
                                           bodyIndex + offset, particles->x[bodyIndex + offset],
                                           particles->y[bodyIndex + offset], particles->z[bodyIndex + offset]);
                                    cudaAssert("level = %i > MAX_LEVEL %i\n", level, MAX_LEVEL);
                                }

#if DEBUGGING
                                // debugging
                                //if (particles->x[bodyIndex + offset] == particles->x[childIndex] &&
                                //    particles->y[bodyIndex + offset] == particles->y[childIndex] &&
                                //    particles->z[bodyIndex + offset] == particles->z[childIndex]) {
                                //    printf("duplicate!!!: %i vs %i (%e, %e, %e) vs (%e, %e, %e)\n",
                                //           bodyIndex + offset, childIndex, particles->x[bodyIndex + offset],
                                //           particles->y[bodyIndex + offset], particles->z[bodyIndex + offset],
                                //           particles->x[childIndex], particles->y[childIndex],
                                //           particles->z[childIndex]);
                                //}
#endif

                                // insert old/original particle
                                childPath = 0;
                                if (particles->x[childIndex] < 0.5 * (min_x + max_x)) { childPath += 1; }
#if DIM > 1
                                if (particles->y[childIndex] < 0.5 * (min_y + max_y)) { childPath += 2; }
#if DIM == 3
                                if (particles->z[childIndex] < 0.5 * (min_z + max_z)) { childPath += 4; }
#endif
#endif

                                //particles->x[cell] = particles->x[childIndex];
                                particles->x[cell] = 0.5 * (min_x + max_x);
#if DIM > 1
                                //particles->y[cell] = particles->y[childIndex];
                                particles->y[cell] = 0.5 * (min_y + max_y);
#if DIM == 3
                                //particles->z[cell] = particles->z[childIndex];
                                particles->z[cell] = 0.5 * (min_z + max_z);
#endif
#endif

                                tree->child[POW_DIM * cell + childPath] = childIndex;
                                //particles->level[temp] = level;

#if DEBUGGING
                                printf("[rank %i] sph(%i) inbetween POW_DIM * %i + %i = %i (%e, %e, %e) vs (%e, %e, %e) min/max = (%e - %e, %e - %e, %e - %e)\n", subDomainKeyTree->rank,
                                    bodyIndex + offset, cell, childPath, childIndex,
                                    particles->x[bodyIndex + offset], particles->y[bodyIndex + offset], particles->z[bodyIndex + offset],
                                    particles->x[childIndex], particles->y[childIndex], particles->z[childIndex],
                                    min_x, max_x, min_y, max_y, min_z, max_z);
#endif
                                particles->level[cell] = level;
                                particles->level[childIndex] += 1;
                                tree->start[cell] = -1;

                                // insert new particle
                                temp = cell;
                                childPath = 0;

                                // find insertion point for body
                                //if (particles->x[bodyIndex + offset] < 0.5 * (min_x + max_x)) {
                                if (x < 0.5 * (min_x + max_x)) {
                                    childPath += 1;
                                    max_x = 0.5 * (min_x + max_x);
                                } else {
                                    min_x = 0.5 * (min_x + max_x);
                                }
#if DIM > 1
                                //if (particles->y[bodyIndex + offset] < 0.5 * (min_y + max_y)) {
                                if (y < 0.5 * (min_y + max_y)) {
                                    childPath += 2;
                                    max_y = 0.5 * (min_y + max_y);
                                } else {
                                    min_y = 0.5 * (min_y + max_y);
                                }
#if DIM == 3
                                //if (particles->z[bodyIndex + offset] < 0.5 * (min_z + max_z)) {
                                if (z < 0.5 * (min_z + max_z)) {
                                    childPath += 4;
                                    max_z = 0.5 * (min_z + max_z);
                                } else {
                                    min_z = 0.5 * (min_z + max_z);
                                }
#endif
#endif

                                tree->count[cell] += tree->count[bodyIndex + offset];
                                childIndex = tree->child[POW_DIM * temp + childPath];
#if DEBUGGING
                                printf("[rank %i] sph(%i) within childIndex = %i level = %i temp = %i childPath = %i -> %i\n",
                                       subDomainKeyTree->rank, bodyIndex + offset, childIndex, level, temp, childPath, tree->child[POW_DIM * temp + childPath]);
#endif
                            }

                            tree->child[POW_DIM * temp + childPath] = bodyIndex + offset;
                            particles->level[bodyIndex + offset] = level + 1;
#if DEBUGGING
                            printf("[rank %i] sph(%i) set temp = %i + childPath = %i = index = %i (level = %i) x[%i] = (%e, %e, %e)\n",
                                    subDomainKeyTree->rank, bodyIndex + offset, temp, childPath, bodyIndex + offset, level, temp,
                                    particles->x[temp], particles->y[temp], particles->z[temp]);
#endif

                            __threadfence();  // written to global memory arrays (child, x, y, mass) thus need to fence
                            tree->child[locked] = patch;
                        }
#if DEBUGGING
                        printf("[rank %i] sph(%i) newBody = %d, offset+=stride (= %i) = %i\n", subDomainKeyTree->rank, bodyIndex + offset, newBody, stride, offset + stride);
#endif
                        offset += stride;
                        newBody = true;
#if DEBUGGING
                        printf("[rank %i] sph(%i) newBody = %d, after offset+=stride (= %i) = %i\n", subDomainKeyTree->rank, bodyIndex + offset, newBody, stride, offset);
#endif
                    }
                    //else {
                    //    //printf("WTF!\n");
                    //    //newBody = 1; //TODO: needed?
                    //}
                //}
                //else {
//#if DEBUGGING
//                    printf("[rank %i] sph(%i) waiting... childIndex = %i level = %i newBody = %d\n", subDomainKeyTree->rank, bodyIndex + offset, childIndex, level,
//                           newBody);
//#endif
                }
                __syncthreads(); //TODO: needed?
            }
        }
        */

        __global__ void calculateCentersOfMass(Tree *tree, Particles *particles, integer level) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;

            integer offset = tree->toDeleteNode[0];

            //int counter[21];
            //for (int i=0; i<21;i++) {
            //    counter[i] = 0;
            //}

            integer index;

            while ((bodyIndex + offset) < tree->toDeleteNode[1]) {

                if (particles->level[bodyIndex + offset] == level) {

                    if (particles->level[bodyIndex + offset] == -1 || particles->level[bodyIndex + offset] > 21) {
                        printf("level[%i] = %i!!!\n", bodyIndex + offset, particles->level[bodyIndex + offset]);
                    }

                    particles->mass[bodyIndex + offset] = 0.;
                    particles->x[bodyIndex + offset] = 0.;
#if DIM > 1
                    particles->y[bodyIndex + offset] = 0.;
#if DIM == 3
                    particles->z[bodyIndex + offset] = 0.;
#endif
#endif

                    for (int child = 0; child < POW_DIM; ++child) {
                        index = POW_DIM * (bodyIndex + offset) + child;
                        if (tree->child[index] != -1) {
                            particles->x[bodyIndex + offset] += particles->weightedEntry(tree->child[index], Entry::x);
#if DIM > 1
                            particles->y[bodyIndex + offset] += particles->weightedEntry(tree->child[index], Entry::y);
#if DIM == 3
                            particles->z[bodyIndex + offset] += particles->weightedEntry(tree->child[index], Entry::z);
#endif
#endif
                            particles->mass[bodyIndex + offset] += particles->mass[tree->child[index]];
                        }
                    }

                    if (particles->mass[bodyIndex + offset] > 0.) {
                        particles->x[bodyIndex + offset] /= particles->mass[bodyIndex + offset];
#if DIM > 1
                        particles->y[bodyIndex + offset] /= particles->mass[bodyIndex + offset];
#if DIM == 3
                        particles->z[bodyIndex + offset] /= particles->mass[bodyIndex + offset];
#endif
#endif
                    }

                    //counter[particles->level[bodyIndex + offset]] += 1;

                }
                offset += stride;
            }

            //for (int i=0; i<21;i++) {
            //    printf("counter[%i] = %i\n", i, counter[i]);
            //}

        }

        // TODO: use memory bounding boxes ...
        __global__ void determineSearchRadii(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                             DomainList *domainList, DomainList *lowestDomainList, real *searchRadii,
                                             int n, int m, Curve::Type curveType) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            int lowestDomainIndex;
            real searchRadius;
            int path;
            real distance;
            keyType key;
            int proc;

            real min_x, max_x, dx;
#if DIM > 1
            real min_y, max_y, dy;
#if DIM == 3
            real min_z, max_z, dz;
#endif
#endif

            while ((bodyIndex + offset) < n) {

                searchRadius = 0.;

                //for (int i=0; i<*lowestDomainList->domainListIndex; i++) {
                for (int i=0; i<*lowestDomainList->domainListCounter; i++) {
                    //lowestDomainIndex = lowestDomainList->domainListIndices[i];
                    lowestDomainIndex = lowestDomainList->relevantDomainListIndices[i];
                    //key = tree->getParticleKey(particles, lowestDomainIndex, MAX_LEVEL, curveType);
                    //proc = subDomainKeyTree->key2proc(key);
                    proc = lowestDomainList->relevantDomainListProcess[i];
                    if (proc != subDomainKeyTree->rank) {
                        // determine distance
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
                        for (int level=0; level<lowestDomainList->domainListLevels[i]; level++) {
                            path = (integer)(lowestDomainList->domainListKeys[i] >> (MAX_LEVEL * DIM - DIM * (level + 1))& (integer) (POW_DIM - 1));

                            // Possibility 1
                            //if (path % 2 != 0) {
                            if (path & 1) {
                                max_x = 0.5 * (min_x + max_x);
                                //path -= 1;
                            } else {
                                min_x = 0.5 * (min_x + max_x);
                            }
#if DIM > 1
                            if ((path >> 1) & 1) {
                            //if (path % 2 == 0 && path % 4 != 0) {
                                max_y = 0.5 * (min_y + max_y);
                                //path -= 2;
                            } else {
                                min_y = 0.5 * (min_y + max_y);
                            }
#if DIM == 3
                            if ((path >> 2) & 1) {
                            //if (path == 4) {
                                max_z = 0.5 * (min_z + max_z);
                                //path -= 4;
                            } else {
                                min_z = 0.5 * (min_z + max_z);
                            }
#endif
#endif

                            // Possibility 2
                            /* // find insertion point for body
                            if (particles->x[lowestDomainIndex] < 0.5 * (min_x + max_x)) { // x direction
                                //childPath += 1;
                                max_x = 0.5 * (min_x + max_x);
                            }
                            else {
                                min_x = 0.5 * (min_x + max_x);
                            }
#if DIM > 1
                            if (particles->y[lowestDomainIndex] < 0.5 * (min_y + max_y)) { // y direction
                                //childPath += 2;
                                max_y = 0.5 * (min_y + max_y);
                            }
                            else {
                                min_y = 0.5 * (min_y + max_y);
                            }
#if DIM == 3
                            if (particles->z[lowestDomainIndex] < 0.5 * (min_z + max_z)) {  // z direction
                                //childPath += 4;
                                max_z = 0.5 * (min_z + max_z);
                            }
                            else {
                                min_z = 0.5 * (min_z + max_z);
                            }
#endif
#endif*/
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
                        distance = cuda::math::sqrt(dx*dx);
#elif DIM == 2
                        distance = cuda::math::sqrt(dx*dx + dy*dy);
#else
                        distance = cuda::math::sqrt(dx*dx + dy*dy + dz*dz);
#endif

                        if (distance < particles->sml[bodyIndex + offset]) {
                            searchRadius = particles->sml[bodyIndex + offset] - distance;
                            //printf("search: distance %e level = %i\n", distance, lowestDomainList->domainListLevels[i]);
                        }
                    }
                }

                searchRadii[bodyIndex + offset] = searchRadius;
                //printf("search: searchRadius: %e\n", searchRadius);

                offset += stride;
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

            real fixedRadiusNN_bruteForce(Tree *tree, Particles *particles, integer *interactions, integer numParticlesLocal,
                                     integer numParticles, integer numNodes) {
                ExecutionPolicy executionPolicy; // 4 * numMultiProcessors, 256
                return cuda::launch(true, executionPolicy, ::SPH::Kernel::fixedRadiusNN_bruteForce, tree, particles, interactions,
                                    numParticlesLocal, numParticles, numNodes);
            }

            real fixedRadiusNN(Tree *tree, Particles *particles, integer *interactions, real radius,
                               integer numParticlesLocal, integer numParticles, integer numNodes) {
                //ExecutionPolicy executionPolicy(numParticlesLocal, ::SPH::Kernel::fixedRadiusNN, tree, particles, interactions,
                //                                numParticlesLocal, numParticles, numNodes);
                ExecutionPolicy executionPolicy; // 4 * numMultiProcessors, 256
                return cuda::launch(true, executionPolicy, ::SPH::Kernel::fixedRadiusNN, tree, particles, interactions,
                                    radius, numParticlesLocal, numParticles, numNodes);
            }

            real fixedRadiusNN_withinBox(Tree *tree, Particles *particles, integer *interactions, integer numParticlesLocal,
                               integer numParticles, integer numNodes) {
                //ExecutionPolicy executionPolicy(numParticlesLocal, ::SPH::Kernel::fixedRadiusNN, tree, particles, interactions,
                //                                numParticlesLocal, numParticles, numNodes);
                Logger(INFO) << "calling new fixed radius...";
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::SPH::Kernel::fixedRadiusNN_withinBox, tree, particles, interactions,
                                    numParticlesLocal, numParticles, numNodes);
            }

            real fixedRadiusNN_sharedMemory(Tree *tree, Particles *particles, integer *interactions, integer numParticlesLocal,
                               integer numParticles, integer numNodes) {
                size_t sharedMemory = 20 * 2 * sizeof(integer) * MAX_DEPTH;
                //int _blockSize;
                //int minGridSize;
                //int _gridSize;
                //cudaOccupancyMaxPotentialBlockSize(&minGridSize, &_blockSize, ::SPH::Kernel::fixedRadiusNN, sharedMemory, 0);
                //_gridSize = (numParticlesLocal + _blockSize - 1) / _blockSize;
                ExecutionPolicy executionPolicy(256, 10, sharedMemory);
                //printf("gridSize: %i, blockSize: %i\n", _gridSize, _blockSize);
                //ExecutionPolicy executionPolicy(_gridSize, _blockSize, sharedMemory);
                return cuda::launch(true, executionPolicy, ::SPH::Kernel::fixedRadiusNN_sharedMemory, tree, particles, interactions,
                                    numParticlesLocal, numParticles, numNodes);
            }

            real fixedRadiusNN_variableSML(Material *materials, Tree *tree, Particles *particles, integer *interactions,
                                           integer numParticlesLocal, integer numParticles,
                                           integer numNodes) {
                //ExecutionPolicy executionPolicy(256, 256);
                ExecutionPolicy executionPolicy; //(1, 256); // 256, 256
                return cuda::launch(true, executionPolicy, ::SPH::Kernel::fixedRadiusNN_variableSML, materials, tree, particles, interactions,
                                    numParticlesLocal, numParticles, numNodes);
            }

            real compTheta(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                           DomainList *lowestDomainList, Curve::Type curveType) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::SPH::Kernel::compTheta, subDomainKeyTree, tree,
                                    particles, lowestDomainList, curveType);
            }

            real symbolicForce(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                               DomainList *lowestDomainList, integer *sendIndices, real searchRadius,
                               integer n, integer m, integer relevantIndex,
                               Curve::Type curveType) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::SPH::Kernel::symbolicForce, subDomainKeyTree, tree, particles,
                                    lowestDomainList, sendIndices, searchRadius, n, m, relevantIndex, curveType);
            }

            real symbolicForce_test(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                    DomainList *lowestDomainList, integer *sendIndices, real searchRadius,
                                    integer n, integer m, integer relevantProc, integer relevantIndicesCounter,
                                    integer relevantIndexOld, Curve::Type curveType) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::SPH::Kernel::symbolicForce_test, subDomainKeyTree, tree, particles,
                                    lowestDomainList, sendIndices, searchRadius, n, m, relevantProc, relevantIndicesCounter,
                                    relevantIndexOld, curveType);
            }

            real symbolicForce_test2(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                DomainList *domainList, integer *sendIndices, real searchRadius,
                                                integer n, integer m, integer relevantProc, integer relevantIndicesCounter,
                                                Curve::Type curveType) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::SPH::Kernel::symbolicForce_test2, subDomainKeyTree, tree, particles,
                                    domainList, sendIndices, searchRadius, n, m, relevantProc, relevantIndicesCounter, curveType);
            }

            real collectSendIndices(Tree *tree, Particles *particles, integer *sendIndices,
                                    integer *particles2Send, integer *particlesCount,
                                    integer n, integer length, Curve::Type curveType) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::SPH::Kernel::collectSendIndices, tree, particles,
                                    sendIndices, particles2Send, particlesCount, n, length, curveType);
            }

            real collectSendIndices_test2(Tree *tree, Particles *particles, integer *sendIndices,
                                                     integer *particles2Send, integer *particlesCount,
                                                     integer numParticlesLocal, integer numParticles,
                                                     integer treeIndex, int currentProc, Curve::Type curveType) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::SPH::Kernel::collectSendIndices_test2, tree, particles,
                                    sendIndices, particles2Send, particlesCount, numParticlesLocal, numParticles,
                                    treeIndex, currentProc, curveType);
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
                ExecutionPolicy executionPolicy(24, 32); //(24, 32);//(1, 1)//(256,1);
                return cuda::launch(true, executionPolicy, ::SPH::Kernel::insertReceivedParticles, subDomainKeyTree,
                                    tree, particles, domainList, lowestDomainList, n, m);
            }

            real info(Tree *tree, Particles *particles, Helper *helper,
                      integer numParticlesLocal, integer numParticles, integer numNodes) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::SPH::Kernel::info, tree, particles, helper,
                                    numParticlesLocal, numParticles, numNodes);
            }

            real calculateCentersOfMass(Tree *tree, Particles *particles, integer level) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::SPH::Kernel::calculateCentersOfMass, tree,
                                    particles, level);
            }

            real determineSearchRadii(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                 DomainList *domainList, DomainList *lowestDomainList, real *searchRadii,
                                                 int n, int m, Curve::Type curveType) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::SPH::Kernel::determineSearchRadii, subDomainKeyTree, tree,
                                    particles, domainList, lowestDomainList, searchRadii, n, m, curveType);
            }
        }
    }

}
