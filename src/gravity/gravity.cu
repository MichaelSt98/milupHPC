#include "../../include/gravity/gravity.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

namespace Gravity {

    namespace Kernel {

        __global__ void globalCOM(Tree *tree, Particles *particles, real com[DIM]) {

            real mass = 0;
            for (int i=0; i<DIM; i++) {
                com[i] = 0;
            }
            for (int i=0; i<POW_DIM; i++) {
                if (tree->child[i] != -1) {
                    mass += particles->mass[tree->child[i]];
                    com[0] += particles->weightedEntry(tree->child[i], Entry::x);
#if DIM > 1
                    com[1] += particles->weightedEntry(tree->child[i], Entry::y);
#if DIM == 3
                    com[2] += particles->weightedEntry(tree->child[i], Entry::z);
#endif
#endif
                }
            }

            if (mass > 0) {
                com[0] /= mass;
#if DIM > 1
                com[1] /= mass;
#if DIM == 3
                com[2] /= mass;
#endif
#endif
            }

        }

        __global__ void collectSendIndices(Tree *tree, Particles *particles, integer *sendIndices,
                                           integer *particles2Send, integer *pseudoParticles2Send,
                                           integer *pseudoParticlesLevel,
                                           integer *particlesCount, integer *pseudoParticlesCount,
                                           integer n, integer length, Curve::Type curveType) {

            integer bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
            integer stride = blockDim.x*gridDim.x;
            integer offset = 0;

            integer particleInsertIndex;
            integer pseudoParticleInsertIndex;

            while ((bodyIndex + offset) < length) {

                if (sendIndices[bodyIndex + offset] == 1) {

                    // it is a particle
                    if (bodyIndex + offset < n) {
                        particleInsertIndex = atomicAdd(particlesCount, 1);
                        particles2Send[particleInsertIndex] = bodyIndex + offset;
                    }
                    // it is a pseudo-particle
                    else {
                        pseudoParticleInsertIndex = atomicAdd(pseudoParticlesCount, 1);
                        pseudoParticles2Send[pseudoParticleInsertIndex] = bodyIndex + offset;
                        pseudoParticlesLevel[pseudoParticleInsertIndex] = particles->level[bodyIndex + offset];
                        //printf("pseudo-particle level to be sent: %i (%i)\n", particles->level[bodyIndex + offset],
                        //       bodyIndex + offset);
                        //pseudoParticlesLevel[pseudoParticleInsertIndex] = tree->getTreeLevel(particles,
                        //                                                                     bodyIndex + offset,
                        //                                                                     MAX_LEVEL, curveType);

                        // debug
                        //if (pseudoParticlesLevel[pseudoParticleInsertIndex] == -1) {
                        //    printf("level = -1 within collectSendIndices for index: %i\n", bodyIndex + offset);
                        //}
                        // end: debug
                    }
                }
                __threadfence();
                offset += stride;
            }
        }

        __global__ void testSendIndices(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                        integer *sendIndices, integer *markedSendIndices,
                                        integer *levels, Curve::Type curveType, integer length) {

            integer bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
            integer stride = blockDim.x*gridDim.x;
            integer offset = 0;

            integer key;
            integer temp;
            integer childIndex;
            integer childPath;

            bool available = false;

            real min_x, max_x;
#if DIM > 1
            real min_y, max_y;
#if DIM == 3
            real min_z, max_z;
#endif
#endif

            /*while ((bodyIndex + offset) < length) {

                printf("x[%i] = (%f, %f, %f) %f\n", sendIndices[bodyIndex + offset], particles->x[sendIndices[bodyIndex + offset]],
                       particles->y[sendIndices[bodyIndex + offset]], particles->z[sendIndices[bodyIndex + offset]],
                       particles->mass[sendIndices[bodyIndex + offset]]);

                offset += stride;
            }*/

            //while ((bodyIndex + offset) < length) {
            //    if (particles->x[sendIndices[bodyIndex + offset]] == 0.f &&
            //        particles->y[sendIndices[bodyIndex + offset]] == 0.f &&
            //        particles->z[sendIndices[bodyIndex + offset]] == 0.f &&
            //        particles->mass[sendIndices[bodyIndex + offset]]) {
            //
            //    }
            //    offset += stride;
            //}

            // ///////////////////////////////////////////////////////////////////////////////////

            //if (bodyIndex == 0) {
            //    for (int i = 0; i<10; i++) {
            //        printf("sendIndices[%i] = %i (length = %i)\n", length - 1 + i, sendIndices[length -1 + i], length);
            //    }
            //}

            //if (bodyIndex == 0) {
            //    integer i=0;
            //    for (int i = 0; i<30000; i++) {
            //        if (markedSendIndices[100000 + i] == 1) {
            //            printf("[rank %i] markedSendIndices[%i] = %i!\n", subDomainKeyTree->rank, 100000 + i, markedSendIndices[100000 + i]);
            //            break;
            //        }
            //    }
            //}

            while ((bodyIndex + offset) < length) {

                //printf("index = %i sendIndex = %i level = %i\n", bodyIndex + offset, sendIndices[bodyIndex + offset],
                //       levels[bodyIndex + offset]);

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

                available = false;

                childIndex = 0;
                if (levels[bodyIndex + offset] > 3) {
                    //key = tree->getParticleKey(particles, bodyIndex + offset + tree->toDeleteNode[0], MAX_LEVEL,
                    //                           curveType);

                    //printf("level = %i\n", levels[bodyIndex + offset]);

                    childIndex = 0;

                    for (int j = 0; j < levels[bodyIndex + offset] - 1; j++) {

                        temp = childIndex;

                        childPath = 0;
                        if (particles->x[sendIndices[bodyIndex + offset]] < 0.5 * (min_x + max_x)) {
                            childPath += 1;
                            max_x = 0.5 * (min_x + max_x);
                        } else {
                            min_x = 0.5 * (min_x + max_x);
                        }
#if DIM > 1
                        if (particles->y[sendIndices[bodyIndex + offset]] < 0.5 * (min_y + max_y)) {
                            childPath += 2;
                            max_y = 0.5 * (min_y + max_y);
                        } else {
                            min_y = 0.5 * (min_y + max_y);
                        }
#if DIM == 3
                        if (particles->z[sendIndices[bodyIndex + offset]] < 0.5 * (min_z + max_z)) {
                            childPath += 4;
                            max_z = 0.5 * (min_z + max_z);
                        } else {
                            min_z = 0.5 * (min_z + max_z);
                        }
#endif
#endif
                        //printf("childIndex = %i\n", childIndex);
                        childIndex = tree->child[POW_DIM * temp + childPath];
                        if (bodyIndex + offset == 0) {
                            printf("tree->child[POW_DIM * %i + %i] = %i (%i)\n", temp, childPath, tree->child[POW_DIM * temp + childPath], sendIndices[bodyIndex + offset]);
                        }
                    }


                    for (int i = 0; i < length; i++) {
                        if (temp == sendIndices[i]) {
                            available = true;
                            break;
                        }
                    }

                    if (!available) {
                        //integer a = -1;
                        //markedSendIndices[childIndex] = a;
                        printf("[rank %i] %i (relevant son: %i) NOT Available sendIndices[%i] = %i, [%i] = %i)!\n",
                               subDomainKeyTree->rank, temp, childIndex,
                               childIndex, markedSendIndices[childIndex], temp, markedSendIndices[temp]);
                        assert(0);
                    }

                    //if (childIndex != sendIndices[bodyIndex + offset]) {
                        //printf("ATTENTION childIndex != bodyIndex level = %i (%i != %i) (%f, %f, %f)!\n", levels[bodyIndex + offset], childIndex, sendIndices[bodyIndex + offset],
                        //       particles->x[sendIndices[bodyIndex + offset]], particles->y[sendIndices[bodyIndex + offset]],
                        //       particles->z[sendIndices[bodyIndex + offset]]);
                    //} else {
                        //printf("--\n");
                    //}
                }

                offset += stride;
            }
        }

        __global__ void zeroDomainListNodes(Particles *particles, DomainList *domainList,
                                            DomainList *lowestDomainList) {

            integer bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
            integer stride = blockDim.x*gridDim.x;
            integer offset = 0;
            integer domainIndex;
            bool zero;

            while ((bodyIndex + offset) < *domainList->domainListIndex) {
                zero = true;
                domainIndex = domainList->domainListIndices[bodyIndex + offset];
                //for (int i=0; i<*lowestDomainList->domainListIndex; i++) {
                //    if (domainIndex == lowestDomainList->domainListIndices[i]) {
                //        zero = false;
                //        break;
                //    }
                //}

                if (particles->nodeType[domainIndex] == 2) {
                    zero = false;
                }

                if (zero) {
                    //printf("zero %i\n", domainIndex);
                    particles->x[domainIndex] = (real)0;
#if DIM > 1
                    particles->y[domainIndex] = (real)0;
#if DIM == 3
                    particles->z[domainIndex] = (real)0;
#endif
#endif

                    particles->mass[domainIndex] = (real)0;
                }
                /*
                else {
                    //printf("domainIndex = %i *= mass = %f\n", domainIndex, particles->mass[domainIndex]);
                    particles->x[domainIndex] *= particles->mass[domainIndex];
#if DIM > 1
                    particles->y[domainIndex] *= particles->mass[domainIndex];
#if DIM == 3
                    particles->z[domainIndex] *= particles->mass[domainIndex];
#endif
#endif
                }
                */


                offset += stride;
            }

        }

        __global__ void prepareLowestDomainExchange(Particles *particles, DomainList *lowestDomainList,
                                                    Helper *helper, Entry::Name entry) {

            integer bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
            integer stride = blockDim.x*gridDim.x;
            integer offset = 0;
            integer index;
            integer lowestDomainIndex;

            //copy x, y, z, mass of lowest domain list nodes into arrays
            //sorting using cub (not here)
            while ((bodyIndex + offset) < *lowestDomainList->domainListIndex) {
                lowestDomainIndex = lowestDomainList->domainListIndices[bodyIndex + offset];
                if (lowestDomainIndex >= 0) {
                    switch (entry) {
                        case Entry::x: {
                            helper->realBuffer[bodyIndex + offset] = particles->x[lowestDomainIndex];
                        } break;
#if DIM > 1
                        case Entry::y: {
                            helper->realBuffer[bodyIndex + offset] = particles->y[lowestDomainIndex];
                        } break;
#if DIM == 3
                        case Entry::z: {
                            helper->realBuffer[bodyIndex + offset] = particles->z[lowestDomainIndex];
                        } break;
#endif
#endif
                        case Entry::mass: {
                            helper->realBuffer[bodyIndex + offset] = particles->mass[lowestDomainIndex];
                        } break;
                        default:
                            printf("prepareLowestDomainExchange(): Not available!\n");
                    }
                }
                offset += stride;
            }
        }

        __global__ void updateLowestDomainListNodes(Particles *particles, DomainList *lowestDomainList,
                                                    Helper *helper, Entry::Name entry) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            integer originalIndex = -1;

            while ((bodyIndex + offset) < *lowestDomainList->domainListIndex) {
                originalIndex = -1;
                for (int i = 0; i < *lowestDomainList->domainListIndex; i++) {
                    if (lowestDomainList->sortedDomainListKeys[bodyIndex + offset] ==
                        lowestDomainList->domainListKeys[i]) {
                        originalIndex = i;
                    }
                }

                if (originalIndex == -1) {
                    cudaTerminate("ATTENTION: originalIndex = -1 (index = %i)!\n",
                           lowestDomainList->sortedDomainListKeys[bodyIndex + offset]);
                }

                switch (entry) {
                    case Entry::x: {
                        particles->x[lowestDomainList->domainListIndices[originalIndex]] =
                                helper->realBuffer[DOMAIN_LIST_SIZE + bodyIndex + offset];
                    } break;
#if DIM > 1
                    case Entry::y: {
                        particles->y[lowestDomainList->domainListIndices[originalIndex]] =
                                helper->realBuffer[DOMAIN_LIST_SIZE + bodyIndex + offset];
                    } break;
#if DIM == 3
                    case Entry::z: {
                        particles->z[lowestDomainList->domainListIndices[originalIndex]] =
                                helper->realBuffer[DOMAIN_LIST_SIZE + bodyIndex + offset];
                    } break;
#endif
#endif
                    case Entry::mass: {
                        particles->mass[lowestDomainList->domainListIndices[originalIndex]] =
                                helper->realBuffer[DOMAIN_LIST_SIZE + bodyIndex + offset];
                    } break;
                    default: {
                        printf("Entry not available!\n");
                    }
                }

                offset += stride;
            }
        }

        __global__ void compLowestDomainListNodes(Tree *tree, Particles *particles, DomainList *lowestDomainList) {

            integer bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
            integer stride = blockDim.x*gridDim.x;
            integer offset = 0;
            integer lowestDomainIndex;
            bool divide;

            while ((bodyIndex + offset) < *lowestDomainList->domainListIndex) {

                //divide = false;
                lowestDomainIndex = lowestDomainList->domainListIndices[bodyIndex + offset];

                //for (int child=0; child<POW_DIM; child++) {
                //    if (tree->child[POW_DIM * lowestDomainIndex + child] != -1) {
                //        printf("lowestDomainIndex: tree->child[8 * %i + %i] = %i\n", lowestDomainIndex, child,
                //               tree->child[POW_DIM * lowestDomainIndex + child]);
                //        divide = true;
                //        break;
                //    }
                //}

                //if (particles->mass[lowestDomainIndex] != (real)0) {
                //if (particles->mass[lowestDomainIndex] > (real)0) {
                if (/*divide && */particles->mass[lowestDomainIndex] > (real)0) {

#if DIM == 3
                    //printf("lowestDomainIndex: %i (%f, %f, %f) %f\n", lowestDomainIndex, particles->x[lowestDomainIndex],
                    //       particles->y[lowestDomainIndex], particles->z[lowestDomainIndex], particles->mass[lowestDomainIndex]);
#endif

                    /*particles->x[lowestDomainIndex] /= particles->mass[lowestDomainIndex];
#if DIM > 1
                    particles->y[lowestDomainIndex] /= particles->mass[lowestDomainIndex];
#if DIM == 3
                    particles->z[lowestDomainIndex] /= particles->mass[lowestDomainIndex];
#endif
#endif
                     */

                    //if (std::isnan(particles->x[lowestDomainIndex])) {
                    //    printf("NAN! within compLowestDomainListNodes for %i: x = %f, m = %f\n", lowestDomainIndex, particles->x[lowestDomainIndex],
                    //           particles->mass[lowestDomainIndex]);
                    //    assert(0);
                    //}

#if DIM == 3
                    //printf("lowestDomainIndex: %i (%f, %f, %f) %f\n", lowestDomainIndex, particles->x[lowestDomainIndex],
                    //       particles->y[lowestDomainIndex], particles->z[lowestDomainIndex], particles->mass[lowestDomainIndex]);
#endif
                }

                //printf("lowestDomainIndex = %i (%f, %f, %f) %f\n", lowestDomainIndex, particles->x[lowestDomainIndex],
                //       particles->y[lowestDomainIndex], particles->z[lowestDomainIndex], particles->mass[lowestDomainIndex]);

                offset += stride;
            }
        }

        __global__ void compLocalPseudoParticles(Tree *tree, Particles *particles, DomainList *domainList, int n) {

            integer bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
            integer stride = blockDim.x*gridDim.x;
            integer offset = 0;
            bool isDomainList;

            bodyIndex += n;

            while (bodyIndex + offset < *tree->index) {
                isDomainList = false;

                //for (integer i=0; i<*domainList->domainListIndex; i++) {
                //    if ((bodyIndex + offset) == domainList->domainListIndices[i]) {
                //        isDomainList = true; // hence do not insert
                //        break;
                //    }
                //}
                if (particles->nodeType[bodyIndex + offset] >= 1) {
                    isDomainList = true;
                }

                if (/*particles->mass[bodyIndex + offset] != 0 && */!isDomainList) {
                    particles->x[bodyIndex + offset] /= particles->mass[bodyIndex + offset];
#if DIM > 1
                    particles->y[bodyIndex + offset] /= particles->mass[bodyIndex + offset];
#if DIM == 3
                    particles->z[bodyIndex + offset] /= particles->mass[bodyIndex + offset];
#endif
#endif
                }

                offset += stride;
            }
        }

        __global__ void compDomainListPseudoParticlesPerLevel(Tree *tree, Particles *particles, DomainList *domainList,
                                                              DomainList *lowestDomainList, int n, int level) {

            integer bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
            integer stride = blockDim.x*gridDim.x;
            integer offset;

            integer domainIndex;
            //integer level = MAX_LEVEL; // max level
            bool compute;

            real totalMass, x_mass, y_mass, z_mass;
            int childToLookAt = 5;
            int lowestDomainIndex, childLowestDomain;

            offset = 0;
            compute = true;
            while ((bodyIndex + offset) < *domainList->domainListIndex) {
                compute = true;
                domainIndex = domainList->domainListIndices[bodyIndex + offset];
                if (particles->nodeType[domainIndex] == 2) {
                    compute = false;
                }
                //for (int i=0; i<*lowestDomainList->domainListIndex; i++) {
                //    if (domainIndex == lowestDomainList->domainListIndices[i]) {
                //        compute = false;
                //        /*lowestDomainIndex = domainIndex;
                //        while (lowestDomainIndex != -1) {
                //            //childLowestDomain;
                //            printf("(%lu) lowestDomainIndex: %i (%e, %e, %e) %e\n", lowestDomainList->domainListKeys[i],
                //                   lowestDomainIndex,
                //                   particles->x[lowestDomainIndex], particles->y[lowestDomainIndex],
                //                   particles->z[lowestDomainIndex], particles->mass[lowestDomainIndex]);
                //            totalMass = 0.;
                //            x_mass = 0.;
                //            y_mass = 0.;
                //            z_mass = 0.;
                //            for (int i_child = 0; i_child < POW_DIM; i_child++) {
                //                childLowestDomain = tree->child[POW_DIM * lowestDomainIndex + i_child];
                //                if (childLowestDomain != -1) {
                //                    //printf("totalMass += %e\n", particles->mass[childLowestDomain]);
                //                    totalMass += particles->mass[childLowestDomain];
                //                    x_mass += particles->mass[childLowestDomain] * particles->x[childLowestDomain];
                //                    y_mass += particles->mass[childLowestDomain] * particles->y[childLowestDomain];
                //                    z_mass += particles->mass[childLowestDomain] * particles->z[childLowestDomain];
                //                    //if (totalMass > 0.) {
                //                    //    printf("totalMass = %e > %e\n", totalMass, particles->mass[lowestDomainIndex]);
                //                    //}
                //                }
                //                //if (particles->mass[childLowestDomain] > 0.) {
                //                //    printf("!= 0: %e\n", particles->mass[childLowestDomain]);
                //                //}
                //                printf("(%lu) lowestDomainIndex: %i child #%i -> %i (%e, %e, %e) %e\n",
                //                       lowestDomainList->domainListKeys[i],
                //                       lowestDomainIndex, i_child, childLowestDomain,
                //                       particles->x[childLowestDomain], particles->y[childLowestDomain],
                //                       particles->z[childLowestDomain], particles->mass[childLowestDomain]);
                //            }
                //            x_mass /= totalMass;
                //            y_mass /= totalMass;
                //            z_mass /= totalMass;
                //            if (totalMass > (particles->mass[lowestDomainIndex] + 1e-7)) { //||
                //                //x_mass < (particles->x[lowestDomainIndex] - 1e-7) ||
                //                //y_mass < (particles->y[lowestDomainIndex] - 1e-7) ||
                //                //z_mass < (particles->z[lowestDomainIndex] - 1e-7)) {
                //                //if (totalMass > 0.) {
                //                printf("totalMass = %e > %e\n", totalMass, particles->mass[lowestDomainIndex]);
                //                //}
                //                assert(0);
                //            }
                //            lowestDomainIndex = tree->child[POW_DIM * lowestDomainIndex + childToLookAt];
                //        }*/
                //        break;
                //    }
                //}
                if (compute && domainList->domainListLevels[bodyIndex + offset] == level) {
                    // do the calculation
                    //particles->x[domainIndex] = 0.;
                    //particles->y[domainIndex] = 0.;
                    //particles->z[domainIndex] = 0.;
                    //particles->mass[domainIndex] = 0.;
                    for (int i=0; i<POW_DIM; i++) {
                        particles->x[domainIndex] += particles->x[tree->child[POW_DIM*domainIndex + i]] *
                                                     particles->mass[tree->child[POW_DIM*domainIndex + i]];
#if DIM > 1
                        particles->y[domainIndex] += particles->y[tree->child[POW_DIM*domainIndex + i]] *
                                                     particles->mass[tree->child[POW_DIM*domainIndex + i]];
#if DIM == 3
                        particles->z[domainIndex] += particles->z[tree->child[POW_DIM*domainIndex + i]] *
                                                     particles->mass[tree->child[POW_DIM*domainIndex + i]];
#endif
#endif
                        particles->mass[domainIndex] += particles->mass[tree->child[POW_DIM*domainIndex + i]];
                    }

                    if (particles->mass[domainIndex] != 0.f) {
                        particles->x[domainIndex] /= particles->mass[domainIndex];
#if DIM > 1
                        particles->y[domainIndex] /= particles->mass[domainIndex];
#if DIM == 3
                        particles->z[domainIndex] /= particles->mass[domainIndex];
#endif
#endif
                    }
                }
                offset += stride;
            }
            __syncthreads();
        }

        __global__ void compDomainListPseudoParticles(Tree *tree, Particles *particles, DomainList *domainList,
                                                      DomainList *lowestDomainList, int n) {
            //calculate position (center of mass) and mass for domain list nodes
            //Problem: start with "deepest" nodes
            integer bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
            integer stride = blockDim.x*gridDim.x;
            integer offset;

            integer domainIndex;
            integer level = MAX_LEVEL; // max level
            bool compute;

            // go from max level to level=0
            while (level >= 0) {
                offset = 0;
                compute = true;
                while ((bodyIndex + offset) < *domainList->domainListIndex) {
                    compute = true;
                    domainIndex = domainList->domainListIndices[bodyIndex + offset];
                    if (particles->nodeType[domainIndex] == 2) {
                        compute = false;
                    }
                    //for (int i=0; i<*lowestDomainList->domainListIndex; i++) {
                    //    if (domainIndex == lowestDomainList->domainListIndices[i]) {
                    //        compute = false;
                    //    }
                    //}
                    if (compute && domainList->domainListLevels[bodyIndex + offset] == level) {
                        // do the calculation
                        for (int i=0; i<POW_DIM; i++) {
                            particles->x[domainIndex] += particles->x[tree->child[POW_DIM*domainIndex + i]] *
                                    particles->mass[tree->child[POW_DIM*domainIndex + i]];
#if DIM > 1
                            particles->y[domainIndex] += particles->y[tree->child[POW_DIM*domainIndex + i]] *
                                    particles->mass[tree->child[POW_DIM*domainIndex + i]];
#if DIM == 3
                            particles->z[domainIndex] += particles->z[tree->child[POW_DIM*domainIndex + i]] *
                                    particles->mass[tree->child[POW_DIM*domainIndex + i]];
#endif
#endif
                            particles->mass[domainIndex] += particles->mass[tree->child[POW_DIM*domainIndex + i]];
                        }

                        if (particles->mass[domainIndex] != 0.f) {
                            particles->x[domainIndex] /= particles->mass[domainIndex];
#if DIM > 1
                            particles->y[domainIndex] /= particles->mass[domainIndex];
#if DIM == 3
                            particles->z[domainIndex] /= particles->mass[domainIndex];
#endif
#endif
                        }
                    }
                    offset += stride;
                }
                __syncthreads();
                level--;
            }
        }

        __global__ void computeForces_v1(Tree *tree, Particles *particles, real radius, integer n, integer m,
                                         SubDomainKeyTree *subDomainKeyTree, real theta, real smoothing) {

            register int i, ii;
            int child, nodeIndex, childNumber, depth;

            real px, ax, dx, f, distance;
#if DIM > 1
            real py, ay, dy;
#if DIM == 3
            real pz, az, dz;
#endif
#endif
            real sml;
            real thetasq = theta*theta;

            int currentNodeIndex[MAX_DEPTH];
            int currentChildNumber[MAX_DEPTH];

            __shared__ volatile real cellSize[MAX_DEPTH];

            if (threadIdx.x == 0) {
                cellSize[0] = 4.0 * radius * radius; //4.0 * radius * radius; //TODO: original one is 4.0 * radi...
#pragma unroll
                for (i = 1; i < MAX_DEPTH; i++) {
                    cellSize[i] = cellSize[i - 1] * 0.25;
                }
            }

            __syncthreads();

            for (ii = threadIdx.x + blockIdx.x * blockDim.x; ii < n; ii += blockDim.x * gridDim.x) {

                i = tree->sorted[ii]; //i = ii;

                px = particles->x[i];
#if DIM > 1
                py = particles->y[i];
#if DIM == 3
                pz = particles->z[i];
#endif
#endif
                //particles->ax[i] = 0.0;
                particles->g_ax[i] = 0.0;
#if DIM > 1
                //particles->ay[i] = 0.0;
                particles->g_ay[i] = 0.0;
#if DIM == 3
                //particles->az[i] = 0.0;
                particles->g_az[i] = 0.0;
#endif
#endif
                ax = 0.0;
#if DIM > 1
                ay = 0.0;
#if DIM == 3
                az = 0.0;
#endif
#endif

                // start at root
                depth = 1;
                currentNodeIndex[depth] = 0;
                currentChildNumber[depth] = 0;

                do {
                    childNumber = currentChildNumber[depth];
                    nodeIndex = currentNodeIndex[depth];

                    while(childNumber < POW_DIM) {
                        do {
                            child = tree->child[POW_DIM * nodeIndex + childNumber];
                            childNumber++;
                        } while(child == -1 && childNumber < POW_DIM);

                        if (child != -1 && child != i) { // dont do self-gravity with yourself!
                            dx = particles->x[child] - px;
                            distance = dx*dx + smoothing; //150329404.287723; //(0.0317 * 0.0317); //0.025;
#if DIM > 1
                            dy = particles->y[child] - py;
                            distance += dy*dy;
#endif
#if DIM == 3
                            dz = particles->z[child] - pz;
                            distance += dz*dz;
#endif
                            // if child is leaf or far away
                            if (child < n || distance * thetasq > cellSize[depth]) {
                                distance = cuda::math::sqrt(distance);
#if SI_UNITS
                                f = Constants::G * particles->mass[child] / (distance * distance * distance);
#else
                                f = particles->mass[child] / (distance * distance * distance);
#endif

                                ax += f*dx;
#if DIM > 1
                                ay += f*dy;
#if DIM == 3
                                az += f*dz;
#endif
#endif
                                //TODO: some flag for calculating potential energy
                                // gravitational potential energy
                                //particles->u[i] -= 0.5 * (particles->mass[child] * particles->mass[i])/distance;
                                // end: gravitational potential energy
                            } else {
                                // put child on stack
                                currentChildNumber[depth] = childNumber;
                                currentNodeIndex[depth] = nodeIndex;
                                depth++;
                                if (depth == MAX_DEPTH) {
                                    cudaTerminate("depth = %i >= MAX_DEPTH = %i\n", depth, MAX_DEPTH);
                                }
                                childNumber = 0;
                                nodeIndex = child;
                            }
                        }
                    }
                    depth--;
                } while(depth > 0);

                //particles->ax[i] = ax;
                particles->g_ax[i] = ax;
#if DIM > 1
                //particles->ay[i] = ay;
                particles->g_ay[i] = ay;
#if DIM == 3
                //particles->az[i] = az;
                particles->g_az[i] = az;
#endif
#endif
            }
        }

        __global__ void computeForces_v1_1(Tree *tree, Particles *particles, real radius, integer n, integer m,
                                         SubDomainKeyTree *subDomainKeyTree, real theta, real smoothing) {

            integer i, ii, child, nodeIndex, childNumber, depth;

            real px, ax, dx, f, distance;
#if DIM > 1
            real py, ay, dy;
#if DIM == 3
            real pz, az, dz;
#endif
#endif

            real sml;
            real thetasq = theta*theta;

            integer currentNodeIndex[MAX_DEPTH];
            integer currentChildNumber[MAX_DEPTH];

            __shared__ volatile real cellSize[MAX_DEPTH];

            if (threadIdx.x == 0) {
                cellSize[0] = 4.0 * radius * radius;
#pragma unroll
                for (i = 1; i < MAX_DEPTH; i++) {
                    cellSize[i] = cellSize[i - 1] * 0.25;
                }
            }

            __syncthreads();

            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {

                px = particles->x[i];
#if DIM > 1
                py = particles->y[i];
#if DIM == 3
                pz = particles->z[i];
#endif
#endif
                //particles->ax[i] = 0.0;
                particles->g_ax[i] = 0.0;
#if DIM > 1
                //particles->ay[i] = 0.0;
                particles->g_ay[i] = 0.0;
#if DIM == 3
                //particles->az[i] = 0.0;
                particles->g_az[i] = 0.0;
#endif
#endif
                ax = 0.0;
#if DIM > 1
                ay = 0.0;
#if DIM == 3
                az = 0.0;
#endif
#endif

                // start at root
                depth = 1;
                currentNodeIndex[depth] = 0;
                currentChildNumber[depth] = 0;

                do {
                    childNumber = currentChildNumber[depth];
                    nodeIndex = currentNodeIndex[depth];

                    while(childNumber < POW_DIM) {
                        do {
                            child = tree->child[POW_DIM * nodeIndex + childNumber]; //childList[childListIndex(nodeIndex, childNumber)];
                            childNumber++;
                        } while(child == -1 && childNumber < POW_DIM);
                        if (child != -1 && child != i) { // dont do self-gravity with yourself!
                            dx = particles->x[child] - px;
                            distance = dx*dx + smoothing; //150329404.287723; //(0.0317 * 0.0317); //0.025;
#if DIM > 1
                            dy = particles->y[child] - py;
                            distance += dy*dy;
#endif
#if DIM == 3
                            dz = particles->z[child] - pz;
                            distance += dz*dz;
#endif
                            // if child is leaf or far away
                            if (child < n || distance * thetasq > cellSize[depth]) {
                                distance = sqrt(distance);
#if SI_UNITS
                                f = Constants::G * particles->mass[child] / (distance * distance * distance);
#else
                                f = particles->mass[child] / (distance * distance * distance);
#endif

                                ax += f*dx;
#if DIM > 1
                                ay += f*dy;
#if DIM == 3
                                az += f*dz;
#endif
#endif
                                //TODO: some flag for calculating potential energy
                                // gravitational potential energy
                                //particles->u[i] -= 0.5 * (particles->mass[child] * particles->mass[i])/distance;
                                // end: gravitational potential energy
                            } else {
                                // put child on stack
                                currentChildNumber[depth] = childNumber;
                                currentNodeIndex[depth] = nodeIndex;
                                depth++;
                                if (depth == MAX_DEPTH) {
                                    cudaTerminate("depth = %i >= MAX_DEPTH = %i\n", depth, MAX_DEPTH);
                                }
                                childNumber = 0;
                                nodeIndex = child;
                            }
                        }
                    }
                    depth--;
                } while(depth > 0);

                //particles->ax[i] = ax;
                particles->g_ax[i] = ax;
#if DIM > 1
                //particles->ay[i] = ay;
                particles->g_ay[i] = ay;
#if DIM == 3
                //particles->az[i] = az;
                particles->g_az[i] = az;
#endif
#endif
            }
        }

        __global__ void computeForces_v1_2(Tree *tree, Particles *particles, real radius, integer n, integer m,
                                           SubDomainKeyTree *subDomainKeyTree, real theta, real smoothing) {

            register int i, ii;
            int child, nodeIndex, childNumber, depth;

            real px, ax, dx, f, distance;
#if DIM > 1
            real py, ay, dy;
#if DIM == 3
            real pz, az, dz;
#endif
#endif
            real sml;
            real thetasq = theta*theta;

            __shared__ int currentNodeIndex[MAX_DEPTH];
            __shared__ int currentChildNumber[MAX_DEPTH];

            __shared__ volatile real cellSize[MAX_DEPTH];

            if (threadIdx.x == 0) {
                cellSize[0] = 4.0 * radius * radius;
                #pragma unroll
                for (i = 1; i < MAX_DEPTH; i++) {
                    cellSize[i] = cellSize[i - 1] * 0.25;
                }
            }

            __syncthreads();

            for (ii = threadIdx.x + blockIdx.x * blockDim.x; ii < n; ii += blockDim.x * gridDim.x) {

                i = tree->sorted[ii]; //i = ii;

                px = particles->x[i];
#if DIM > 1
                py = particles->y[i];
#if DIM == 3
                pz = particles->z[i];
#endif
#endif
                //particles->ax[i] = 0.0;
                particles->g_ax[i] = 0.0;
#if DIM > 1
                //particles->ay[i] = 0.0;
                particles->g_ay[i] = 0.0;
#if DIM == 3
                //particles->az[i] = 0.0;
                particles->g_az[i] = 0.0;
#endif
#endif
                ax = 0.0;
#if DIM > 1
                ay = 0.0;
#if DIM == 3
                az = 0.0;
#endif
#endif

                // start at root
                depth = 1;
                currentNodeIndex[depth] = 0;
                currentChildNumber[depth] = 0;

                do {
                    childNumber = currentChildNumber[depth];
                    nodeIndex = currentNodeIndex[depth];

                    while(childNumber < POW_DIM) {
                        do {
                            child = tree->child[POW_DIM * nodeIndex + childNumber];
                            childNumber++;
                        } while(child == -1 && childNumber < POW_DIM);
                        if (child != -1 && child != i) { // dont do self-gravity with yourself!
                            dx = particles->x[child] - px;
                            distance = dx*dx + smoothing; //150329404.287723; //(0.0317 * 0.0317); //0.025;
#if DIM > 1
                            dy = particles->y[child] - py;
                            distance += dy*dy;
#endif
#if DIM == 3
                            dz = particles->z[child] - pz;
                            distance += dz*dz;
#endif
                            // if child is leaf or far away
                            if (child < n || distance * thetasq > cellSize[depth]) {
                                distance = cuda::math::sqrt(distance);
#if SI_UNITS
                                f = Constants::G * particles->mass[child] / (distance * distance * distance);
#else
                                f = particles->mass[child] / (distance * distance * distance);
#endif

                                ax += f*dx;
#if DIM > 1
                                ay += f*dy;
#if DIM == 3
                                az += f*dz;
#endif
#endif
                                //TODO: some flag for calculating potential energy
                                // gravitational potential energy
                                //particles->u[i] -= 0.5 * (particles->mass[child] * particles->mass[i])/distance;
                                // end: gravitational potential energy
                            } else {
                                // put child on stack
                                currentChildNumber[depth] = childNumber;
                                currentNodeIndex[depth] = nodeIndex;
                                depth++;
                                if (depth == MAX_DEPTH) {
                                    cudaTerminate("depth = %i >= MAX_DEPTH = %i\n", depth, MAX_DEPTH);
                                }
                                childNumber = 0;
                                nodeIndex = child;
                            }
                        }
                    }
                    depth--;
                } while(depth > 0);

                //particles->ax[i] = ax;
                particles->g_ax[i] = ax;
#if DIM > 1
                //particles->ay[i] = ay;
                particles->g_ay[i] = ay;
#if DIM == 3
                //particles->az[i] = az;
                particles->g_az[i] = az;
#endif
#endif
            }
        }

        // see: https://iss.oden.utexas.edu/Publications/Papers/burtscher11.pdf
        /*__global__ void test() {
            // precompute and cache info
            // determine first thread in each warp
            for (//sorted body indexes assigned to me) {
                // cache body data
                // initialize iteration stack
                depth = 0;
                while (depth >= 0) {
                    while (//there are more nodes to visit) {
                        if (//I’m the first thread in the warp) {
                            // move on to next node
                            // read node data and put in shared memory
                        }
                        threadfence block();
                        if (//node is not null) {
                            // get node data from shared memory
                            // compute distance to node
                            if ((//node is a body) || all(//distance >= cutoff)) {
                                // compute interaction force contribution
                            } else {
                                depth++; // descend to next tree level
                                if (//I’m the first thread in the warp) {
                                    // push node’s children onto iteration stack
                                }
                                threadfence block();
                            }
                        } else {
                            depth = max(0, depth-1); // early out because remaining nodes are also null
                        }
                    }
                    depth−−;
                }
            // update body data
            }
        }*/

        __global__ void computeForces_v2(Tree *tree, Particles *particles, real radius, integer n, integer m,
                                         integer blockSize, integer warp, integer stackSize,
                                         SubDomainKeyTree *subDomainKeyTree, real theta,
                                         real smoothing) {

            integer bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
            integer stride = blockDim.x*gridDim.x;
            integer offset = 0;

            register int sortedIndex;

            //__shared__ real depth[stackSize * blockSize/warp];
            //__shared__ integer stack[stackSize * blockSize/warp];
            extern __shared__ real buffer[];

            real* depth = (real*)buffer;
            integer* stack = (integer*)&depth[stackSize * blockSize/warp];

            real pos_x;
#if DIM > 1
            real pos_y;
#if DIM == 3
            real pos_z;
#endif
#endif

            real acc_x;
#if DIM > 1
            real acc_y;
#if DIM == 3
            real acc_z;
#endif
#endif

            // in case that one of the first children are a leaf
            int jj = -1;
            #pragma unroll
            for (integer i=0; i<POW_DIM; i++) {
                if (tree->child[i] != -1) {
                    jj++;
                }
            }

            int counter = threadIdx.x % warp;
            int stackStartIndex = stackSize*(threadIdx.x / warp);

            while ((bodyIndex + offset) < n) {

                //sortedIndex = bodyIndex + offset;
                sortedIndex = tree->sorted[bodyIndex + offset];

                pos_x = particles->x[sortedIndex];
#if DIM > 1
                pos_y = particles->y[sortedIndex];
#if DIM == 3
                pos_z = particles->z[sortedIndex];
#endif
#endif

                acc_x = 0.0;
#if DIM > 1
                acc_y = 0.0;
#if DIM == 3
                acc_z = 0.0;
#endif
#endif

                // initialize stack
                integer top = jj + stackStartIndex;

                if (counter == 0) {

                    int temp = 0;

                    #pragma unroll
                    for (int i=0; i<POW_DIM; i++) {
                        if (tree->child[i] != -1) {
                            stack[stackStartIndex + temp] = tree->child[i];
                            depth[stackStartIndex + temp] = radius*radius/theta;
                            temp++;
                        }
                    }
                }
                __syncthreads();

                // while stack is not empty / more nodes to visit
                while (top >= stackStartIndex) {

                    integer node = stack[top];

                    real dp = 0.25 * depth[top]; //powf(0.5, DIM) * depth[top]; //0.25*depth[top]; // float dp = depth[top];

                    for (integer i=0; i<POW_DIM; i++) {

                        integer ch = tree->child[POW_DIM*node + i];

                        //__threadfence();

                        if (ch >= 0) {

                            real dx = particles->x[ch] - pos_x;
#if DIM > 1
                            real dy = particles->y[ch] - pos_y;
#if DIM == 3
                            real dz = particles->z[ch] - pos_z;
#endif
#endif

                            real r = dx*dx + smoothing; // SMOOTHING
#if DIM > 1
                            r += dy*dy;
#if DIM == 3
                            r += dz*dz;
#endif
#endif

                            //if (ch < n /*is leaf node*/ || !__any_sync(activeMask, dp > r)) {
                            if (ch < m /*is leaf node*/ || __all_sync(__activemask(), dp <= r)) {

                                // calculate interaction force contribution
                                if (r > 0.f) { //NEW //TODO: how to avoid r = 0?
                                    r = cuda::math::rsqrt(r);
                                }

#if SI_UNITS
                                real f = Constants::G * particles->mass[ch] * r * r * r;
#else
                                real f = particles->mass[ch] * r * r * r;
#endif

                                acc_x += f*dx;
#if DIM > 1
                                acc_y += f*dy;
#if DIM == 3
                                acc_z += f*dz;
#endif
#endif
                            }
                            else {
                                // if first thread in warp: push node's children onto iteration stack
                                if (counter == 0) {
                                    stack[top] = ch;
                                    depth[top] = dp; // depth[top] = 0.25*dp;
                                }
                                top++; // descend to next tree level
                                __threadfence_block();
                            }
                        }
                        // this is not working
                        //else {
                        //    top = cuda::math::max(stackStartIndex, top-1);
                        //}
                    }
                    top--;
                }
                // update body data
                particles->g_ax[sortedIndex] = acc_x;
#if DIM > 1
                particles->g_ay[sortedIndex] = acc_y;
#if DIM == 3
                particles->g_az[sortedIndex] = acc_z;
#endif
#endif

                offset += stride;
                __syncthreads();
            }

        }

        __global__ void computeForces_v2_1(Tree *tree, Particles *particles, integer n, integer m, integer blockSize,
                                           integer warp, integer stackSize, SubDomainKeyTree *subDomainKeyTree,
                                           real theta, real smoothing) {

            integer bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
            integer stride = blockDim.x*gridDim.x;
            integer offset = 0;

            //__shared__ real depth[stackSize * blockSize/warp];
            //__shared__ integer stack[stackSize * blockSize/warp];
            extern __shared__ real buffer[];

            real* depth = (real*)buffer;
            integer* stack = (integer*)&depth[stackSize * blockSize/warp];

            real x_radius = 0.5*(*tree->maxX - (*tree->minX));
#if DIM > 1
            real y_radius = 0.5*(*tree->maxY - (*tree->minY));
#if DIM == 3
            real z_radius = 0.5*(*tree->maxZ - (*tree->minZ));
#endif
#endif

#if DIM == 1
            real radius = x_radius;
#elif DIM == 2
            real radius = cuda::math::max(x_radius, y_radius);
#else
            real radius_max = cuda::math::max(x_radius, y_radius);
            real radius = cuda::math::max(radius_max, z_radius);
#endif

            // in case that one of the first children are a leaf
            integer jj = -1;
            for (integer i=0; i<POW_DIM; i++) {
                if (tree->child[i] != -1) {
                    jj++;
                }
            }

            integer counter = threadIdx.x % warp;
            integer stackStartIndex = stackSize*(threadIdx.x / warp);

            while ((bodyIndex + offset) < n) {

                real pos_x = particles->x[bodyIndex + offset];
#if DIM > 1
                real pos_y = particles->y[bodyIndex + offset];
#if DIM == 3
                real pos_z = particles->z[bodyIndex + offset];
#endif
#endif

                real acc_x = 0.0;
#if DIM > 1
                real acc_y = 0.0;
#if DIM == 3
                real acc_z = 0.0;
#endif
#endif

                // initialize stack
                integer top = jj + stackStartIndex;

                if (counter == 0) {

                    integer temp = 0;

                    for (int i=0; i<POW_DIM; i++) {
                        if (tree->child[i] != -1) {
                            stack[stackStartIndex + temp] = tree->child[i];
                            depth[stackStartIndex + temp] = radius*radius/theta;
                            temp++;
                        }
                    }
                }
                __syncthreads();

                // while stack is not empty / more nodes to visit
                while (top >= stackStartIndex) {

                    integer node = stack[top];

                    real dp = 0.5 * depth[top]; //powf(0.5, DIM) * depth[top]; //0.25*depth[top]; // float dp = depth[top];

                    for (integer i=0; i<POW_DIM; i++) {

                        integer ch = tree->child[POW_DIM*node + i];

                        //__threadfence();

                        if (ch >= 0) {

                            real dx = particles->x[ch] - pos_x;
#if DIM > 1
                            real dy = particles->y[ch] - pos_y;
#if DIM == 3
                            real dz = particles->z[ch] - pos_z;
#endif
#endif

                            real r = dx*dx + smoothing; // SMOOTHING
#if DIM > 1
                            r += dy*dy;
#if DIM == 3
                            r += dz*dz;
#endif
#endif

                            //if (ch < n /*is leaf node*/ || !__any_sync(activeMask, dp > r)) {
                            if (ch < m /*is leaf node*/ || __all_sync(__activemask(), dp <= r)) {

                                // calculate interaction force contribution
                                if (r > 0.f) { //NEW //TODO: how to avoid r = 0?
                                    r = cuda::math::rsqrt(r);
                                }

#if SI_UNITS
                                real f = Constants::G * particles->mass[ch] * r * r * r;
#else
                                real f = particles->mass[ch] * r * r * r;
#endif

                                acc_x += f*dx;
#if DIM > 1
                                acc_y += f*dy;
#if DIM == 3
                                acc_z += f*dz;
#endif
#endif
                            }
                            else {
                                // if first thread in warp: push node's children onto iteration stack
                                if (counter == 0) {
                                    stack[top] = ch;
                                    depth[top] = dp; // depth[top] = 0.25*dp;
                                }
                                top++; // descend to next tree level
                                //__threadfence();
                            }
                        }
                        else {
                            /*top = max(stackStartIndex, top-1); */
                        }
                    }
                    top--;
                }
                // update body data
                particles->g_ax[bodyIndex + offset] = acc_x;
#if DIM > 1
                particles->g_ay[bodyIndex + offset] = acc_y;
#if DIM == 3
                particles->g_az[bodyIndex + offset] = acc_z;
#endif
#endif
                offset += stride;

                __syncthreads();
            }

        }

        __global__ void createKeyHistRanges(Helper *helper, integer bins) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            keyType max_key = 1UL << (DIM * 21);//1UL << 63;

            while ((bodyIndex + offset) < bins) {

                helper->keyTypeBuffer[bodyIndex + offset] = (bodyIndex + offset) * (max_key/bins);
                //printf("keyHistRanges[%i] = %lu\n", bodyIndex + offset, keyHistRanges[bodyIndex + offset]);

                if ((bodyIndex + offset) == (bins - 1)) {
                    helper->keyTypeBuffer[bins-1] = KEY_MAX;
                }
                offset += stride;
            }
        }

        __global__ void intermediateSymbolicForce(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                  DomainList *domainList, integer *sendIndices, real diam, real theta_,
                                                  integer n, integer m, integer relevantIndex, integer level,
                                                  Curve::Type curveType) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            while ((bodyIndex + offset) < *tree->index) {
                if (sendIndices[bodyIndex + offset] == 2) {
                    sendIndices[bodyIndex + offset] = 0;
                }
                if (sendIndices[bodyIndex + offset] == 3) {
                    sendIndices[bodyIndex + offset] = 1;
                }

                offset += stride;
            }
        }

        __global__ void symbolicForce(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                      DomainList *domainList, integer *sendIndices, real diam, real theta_,
                                      integer n, integer m, integer relevantIndex, integer level,
                                      Curve::Type curveType) {

            int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;
            int offset = 0;

            int particleLevel;
            int domainListLevel;
            int currentDomainListIndex;
            int currentParticleIndex;
            int childIndex;

            int childPath;
            int tempChildIndex;

            bool isDomainListNode;
            bool insert;

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
            real r;

            // IDEA: sendIndices = [-1, -1, -1, ..., -1, -1]
            // mark to be tested indices with 2's: e.g.: sendIndices = [-1, 2, -1, ..., 2, -1]
            // the 2's are converted within a separate kernel to zeros (which will be tested within this kernel)
            //  separate kernel necessary to avoid race conditions
            // mark to be sent indices/particles with 3's: e.g.: sendIndices = [-1, 0, 3, ..., 3, 3]
            //  the 3's are converted within a separate kernel to ones

            if (level == 0) { // mark first level children as starting point
                while ((bodyIndex + offset) < POW_DIM) {
                    //if ((bodyIndex + offset) == 0) {
                    //    printf("symbolicForce: [rank %i] relevantDomainListIndices[%i] = %i (%f, %f, %f)\n",
                    //           subDomainKeyTree->rank,
                    //           relevantIndex, domainList->relevantDomainListIndices[relevantIndex],
                    //           particles->x[domainList->relevantDomainListIndices[relevantIndex]],
                    //           particles->y[domainList->relevantDomainListIndices[relevantIndex]],
                    //           particles->z[domainList->relevantDomainListIndices[relevantIndex]]);
                    //}
                    if (tree->child[bodyIndex + offset] != -1) {
                        sendIndices[tree->child[bodyIndex + offset]] = 0;
                    }
                    offset += stride;
                }
            }
            else {

                while ((bodyIndex + offset) < *tree->index) {

                    //if (bodyIndex + offset == 0) {
                    //    printf("[rank %i] relevantIndex = %i domainListIndex = %i (%f, %f, %f) %f\n", subDomainKeyTree->rank,
                    //           relevantIndex, domainList->relevantDomainListIndices[relevantIndex],
                    //           particles->x[domainList->relevantDomainListIndices[relevantIndex]],
                    //           particles->y[domainList->relevantDomainListIndices[relevantIndex]],
                    //           particles->z[domainList->relevantDomainListIndices[relevantIndex]],
                    //           particles->mass[domainList->relevantDomainListIndices[relevantIndex]]);
                    //}

                    currentParticleIndex = bodyIndex + offset;

                    if ((sendIndices[currentParticleIndex] == 0 || sendIndices[currentParticleIndex] == 3) && (currentParticleIndex < n || currentParticleIndex >= m )) {

                        insert = true;
                        isDomainListNode = false;

                        if (sendIndices[currentParticleIndex] == 0) {

                            // check whether to be inserted index corresponds to a domain list
                            //if (insert) {
                                //for (int i_domain = 0; i_domain < *domainList->domainListIndex; i_domain++) {
                                //    if ((bodyIndex + offset) == domainList->domainListIndices[i_domain]) {
                                //        insert = false;
                                //        isDomainListNode = true;
                                //        break;
                                //    }
                                //}
                            if (particles->nodeType[bodyIndex + offset] >= 1) {
                                insert = false;
                                isDomainListNode = true;
                            }
                            //}
                            // TODO: this is probably not necessary, since only domain list indices can correspond to another process
                            if (!isDomainListNode) {
                                if (subDomainKeyTree->key2proc(
                                        tree->getParticleKey(particles, currentParticleIndex, MAX_LEVEL, curveType)) !=
                                    subDomainKeyTree->rank) {
                                    insert = false;
                                    //printf("Happening?\n");
                                }
                            }

                            if (insert) {
                                sendIndices[currentParticleIndex] = 3;
                            } else {
                                sendIndices[currentParticleIndex] = -1;
                            }
                        }

                        // get the particle's level
                        //particleLevel /*int tempParticleLevel*/ = tree->getTreeLevel(particles, currentParticleIndex, MAX_LEVEL, curveType);
                        particleLevel = particles->level[currentParticleIndex];
                        //if (tempParticleLevel != particleLevel) {
                        //    printf("%i vs %i\n", tempParticleLevel, particleLevel);
                        //}
#if DEBUGGING
#if DIM == 3
                        if (particleLevel < 0) {
                            printf("WTF particleLevel = %i for %i (%e, %e, %e) (numParticlesLocal = %i, index = %i)\n",
                                   particleLevel, currentParticleIndex, particles->x[currentParticleIndex],
                                   particles->y[currentParticleIndex], particles->z[currentParticleIndex],
                                   n, *tree->index);
                        }
#endif
#endif

                        // get the domain list node's level
                        //domainListLevel = tree->getTreeLevel(particles,
                        //                                     domainList->relevantDomainListIndices[relevantIndex],
                        //                                     MAX_LEVEL, curveType);
                        domainListLevel = domainList->relevantDomainListLevels[relevantIndex];
                        currentDomainListIndex = domainList->relevantDomainListIndices[relevantIndex];
                        //printf("domainListLevel = %i\n", domainListLevel);
                        if (domainListLevel == -1) {
                            cudaAssert("symbolicForce(): domainListLevel == -1 for (relevant) index: %i\n",
                                       relevantIndex);
                        }

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

                        // determine domain list node's bounding box (in order to determine the distance)
                        //if (domainListLevel != 1) {
                        //    printf("domainListLevel = %i\n", domainListLevel);
                        //    assert(0);
                        //}
                        for (int j = 0; j < domainListLevel; j++) {

                            /*
#if DIM == 3
                            if (particles->x[domainList->relevantDomainListIndices[relevantIndex]] <= max_x && particles->x[domainList->relevantDomainListIndices[relevantIndex]] >= min_x &&
                                particles->y[domainList->relevantDomainListIndices[relevantIndex]] <= max_y && particles->y[domainList->relevantDomainListIndices[relevantIndex]] >= min_y &&
                                particles->z[domainList->relevantDomainListIndices[relevantIndex]] <= max_z && particles->z[domainList->relevantDomainListIndices[relevantIndex]] >= min_z) {

                            }
                            else {
                                printf("not within box %i, %i  level: %i (%f, %f, %f) box (%f, %f), (%f, %f), (%f, %f)!\n", relevantIndex, domainList->relevantDomainListIndices[relevantIndex],
                                       domainList->relevantDomainListLevels[relevantIndex],
                                       particles->x[domainList->relevantDomainListIndices[relevantIndex]], particles->y[domainList->relevantDomainListIndices[relevantIndex]],
                                       particles->z[domainList->relevantDomainListIndices[relevantIndex]],
                                       min_x, max_x, min_y, max_y, min_z, max_z);
                                assert(0);
                            }
#endif
                            */

                            childPath = 0;
                            if (particles->x[currentDomainListIndex] < 0.5 * (min_x + max_x)) {
                                childPath += 1;
                                max_x = 0.5 * (min_x + max_x);
                            } else {
                                min_x = 0.5 * (min_x + max_x);
                            }
#if DIM > 1
                            if (particles->y[currentDomainListIndex] < 0.5 * (min_y + max_y)) {
                                childPath += 2;
                                max_y = 0.5 * (min_y + max_y);
                            } else {
                                min_y = 0.5 * (min_y + max_y);
                            }
#if DIM == 3
                            if (particles->z[currentDomainListIndex] < 0.5 * (min_z + max_z)) {
                                childPath += 4;
                                max_z = 0.5 * (min_z + max_z);
                            } else {
                                min_z = 0.5 * (min_z + max_z);
                            }
#endif
#endif
                        }

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
                        } else if (particles->z[currentParticleIndex] > max_z) { dz =
                        particles->z[currentParticleIndex] - max_z;
                        } else {
                            dz = 0.;
                        }

#endif
#endif

#if DIM == 1
                        r = cuda::math::sqrt(dx*dx);
#elif DIM == 2
                        r = cuda::math::sqrt(dx*dx + dy*dy);
#else
                        r = cuda::math::sqrt(dx*dx + dy*dy + dz*dz);
#endif

                        // TODO: depending on gravity force version and amount of processes: 2 * diam or 1 * diam (why?)
                        //printf("%f >= %f (particleLevel = %i, theta = %f, r = %f)\n", powf(0.5, particleLevel-1) /* * 2*/ * diam, (theta_ * r), particleLevel, theta_, r);
                        if (particleLevel != -1 && (((powf(0.5, particleLevel-1) /* * 2*/ * diam) >= (theta_ * r)) || isDomainListNode)) {

                            #pragma unroll
                            for (int i = 0; i < POW_DIM; i++) {

                                //if (sendIndices[tree->child[POW_DIM * (bodyIndex + offset) + i]] != 1 && tree->child[POW_DIM * (bodyIndex + offset) + i] != -1) {
                                //    sendIndices[tree->child[POW_DIM * (bodyIndex + offset) + i]] = 2;
                                //}

                                //if (insert && tree->child[POW_DIM * (bodyIndex + offset) + i] != -1 && particles->x[tree->child[POW_DIM * (bodyIndex + offset) + i]] == particles->x[bodyIndex + offset] &&
                                //        particles->y[tree->child[POW_DIM * (bodyIndex + offset) + i]] == particles->y[bodyIndex + offset]) {
                                    //printf("[rank %i] index = %i == child = %i ^= %i (%f, %f, %f) vs (%f, %f, %f)\n", subDomainKeyTree->rank, bodyIndex + offset, i, tree->child[POW_DIM * (bodyIndex + offset) + i],
                                    //       particles->x[bodyIndex + offset], particles->y[bodyIndex + offset], particles->z[bodyIndex + offset],
                                    //
                                    //       particles->x[tree->child[POW_DIM * (bodyIndex + offset) + i]], particles->y[tree->child[POW_DIM * (bodyIndex + offset) + i]], particles->z[tree->child[POW_DIM * (bodyIndex + offset) + i]]);
                                //}

                                if (tree->child[POW_DIM * currentParticleIndex + i] != -1) {
                                    if (sendIndices[tree->child[POW_DIM * currentParticleIndex + i]] != 1) {
                                        sendIndices[tree->child[POW_DIM * currentParticleIndex + i]] = 2;
                                    }
                                }

                            }
                        }
                    }

                    __threadfence();
                    offset += stride;
                }

            }

        }

        __global__ void compTheta(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                  DomainList *domainList, Helper *helper, Curve::Type curveType) {

            integer index = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;
            integer bodyIndex;
            keyType key, hilbert;
            integer domainIndex;
            integer proc;

            //"loop" over domain list nodes
            while ((index + offset) < *domainList->domainListIndex) {

                bodyIndex = domainList->domainListIndices[index + offset];
                //calculate key
                //TODO: why not
                //key =  domainList->domainListKeys[index + offset]; //???
                //hilbert = KeyNS::lebesgue2hilbert(key, 21);
                key = tree->getParticleKey(particles, bodyIndex, MAX_LEVEL, curveType); // working version
                //if domain list node belongs to other process: add to relevant domain list indices
                proc = subDomainKeyTree->key2proc(key);

                //printf("[rank %i] potential relevant domain list node: %i (%f, %f, %f)\n", subDomainKeyTree->rank,
                //       bodyIndex, particles->x[bodyIndex],
                //       particles->y[bodyIndex], particles->z[bodyIndex]);

                if (proc < 0 && particles->mass[bodyIndex] <= 0.f) {
                    printf("proc = %i, mass = %e\n", proc, particles->mass[bodyIndex]);
                    //assert(0);
                }
                if (proc != subDomainKeyTree->rank && proc >= 0 && particles->mass[bodyIndex] > 0.f) {
                    //printf("[rank = %i] proc = %i, key = %lu for x = (%f, %f, %f)\n", subDomainKeyTree->rank, proc, key, particles->x[bodyIndex], particles->y[bodyIndex], particles->z[bodyIndex]);
                    domainIndex = atomicAdd(domainList->domainListCounter, 1);
                    domainList->relevantDomainListIndices[domainIndex] = bodyIndex;
                    domainList->relevantDomainListLevels[domainIndex] = domainList->domainListLevels[index + offset];
                    domainList->relevantDomainListProcess[domainIndex] = proc;

                    //printf("[rank %i] Adding relevant domain list node: %i (%f, %f, %f)\n", subDomainKeyTree->rank,
                    //       bodyIndex, particles->x[bodyIndex],
                    //       particles->y[bodyIndex], particles->z[bodyIndex]);
                }
                offset += stride;
            }

        }

        __global__ void keyHistCounter(Tree *tree, Particles *particles, SubDomainKeyTree *subDomainKeyTree,
                                       Helper *helper,
                                       /*keyType *keyHistRanges, integer *keyHistCounts,*/ int bins, int n,
                                       Curve::Type curveType) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            keyType key;

            while ((bodyIndex + offset) < n) {

                key = tree->getParticleKey(particles, bodyIndex + offset, MAX_LEVEL, curveType);

                for (int i = 0; i < (bins); i++) {
                    if (key >= helper->keyTypeBuffer[i] && key < helper->keyTypeBuffer[i + 1]) {
                        //keyHistCounts[i] += 1;
                        atomicAdd(&helper->integerBuffer[i], 1);
                        break;
                    }
                }

                offset += stride;
            }

        }

        //TODO: resetting helper (buffers)?!
        __global__ void calculateNewRange(SubDomainKeyTree *subDomainKeyTree, Helper *helper,
                                          /*keyType *keyHistRanges, integer *keyHistCounts,*/ int bins, int n,
                                          Curve::Type curveType) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            integer sum;
            keyType newRange;

            while ((bodyIndex + offset) < (bins-1)) {

                sum = 0;
                for (integer i=0; i<(bodyIndex+offset); i++) {
                    sum += helper->integerBuffer[i];
                }

                for (integer i=1; i<subDomainKeyTree->numProcesses; i++) {
                    if ((sum + helper->integerBuffer[bodyIndex + offset]) >= (i*n) && sum < (i*n)) {
                        printf("[rank %i] new range: %lu\n", subDomainKeyTree->rank,
                               helper->keyTypeBuffer[bodyIndex + offset]);
                        subDomainKeyTree->range[i] = (helper->keyTypeBuffer[bodyIndex + offset] >> (1*DIM)) << (1*DIM);
                    }
                }


                //printf("[rank %i] keyHistCounts[%i] = %i\n", s->rank, bodyIndex+offset, keyHistCounts[bodyIndex+offset]);
                atomicAdd(helper->integerVal, helper->integerBuffer[bodyIndex+offset]);
                offset += stride;
            }

        }

        __global__ void insertReceivedPseudoParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                      integer *levels, int level, int n, int m) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset;

            integer childPath;
            integer temp;

            integer insertionLevel;

            real min_x, max_x;
#if DIM > 1
            real min_y, max_y;
#if DIM == 3
            real min_z, max_z;
#endif
#endif

            // debug
            //if (bodyIndex == 0 && level == 0) {
            //    integer levelCounter;
            //    for (int debugLevel = 0; debugLevel< MAX_LEVEL; debugLevel++) {
            //        levelCounter = 0;
            //        for (int i = 0; i < (tree->toDeleteNode[1] - tree->toDeleteNode[0]); i++) {
            //            if (debugLevel == 0) {
            //                if (subDomainKeyTree->key2proc(tree->getParticleKey(particles, tree->toDeleteNode[0] + i, MAX_LEVEL, Curve::lebesgue)) == subDomainKeyTree->rank) {
            //                    printf("\n-------------------------------------------------\nATTENTION\n\n-------------------------------------------------\n");
            //                }
            //                //if (particles->x[tree->toDeleteNode[0] + i] == 0.f && particles->y[tree->toDeleteNode[0] + i] == 0.f &&
            //                //        particles->z[tree->toDeleteNode[0] + i] == 0.f) {
            //                //    printf("\n-------------------------------------------------\nATTENTION\n\n-------------------------------------------------\n");
            //                //}
            //              //printf("[rank %i] index = %i level = %i x = (%f, %f, %f) m = %f\n", subDomainKeyTree->rank,
            //                //       tree->toDeleteNode[0] + i,
            //                //       levels[i],
            //                //       particles->x[tree->toDeleteNode[0] + i],
            //                //       particles->y[tree->toDeleteNode[0] + i],
            //                //       particles->z[tree->toDeleteNode[0] + i],
            //                //       particles->mass[tree->toDeleteNode[0] + i]);
            //            }
            //            if (levels[i] == debugLevel) {
            //                //printf("[rank %i] level available: %i\n", subDomainKeyTree->rank, debugLevel);
            //                levelCounter++;
            //            }
            //        }
            //        if (levelCounter > 0) {
            //            printf("[rank %i] level available: %i (# = %i)\n", subDomainKeyTree->rank, debugLevel, levelCounter);
            //        }
            //    }
            //}

            offset = 0;
            while ((bodyIndex + offset) < (tree->toDeleteNode[1] - tree->toDeleteNode[0])) {

                insertionLevel = 0;

                //if (levels[bodyIndex + offset] < 0 || levels[bodyIndex + offset] > 21) {
                //    printf("[rank %i] levels[%i] = %i!\n", subDomainKeyTree->rank, bodyIndex + offset, levels[bodyIndex + offset]);
                //    assert(0);
                //}

                if (levels[bodyIndex + offset] == level) {

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
                    if (particles->x[tree->toDeleteNode[0] + bodyIndex + offset] < 0.5 * (min_x + max_x)) { // x direction
                        childPath += 1;
                        max_x = 0.5 * (min_x + max_x);
                    }
                    else {
                        min_x = 0.5 * (min_x + max_x);
                    }
#if DIM > 1
                    if (particles->y[tree->toDeleteNode[0] + bodyIndex + offset] < 0.5 * (min_y + max_y)) { // y direction
                        childPath += 2;
                        max_y = 0.5 * (min_y + max_y);
                    }
                    else {
                        min_y = 0.5 * (min_y + max_y);
                    }
#if DIM == 3
                    if (particles->z[tree->toDeleteNode[0] + bodyIndex + offset] < 0.5 * (min_z + max_z)) {  // z direction
                        childPath += 4;
                        max_z = 0.5 * (min_z + max_z);
                    }
                    else {
                        min_z = 0.5 * (min_z + max_z);
                    }
#endif
#endif
                    int childIndex = tree->child[temp*POW_DIM + childPath];
                    atomicAdd(&tree->count[childIndex], 1);
                    insertionLevel++;

                    // debug
                    //if (subDomainKeyTree->rank == 0) {
                    //    if (childPath < 4) {
                    //        printf("[rank %i] childPath = %i WTF?\n", subDomainKeyTree->rank, childPath);
                    //    }
                    //}
                    //else {
                    //    if (childPath >= 4) {
                    //        printf("[rank %i] childPath = %i WTF?\n", subDomainKeyTree->rank, childPath);
                    //    }
                    //}
                    // end: debug

                    // debug
                    //if ((bodyIndex + offset) % 100 == 0) {
                    //    printf("[rank %i] childPath = %i, childIndex = %i\n", subDomainKeyTree->rank, childPath,
                    //           childIndex);
                    //}
                    // end: debug

                    // traverse tree until hitting leaf node
                    while (childIndex >= m) {
                        insertionLevel++;

                        temp = childIndex;
                        childPath = 0;

                        // find insertion point for body
                        if (particles->x[tree->toDeleteNode[0] + bodyIndex + offset] < 0.5 * (min_x + max_x)) { // x direction
                            childPath += 1;
                            max_x = 0.5 * (min_x + max_x);
                        }
                        else {
                            min_x = 0.5 * (min_x + max_x);
                        }
#if DIM > 1
                        if (particles->y[tree->toDeleteNode[0] + bodyIndex + offset] < 0.5 * (min_y + max_y)) { // y direction
                            childPath += 2;
                            max_y = 0.5 * (min_y + max_y);
                        }
                        else {
                            min_y = 0.5 * (min_y + max_y);
                        }
#if DIM == 3
                        if (particles->z[tree->toDeleteNode[0] + bodyIndex + offset] < 0.5 * (min_z + max_z)) { // z direction
                            childPath += 4;
                            max_z = 0.5 * (min_z + max_z);
                        }
                        else {
                            min_z = 0.5 * (min_z + max_z);
                        }
#endif
#endif
                        atomicAdd(&tree->count[temp], 1); // ? do not count, since particles are just temporarily saved on this process
                        childIndex = tree->child[POW_DIM*temp + childPath];

                    }
#if DIM == 3
                    if (childIndex != -1) {
                        printf("[rank %i] (%f, %f, %f) vs (%f, %f, %f)\n", subDomainKeyTree->rank,
                               particles->x[tree->toDeleteNode[0] + bodyIndex + offset],
                               particles->y[tree->toDeleteNode[0] + bodyIndex + offset],
                               particles->z[tree->toDeleteNode[0] + bodyIndex + offset],
                               particles->x[childIndex],
                               particles->y[childIndex],
                               particles->z[childIndex]);
                        cudaAssert("insertReceivedPseudoParticles(): childIndex = %i temp = %i\n", childIndex, temp);
                    }
#endif

                    //insertionLevel++;

                    //temp = childIndex;
                    tree->child[POW_DIM*temp + childPath] = tree->toDeleteNode[0] + bodyIndex + offset;
                    //printf("[rank %i] gravity inserting POWDIM * %i + %i = %i (level = %i)\n", subDomainKeyTree->rank,
                    //       temp, childPath, tree->toDeleteNode[0] + bodyIndex + offset, level);

                    if (levels[bodyIndex + offset] != insertionLevel) {
                        // debug
                        //printf("[rank %i] index = %i childIndex = %i level = %i insertionLevel = %i path = %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i\n",
                        //       subDomainKeyTree->rank, tree->toDeleteNode[0] + bodyIndex + offset, childIndex,
                        //       levels[bodyIndex + offset], insertionLevel, path[0], path[1], path[2], path[3], path[4],
                        //       path[5], path[6], path[7], path[8], path[9], path[10]);
                        //printf("[rank %i] level = %i, insertionLevel = %i x = (%f, %f, %f) min/max = (%f, %f | %f, %f | %f, %f))\n", subDomainKeyTree->rank,
                        //       levels[bodyIndex + offset], insertionLevel,
                        //       particles->x[tree->toDeleteNode[0] + bodyIndex + offset],
                        //       particles->y[tree->toDeleteNode[0] + bodyIndex + offset],
                        //       particles->z[tree->toDeleteNode[0] + bodyIndex + offset],
                        //       min_x, max_x, min_y, max_y, min_z, max_z);
                        //printf("[rank %i] level = %i, insertionLevel = %i x = (%f, %f, %f) min/max = (%f, %f, %f))\n", subDomainKeyTree->rank,
                        //       levels[bodyIndex + offset], insertionLevel,
                        //       particles->x[tree->toDeleteNode[0] + bodyIndex + offset],
                        //       particles->y[tree->toDeleteNode[0] + bodyIndex + offset],
                        //       particles->z[tree->toDeleteNode[0] + bodyIndex + offset],
                        //       0.5 * (min_x + max_x), 0.5 * (min_y + max_y), 0.5 * (min_z + max_z));
                        //for (int i=0; i < (tree->toDeleteNode[1] - tree->toDeleteNode[0]); i++) {
                        //    printf("[rank %i] index = %i level = %i x = (%f, %f, %f) m = %f\n",
                        //            subDomainKeyTree->rank,
                        //            tree->toDeleteNode[0] + i,
                        //            levels[i],
                        //            particles->x[tree->toDeleteNode[0] + i],
                        //            particles->y[tree->toDeleteNode[0] + i],
                        //            particles->z[tree->toDeleteNode[0] + i],
                        //            particles->mass[tree->toDeleteNode[0] + i]);
                        //}

                        printf("insertReceivedPseudoParticles() for %i: level[%i] = %i != insertionLevel = %i!\n",
                               tree->toDeleteNode[0] + bodyIndex + offset, bodyIndex + offset,
                               levels[bodyIndex + offset], insertionLevel);
                        assert(0);
                    }
                }
                __threadfence();
                offset += stride;
            }
        }

        __global__ void insertReceivedParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                DomainList *domainList, DomainList *lowestDomainList, int n, int m) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;

            integer offset;

            real min_x, max_x;
#if DIM > 1
            real min_y, max_y;
#if DIM == 3
            real min_z, max_z;
#endif
#endif

            integer childPath;
            integer temp;

            offset = 0;

            bodyIndex += tree->toDeleteLeaf[0];

            while ((bodyIndex + offset) < tree->toDeleteLeaf[1]) { // && (bodyIndex + offset) >= tree->toDeleteLeaf[0]) {

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

                int childIndex = tree->child[temp*POW_DIM + childPath];

                // traverse tree until hitting leaf node
                while (childIndex >= m) {

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
                    atomicAdd(&tree->count[temp], 1); // do not count, since particles are just temporarily saved on this process
                    childIndex = tree->child[POW_DIM*temp + childPath];

                }

                if (childIndex != -1) {
                    cudaAssert("ATTENTION: insertReceivedParticles(): childIndex = %i (%i, %i) (%i, %i)\n", childIndex,
                               tree->toDeleteLeaf[0], tree->toDeleteLeaf[1], tree->toDeleteNode[0],
                               tree->toDeleteNode[1]);
                    //printf("[rank %i] ATTENTION: childIndex = %i,... child[8 * %i + %i] = %i (%f, %f, %f) vs (%f, %f, %f)\n", subDomainKeyTree->rank,
                    //           childIndex, temp, childPath, bodyIndex + offset,
                    //           particles->x[childIndex], particles->y[childIndex], particles->z[childIndex],
                    //           particles->x[bodyIndex + offset], particles->y[bodyIndex + offset], particles->z[bodyIndex + offset]);

                }

                tree->child[POW_DIM*temp + childPath] = bodyIndex + offset;

                __threadfence();
                offset += stride;
            }

        }

        //TODO: do not delete children of lowest domain list nodes belonging to another process
        // but instead: delete the children of (lowest) domain list nodes if
        //  - toDeleteLeaf[0] < child < numParticles
        //  - child > toDeleteNode[0]
        // since problem for predictor-corrector decoupled gravity (having particles belonging to another process)
        __global__ void repairTree(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                   DomainList *domainList, DomainList *lowestDomainList,
                                   int n, int m, Curve::Type curveType) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            keyType key;
            int domainIndex;
            int proc;

            if (bodyIndex + offset == 0) {
                *tree->index = tree->toDeleteNode[0];
            }

            /*while ((bodyIndex + offset) < *lowestDomainList->domainListIndex) {
                domainIndex = lowestDomainList->domainListIndices[bodyIndex + offset];

                //key = tree->getParticleKey(particles, domainIndex, MAX_LEVEL, curveType); // working version
                //proc = subDomainKeyTree->key2proc(key);
                // //printf("[rank %i] deleting: proc = %i\n", subDomainKeyTree->rank, proc);
                //if (proc != subDomainKeyTree->rank) {
                //    for (int i=0; i<POW_DIM; i++) {
                //        //printf("[rank %i] deleting: POWDIM * %i + %i = %i\n", subDomainKeyTree->rank, domainIndex, i, tree->child[POW_DIM * domainIndex + i]);
                //        tree->child[POW_DIM * domainIndex + i] = -1;
                //    }
                //}

                for (int i=0; i<POW_DIM; i++) {
                    if (tree->child[POW_DIM * domainIndex + i] >= tree->toDeleteNode[0]) {
                        tree->child[POW_DIM * domainIndex + i] = -1;
                    }
                }
                offset += stride;
            }*/

            while ((bodyIndex + offset) < *domainList->domainListIndex) {
                domainIndex = domainList->domainListIndices[bodyIndex + offset];
                for (int i=0; i<POW_DIM; i++) {
                    if ((tree->child[POW_DIM * domainIndex + i] >= tree->toDeleteNode[0]
                    || (tree->child[POW_DIM * domainIndex + i] >= tree->toDeleteLeaf[0] &&
                    tree->child[POW_DIM * domainIndex + i] < n))
                    && particles->nodeType[tree->child[POW_DIM * domainIndex + i]] < 1) {

                        tree->child[POW_DIM * domainIndex + i] = -1;

                    }
                }
                offset += stride;
            }

            offset = tree->toDeleteLeaf[0];
            //delete inserted leaves
            while ((bodyIndex + offset) >= tree->toDeleteLeaf[0] && (bodyIndex + offset) < tree->toDeleteLeaf[1]) {
                for (int i=0; i<POW_DIM; i++) {
                    tree->child[(bodyIndex + offset)*POW_DIM + i] = -1;
                }
                tree->count[bodyIndex + offset] = 1;

                particles->x[bodyIndex + offset] = 0.;
                particles->vx[bodyIndex + offset] = 0.;
                particles->ax[bodyIndex + offset] = 0.;
                particles->g_ax[bodyIndex + offset] = 0.;
#if DIM > 1
                particles->y[bodyIndex + offset] = 0.;
                particles->vy[bodyIndex + offset] = 0.;
                particles->ay[bodyIndex + offset] = 0.;
                particles->g_ay[bodyIndex + offset] = 0.;
#if DIM == 3
                particles->z[bodyIndex + offset] = 0.;
                particles->vz[bodyIndex + offset] = 0.;
                particles->az[bodyIndex + offset] = 0.;
                particles->g_az[bodyIndex + offset] = 0.;
#endif
#endif
                particles->mass[bodyIndex + offset] = 0.;
                tree->start[bodyIndex + offset] = -1;
                tree->sorted[bodyIndex + offset] = 0;

                offset += stride;
            }

            offset = tree->toDeleteNode[0]; //0;
            //delete inserted cells
            while ((bodyIndex + offset) >= tree->toDeleteNode[0] && (bodyIndex + offset) < tree->toDeleteNode[1]) {
                for (int i=0; i<POW_DIM; i++) {
                    tree->child[(bodyIndex + offset)*POW_DIM + i] = -1;
                }
                tree->count[bodyIndex + offset] = 0;
                particles->x[bodyIndex + offset] = 0.;
                particles->vx[bodyIndex + offset] = 0.;
                particles->ax[bodyIndex + offset] = 0.;
#if DIM > 1
                particles->y[bodyIndex + offset] = 0.;
                particles->vy[bodyIndex + offset] = 0.;
                particles->ay[bodyIndex + offset] = 0.;
#if DIM == 3
                particles->z[bodyIndex + offset] = 0.;
                particles->vz[bodyIndex + offset] = 0.;
                particles->az[bodyIndex + offset] = 0.;
#endif
#endif
                particles->mass[bodyIndex + offset] = 0.;
                tree->start[bodyIndex + offset] = -1;
                tree->sorted[bodyIndex + offset] = 0;

                offset += stride;
            }
        }

        real Launch::globalCOM(Tree *tree, Particles *particles, real com[DIM]) {
            ExecutionPolicy executionPolicy(1, 1);
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::globalCOM, tree, particles, com);
        }

        real Launch::collectSendIndices(Tree *tree, Particles *particles, integer *sendIndices,
                                integer *particles2Send, integer *pseudoParticles2Send,
                                integer *pseudoParticlesLevel,
                                integer *particlesCount, integer *pseudoParticlesCount,
                                integer n, integer length, Curve::Type curveType) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::collectSendIndices, tree, particles, sendIndices,
                                particles2Send, pseudoParticles2Send, pseudoParticlesLevel, particlesCount,
                                pseudoParticlesCount, n, length, curveType);
        }

        real Launch::testSendIndices(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                             integer *sendIndices, integer *markedSendIndices,
                             integer *levels, Curve::Type curveType, integer length) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::testSendIndices, subDomainKeyTree,
                                tree, particles, sendIndices, markedSendIndices, levels, curveType, length);
        }

        real Launch::zeroDomainListNodes(Particles *particles, DomainList *domainList, DomainList *lowestDomainList) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::zeroDomainListNodes, particles, domainList,
                                lowestDomainList);
        }

        real Launch::prepareLowestDomainExchange(Particles *particles, DomainList *lowestDomainList,
                                                 Helper *helper, Entry::Name entry) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::prepareLowestDomainExchange, particles,
                                lowestDomainList, helper, entry);
        }

        real Launch::updateLowestDomainListNodes(Particles *particles, DomainList *lowestDomainList,
                                                 Helper *helper, Entry::Name entry) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::updateLowestDomainListNodes, particles,
                                lowestDomainList, helper, entry);

        }

        real Launch::compLowestDomainListNodes(Tree *tree, Particles *particles, DomainList *lowestDomainList) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::compLowestDomainListNodes, tree, particles,
                                lowestDomainList);
        }

        real Launch::compLocalPseudoParticles(Tree *tree, Particles *particles, DomainList *domainList, int n) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::compLocalPseudoParticles, tree, particles,
                                domainList, n);
        }

        real Launch::compDomainListPseudoParticlesPerLevel(Tree *tree, Particles *particles, DomainList *domainList,
                                                   DomainList *lowestDomainList, int n, int level) {
            ExecutionPolicy executionPolicy(256, 1);
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::compDomainListPseudoParticlesPerLevel, tree,
                                particles, domainList, lowestDomainList, n, level);
        }

        real Launch::compDomainListPseudoParticles(Tree *tree, Particles *particles, DomainList *domainList,
                                                   DomainList *lowestDomainList, int n) {
            ExecutionPolicy executionPolicy(256, 1);
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::compDomainListPseudoParticles, tree,
                                particles, domainList, lowestDomainList, n);
        }

        real Launch::computeForces_v1(Tree *tree, Particles *particles, real radius, integer n, integer m,
                                      SubDomainKeyTree *subDomainKeyTree, real theta, real smoothing) {
            size_t sharedMemory = sizeof(real) * MAX_DEPTH;
            ExecutionPolicy executionPolicy(256, 256, sharedMemory);
            //ExecutionPolicy executionPolicy(512, 256, sharedMemory);
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::computeForces_v1, tree, particles,
                                radius, n, m, subDomainKeyTree, theta, smoothing);
        }

        real Launch::computeForces_v1_1(Tree *tree, Particles *particles, real radius, integer n, integer m,
                                        SubDomainKeyTree *subDomainKeyTree, real theta, real smoothing) {
            size_t sharedMemory = sizeof(real) * MAX_DEPTH;
            ExecutionPolicy executionPolicy(256, 256, sharedMemory);
            //ExecutionPolicy executionPolicy(512, 256, sharedMemory);
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::computeForces_v1_1, tree, particles,
                                radius, n, m, subDomainKeyTree, theta, smoothing);
        }

        real Launch::computeForces_v1_2(Tree *tree, Particles *particles, real radius, integer n, integer m,
                                      SubDomainKeyTree *subDomainKeyTree, real theta, real smoothing) {
            size_t sharedMemory = (2*sizeof(int) + sizeof(real)) * MAX_DEPTH;
            ExecutionPolicy executionPolicy(256, 256, sharedMemory);
            //ExecutionPolicy executionPolicy(512, 256, sharedMemory);
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::computeForces_v1_2, tree, particles,
                                radius, n, m, subDomainKeyTree, theta, smoothing);
        }

        real Launch::computeForces_v2(Tree *tree, Particles *particles, real radius, integer n, integer m,
                                      integer blockSize, integer warp, integer stackSize,
                                      SubDomainKeyTree *subDomainKeyTree, real theta,
                                      real smoothing) {

            size_t sharedMemory = (sizeof(real)+sizeof(integer))*stackSize*blockSize/warp;
            //size_t sharedMemory = 2*sizeof(real)*stackSize*blockSize/warp;
            ExecutionPolicy executionPolicy(256, 256, sharedMemory);
            //ExecutionPolicy executionPolicy(512, 256, sharedMemory);
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::computeForces_v2, tree, particles, radius,
                                n, m, blockSize, warp, stackSize, subDomainKeyTree, theta, smoothing);
        }

        real Launch::computeForces_v2_1(Tree *tree, Particles *particles, integer n, integer m, integer blockSize,
                                        integer warp, integer stackSize, SubDomainKeyTree *subDomainKeyTree,
                                        real theta, real smoothing) {

            size_t sharedMemory = (sizeof(real)+sizeof(integer))*stackSize*blockSize/warp;
            //size_t sharedMemory = 2*sizeof(real)*stackSize*blockSize/warp;
            ExecutionPolicy executionPolicy(256, 256, sharedMemory);
            //ExecutionPolicy executionPolicy(512, 256, sharedMemory);
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::computeForces_v2_1, tree, particles, n, m,
                                blockSize, warp, stackSize, subDomainKeyTree, theta, smoothing);
        }

        //real Launch::symbolicForce(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
        //                   DomainList *domainList, integer *sendIndices,
        //                   real diam, real theta_, integer n, integer m, integer relevantIndex,
        //                   Curve::Type curveType) {
        //    ExecutionPolicy executionPolicy(1, 256);
        //    return cuda::launch(true, executionPolicy, ::Gravity::Kernel::symbolicForce, subDomainKeyTree, tree,
        //                        particles, domainList, sendIndices, diam, theta_, n, m, relevantIndex, curveType);
        //}

        real Launch::intermediateSymbolicForce(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                   DomainList *domainList, integer *sendIndices, real diam, real theta_,
                                   integer n, integer m, integer relevantIndex, integer level,
                                   Curve::Type curveType) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::intermediateSymbolicForce, subDomainKeyTree, tree,
                                particles, domainList, sendIndices, diam, theta_, n, m, relevantIndex, level, curveType);
        }

        real Launch::symbolicForce(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                           DomainList *domainList, integer *sendIndices, real diam, real theta_,
                           integer n, integer m, integer relevantIndex, integer level,
                           Curve::Type curveType) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::symbolicForce, subDomainKeyTree, tree,
                                particles, domainList, sendIndices, diam, theta_, n, m, relevantIndex, level, curveType);
        }

        real Launch::compTheta(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                       DomainList *domainList, Helper *helper, Curve::Type curveType) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::compTheta, subDomainKeyTree, tree, particles,
                                domainList, helper, curveType);
        }

        real Launch::createKeyHistRanges(Helper *helper, integer bins) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::createKeyHistRanges, helper, bins);
        }

        real Launch::keyHistCounter(Tree *tree, Particles *particles, SubDomainKeyTree *subDomainKeyTree,
                            Helper *helper, int bins, int n, Curve::Type curveType) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::keyHistCounter, tree, particles,
                                subDomainKeyTree, helper, bins, n, curveType);
        }

        real Launch::calculateNewRange(SubDomainKeyTree *subDomainKeyTree, Helper *helper, int bins, int n,
                               Curve::Type curveType) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::calculateNewRange, subDomainKeyTree, helper,
                                bins, n, curveType);
        }

        //real Launch::insertReceivedPseudoParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
        //                                   integer *levels, int n, int m) {
        //    ExecutionPolicy executionPolicy(1, 256);
        //    return cuda::launch(true, executionPolicy, ::Gravity::Kernel::insertReceivedPseudoParticles,
        //                        subDomainKeyTree, tree, particles, levels, n, m);
        //}

        real Launch::insertReceivedPseudoParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                   integer *levels, int level, int n, int m) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::insertReceivedPseudoParticles,
                                subDomainKeyTree, tree, particles, levels, level, n, m);
        }

        real Launch::insertReceivedParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                     DomainList *domainList, DomainList *lowestDomainList, int n, int m) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::insertReceivedParticles, subDomainKeyTree,
                                tree, particles, domainList, lowestDomainList, n, m);
        }

        real Launch::repairTree(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                DomainList *domainList, DomainList *lowestDomainList,
                                int n, int m, Curve::Type curveType) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::repairTree, subDomainKeyTree, tree,
                                particles, domainList, lowestDomainList, n, m, curveType);
        }

    }
}
