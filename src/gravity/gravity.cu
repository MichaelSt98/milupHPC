#include "../../include/gravity/gravity.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

namespace Gravity {

    namespace Kernel {

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
                        pseudoParticlesLevel[pseudoParticleInsertIndex] = tree->getTreeLevel(particles,
                                                                                             bodyIndex + offset,
                                                                                             MAX_LEVEL, curveType);
                        // debug
                        if (pseudoParticlesLevel[pseudoParticleInsertIndex] == -1) {
                            printf("level = -1 within collectSendIndices for index: %i\n", bodyIndex + offset);
                        }
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
                for (int i=0; i<*lowestDomainList->domainListIndex-1; i++) {
                    if (domainIndex = lowestDomainList->domainListIndices[i]) {
                        zero = false;
                    }
                }

                if (zero) {
                    particles->x[domainIndex] = 0.f;
                    particles->y[domainIndex] = 0.f;
                    particles->z[domainIndex] = 0.f;

                    particles->mass[domainIndex] = 0.f;
                }

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
                        case Entry::x:
                            helper->realBuffer[bodyIndex + offset] = particles->x[lowestDomainIndex];
                            break;
#if DIM > 1
                        case Entry::y:
                            helper->realBuffer[bodyIndex + offset] = particles->y[lowestDomainIndex];
                            break;
#if DIM == 3
                        case Entry::z:
                            helper->realBuffer[bodyIndex + offset] = particles->z[lowestDomainIndex];
                            break;
#endif
#endif
                        case Entry::mass:
                            helper->realBuffer[bodyIndex + offset] = particles->mass[lowestDomainIndex];
                            break;
                        default:
                            helper->realBuffer[bodyIndex + offset] = particles->mass[lowestDomainIndex];
                            break;
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
                for (int i = 0; i < *lowestDomainList->domainListIndex; i++) {
                    if (lowestDomainList->sortedDomainListKeys[bodyIndex + offset] ==
                        lowestDomainList->domainListKeys[i]) {
                        originalIndex = i;
                    }
                }

                if (originalIndex == -1) {
                    printf("ATTENTION: originalIndex = -1 (index = %i)!\n",
                           lowestDomainList->sortedDomainListKeys[bodyIndex + offset]);
                }

                switch (entry) {
                    case Entry::x:
                        particles->x[lowestDomainList->domainListIndices[originalIndex]] =
                                helper->realBuffer[DOMAIN_LIST_SIZE + bodyIndex + offset];
                        break;
#if DIM > 1
                    case Entry::y:
                        particles->y[lowestDomainList->domainListIndices[originalIndex]] =
                                helper->realBuffer[DOMAIN_LIST_SIZE + bodyIndex + offset];
                        break;
#if DIM == 3
                    case Entry::z:
                        particles->z[lowestDomainList->domainListIndices[originalIndex]] =
                                helper->realBuffer[DOMAIN_LIST_SIZE + bodyIndex + offset];
                        break;
#endif
#endif
                    case Entry::mass:
                        particles->mass[lowestDomainList->domainListIndices[originalIndex]] =
                                helper->realBuffer[DOMAIN_LIST_SIZE + bodyIndex + offset];
                        break;
                    default:
                        printf("Entry not available!\n");
                        break;
                }

                offset += stride;
            }
        }

        __global__ void compLowestDomainListNodes(Tree *tree, Particles *particles, DomainList *lowestDomainList) {

            integer bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
            integer stride = blockDim.x*gridDim.x;
            integer offset = 0;
            integer lowestDomainIndex;

            while ((bodyIndex + offset) < *lowestDomainList->domainListIndex) {

                lowestDomainIndex = lowestDomainList->domainListIndices[bodyIndex + offset];

                if (particles->mass[lowestDomainIndex] != 0) {
                    particles->x[lowestDomainIndex] /= particles->mass[lowestDomainIndex];
#if DIM > 1
                    particles->y[lowestDomainIndex] /= particles->mass[lowestDomainIndex];
#if DIM == 3
                    particles->z[lowestDomainIndex] /= particles->mass[lowestDomainIndex];
#endif
#endif
                }

                //printf("lowestDomainIndex = %i (%f, %f, %f) %f\n", lowestDomainIndex, particles->x[lowestDomainIndex],
                //       particles->y[lowestDomainIndex], particles->z[lowestDomainIndex], particles->mass[lowestDomainIndex]);

                /*if (particles->z[lowestDomainIndex] > 250) {
                    for (int i = 0; i < POW_DIM; i++) {
                        printf("out of box: %i += %i n(%f, %f, %f) %f\n",
                               lowestDomainIndex,
                               tree->child[POW_DIM * lowestDomainIndex + i],
                               particles->x[tree->child[POW_DIM * lowestDomainIndex + i]],
                               particles->y[tree->child[POW_DIM * lowestDomainIndex + i]],
                               particles->z[tree->child[POW_DIM * lowestDomainIndex + i]],
                               particles->mass[tree->child[POW_DIM * lowestDomainIndex + i]]);
                    }
                }*/

                offset += stride;
            }
        }

        __global__ void compLocalPseudoParticles(Tree *tree, Particles *particles, DomainList *domainList, int n) {
            integer bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
            integer stride = blockDim.x*gridDim.x;
            integer offset = 0;
            bool isDomainList;
            //note: most of it already done within buildTreeKernel

            bodyIndex += n;

            while (bodyIndex + offset < *tree->index) {
                isDomainList = false;

                for (integer i=0; i<*domainList->domainListIndex; i++) {
                    if ((bodyIndex + offset) == domainList->domainListIndices[i]) {
                        isDomainList = true; // hence do not insert
                        break;
                    }
                }

                if (particles->mass[bodyIndex + offset] != 0 && !isDomainList) {
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

            offset = 0;
            compute = true;
            while ((bodyIndex + offset) < *domainList->domainListIndex) {
                compute = true;
                domainIndex = domainList->domainListIndices[bodyIndex + offset];
                for (int i=0; i<*lowestDomainList->domainListIndex; i++) {
                    if (domainIndex == lowestDomainList->domainListIndices[i]) {
                        compute = false;
                    }
                }
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
                    /*if (particles->x[domainIndex] > 10.) {
                        printf("out of box: %i (%f, %f, %f) %f compute = %i\n", domainIndex, particles->x[domainIndex], particles->y[domainIndex],
                               particles->z[domainIndex], particles->mass[domainIndex], compute);
                        for (int i=0; i<POW_DIM; i++) {
                            printf("out of box: %i += %i n(%f, %f, %f) %f\n", domainIndex,
                                   tree->child[POW_DIM*domainIndex + i],
                                   particles->x[tree->child[POW_DIM*domainIndex + i]],
                                   particles->y[tree->child[POW_DIM*domainIndex + i]],
                                   particles->z[tree->child[POW_DIM*domainIndex + i]],
                                   particles->mass[tree->child[POW_DIM*domainIndex + i]]);
                        }
                    }*/
                }
                offset += stride;
            }
            //__syncthreads();

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
                    for (int i=0; i<*lowestDomainList->domainListIndex; i++) {
                        if (domainIndex == lowestDomainList->domainListIndices[i]) {
                            compute = false;
                        }
                    }
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
                        /*if (particles->z[domainIndex] > 250.) {
                            printf("out of box: %i (%f, %f, %f) %f compute = %i\n", domainIndex, particles->x[domainIndex], particles->y[domainIndex],
                                   particles->z[domainIndex], particles->mass[domainIndex], compute);
                            for (int i=0; i<POW_DIM; i++) {
                                printf("out of box: %i += %i n(%f, %f, %f) %f\n", domainIndex,
                                       tree->child[POW_DIM*domainIndex + i],
                                       particles->x[tree->child[POW_DIM*domainIndex + i]],
                                       particles->y[tree->child[POW_DIM*domainIndex + i]],
                                       particles->z[tree->child[POW_DIM*domainIndex + i]],
                                       particles->mass[tree->child[POW_DIM*domainIndex + i]]);
                            }
                        }*/
                    }
                    offset += stride;
                }
                __syncthreads();
                level--;
            }
        }

        __global__ void computeForces(Tree *tree, Particles *particles, integer n, integer m, integer blockSize,
                                      integer warp, integer stackSize, SubDomainKeyTree *subDomainKeyTree) {

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
            real radius = fmaxf(x_radius, y_radius);
#else
            real radius_max = fmaxf(x_radius, y_radius);
            real radius = fmaxf(radius_max, z_radius);
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

                integer sortedIndex = tree->sorted[bodyIndex + offset];

                real pos_x = particles->x[sortedIndex];
#if DIM > 1
                real pos_y = particles->y[sortedIndex];
#if DIM == 3
                real pos_z = particles->z[sortedIndex];
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

                            real r = dx*dx + 0.025; // SMOOTHING
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
                                    r = rsqrt(r);
                                }

                                real f = particles->mass[ch] * r * r * r;

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
                particles->ax[sortedIndex] = acc_x;
#if DIM > 1
                particles->ay[sortedIndex] = acc_y;
#if DIM == 3
                particles->az[sortedIndex] = acc_z;
#endif
#endif

                offset += stride;

                __syncthreads();
            }

        }

        __global__ void computeForcesUnsorted(Tree *tree, Particles *particles, integer n, integer m, integer blockSize,
                                      integer warp, integer stackSize, SubDomainKeyTree *subDomainKeyTree) {

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
            real radius = fmaxf(x_radius, y_radius);
#else
            real radius_max = fmaxf(x_radius, y_radius);
            real radius = fmaxf(radius_max, z_radius);
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

                            real r = dx*dx + 0.025; // SMOOTHING
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
                                    r = rsqrt(r);
                                }

                                real f = particles->mass[ch] * r * r * r;

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
                particles->ax[bodyIndex + offset] = acc_x;
#if DIM > 1
                particles->ay[bodyIndex + offset] = acc_y;
#if DIM == 3
                particles->az[bodyIndex + offset] = acc_z;
#endif
#endif
                offset += stride;

                __syncthreads();
            }

        }

        __global__ void computeForcesMiluphcuda(Tree *tree, Particles *particles, integer n, integer m,
                                                SubDomainKeyTree *subDomainKeyTree) {

            integer i, child, nodeIndex, childNumber, depth;
            real px, ax, dx, f, distance;
#if DIM > 1
            real py, ay, dy;
#endif
            integer currentNodeIndex[MAX_DEPTH];
            integer currentChildNumber[MAX_DEPTH];
#if DIM == 3
            real pz, az, dz;
#endif
            real sml;
            real thetasq = theta*theta;

            __shared__ volatile real cellsize[MAX_DEPTH];

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
            real radius = fmaxf(x_radius, y_radius);
#else
            real radius_max = fmaxf(x_radius, y_radius);
            real radius = fmaxf(radius_max, z_radius);
#endif

            if (0 == threadIdx.x) {
                cellsize[0] = 4.0 * radius * radius;
                for (i = 1; i < MAX_DEPTH; i++) {
                    cellsize[i] = cellsize[i - 1] * 0.25;
                }
            }

            __syncthreads();
            //__threadfence();

            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
                px = particles->x[i];
#if DIM > 1
                py = particles->y[i];
#if DIM == 3
                pz = particles->z[i];
#endif
#endif
                particles->ax[i] = 0.0;
#if DIM > 1
                particles->ay[i] = 0.0;
#endif
                ax = 0.0;
#if DIM > 1
                ay = 0.0;
#if DIM == 3
                az = 0.0;
                particles->az[i] = 0.0;
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
                        if (child != -1 && child != i) { // dont do selfgravity with yourself!
                            dx = particles->x[child] - px;
                            distance = dx*dx + 0.025;
#if DIM > 1
                            dy = particles->y[child] - py;
                            distance += dy*dy;
#endif
#if DIM == 3
                            dz = particles->z[child] - pz;
                            distance += dz*dz;
#endif
                            // if child is leaf or far away
                            if (child < n || distance * thetasq > cellsize[depth]) {
                                distance = sqrt(distance);
                                //distance += 1e10;
                                f = particles->mass[child] / (distance * distance * distance);

                                ax += f*dx;
#if DIM > 1
                                ay += f*dy;
#if DIM == 3
                                az += f*dz;
#endif
#endif
                            } else {
                                // put child on stack
                                currentChildNumber[depth] = childNumber;
                                currentNodeIndex[depth] = nodeIndex;
                                depth++;
                                if (depth == MAX_DEPTH) {
                                    printf("\n\nMAX_DEPTH reached in selfgravity... this is not good.\n\n");
                                    assert(depth < MAX_DEPTH);
                                }
                                childNumber = 0;
                                nodeIndex = child;
                            }
                        }
                    }
                    depth--;
                } while(depth > 0);

                particles->ax[i] = ax;
#if DIM > 1
                particles->ay[i] = ay;
#if DIM == 3
                particles->az[i] = az;
#endif
#endif
            }
        }

        __global__ void update(Particles *particles, integer n, real dt, real d) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            while (bodyIndex + offset < n) {

               // calculating/updating the velocities
                particles->vx[bodyIndex + offset] += dt * particles->ax[bodyIndex + offset];
#if DIM > 1
                particles->vy[bodyIndex + offset] += dt * particles->ay[bodyIndex + offset];
#if DIM == 3
                particles->vz[bodyIndex + offset] += dt * particles->az[bodyIndex + offset];
#endif
#endif

                // calculating/updating the positions
                particles->x[bodyIndex + offset] += d * dt * particles->vx[bodyIndex + offset];
#if DIM > 1
                particles->y[bodyIndex + offset] += d * dt * particles->vy[bodyIndex + offset];
#if DIM == 3
                particles->z[bodyIndex + offset] += d * dt * particles->vz[bodyIndex + offset];
#endif
#endif

                // debug
                //if (bodyIndex + offset == n - 1 || bodyIndex + offset == 0) {
                // //if ((bodyIndex + offset) % 100 == 0) {
                //    printf("update: %i (%f, %f, %f) x += (%f, %f, %f)\n", bodyIndex + offset, particles->x[bodyIndex + offset],
                //           particles->y[bodyIndex + offset], particles->z[bodyIndex + offset], d * dt * particles->vx[bodyIndex + offset],
                //           d * dt * particles->vy[bodyIndex + offset], d * dt * particles->vz[bodyIndex + offset]);
                //    printf("update: %i (%f, %f, %f) %f (%f, %f, %f) (%f, %f, %f) %f\n", bodyIndex + offset,
                //           particles->x[bodyIndex + offset],
                //           particles->y[bodyIndex + offset],
                //           particles->z[bodyIndex + offset],
                //           particles->mass[bodyIndex + offset],
                //           particles->vx[bodyIndex + offset],
                //           particles->vy[bodyIndex + offset],
                //           particles->vz[bodyIndex + offset],
                //           particles->ax[bodyIndex + offset],
                //           particles->ay[bodyIndex + offset],
                //           particles->az[bodyIndex + offset],
                //           particles->ax[bodyIndex + offset] * particles->ax[bodyIndex + offset] +
                //           particles->ay[bodyIndex + offset] * particles->ay[bodyIndex + offset] +
                //           particles->az[bodyIndex + offset] * particles->az[bodyIndex + offset]);
                //}
                //if (abs(particles->x[bodyIndex + offset]) < 3 && abs(particles->y[bodyIndex + offset]) < 3 &&
                //        abs(particles->z[bodyIndex + offset]) < 3) {
                //    printf("centered: index = %i (%f, %f, %f) %f\n", bodyIndex + offset,
                //           particles->x[bodyIndex + offset],
                //           particles->y[bodyIndex + offset],
                //           particles->z[bodyIndex + offset],
                //           particles->mass[bodyIndex + offset]);
                //    if (particles->mass[bodyIndex + offset] < 1) {
                //        //assert(0);
                //    }
                //}
                //if (abs(particles->ax[bodyIndex + offset]) < 10 && abs(particles->ay[bodyIndex + offset]) < 10 &&
                //    abs(particles->az[bodyIndex + offset]) < 10) {
                //if (true) {
                //    printf("ACCELERATION tiny! centered: index = %i (%f, %f, %f) %f (%f, %f, %f) (%f, %f, %f)\n", bodyIndex + offset,
                //           particles->x[bodyIndex + offset],
                //           particles->y[bodyIndex + offset],
                //           particles->z[bodyIndex + offset],
                //           particles->mass[bodyIndex + offset],
                //           particles->vx[bodyIndex + offset],
                //           particles->vy[bodyIndex + offset],
                //           particles->vz[bodyIndex + offset],
                //           particles->ax[bodyIndex + offset],
                //           particles->ay[bodyIndex + offset],
                //           particles->az[bodyIndex + offset]);
                //    if (particles->mass[bodyIndex + offset] < 1) {
                //        assert(0);
                //    }
                //}
                // end: debug

                offset += stride;
            }
        }

        __global__ void createKeyHistRanges(Helper *helper, integer bins) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            keyType max_key = 1UL << 63;

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

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            integer particleLevel;
            integer domainListLevel;
            integer childIndex;

            integer childPath;
            integer tempChildIndex;
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

                    if (bodyIndex + offset == 0) {
                        /*printf("[rank %i] relevantIndex = %i domainListIndex = %i (%f, %f, %f) %f\n", subDomainKeyTree->rank,
                               relevantIndex, domainList->relevantDomainListIndices[relevantIndex],
                               particles->x[domainList->relevantDomainListIndices[relevantIndex]],
                               particles->y[domainList->relevantDomainListIndices[relevantIndex]],
                               particles->z[domainList->relevantDomainListIndices[relevantIndex]],
                               particles->mass[domainList->relevantDomainListIndices[relevantIndex]]);*/
                    }

                    insert = true;
                    isDomainListNode = false;

                    if (sendIndices[bodyIndex + offset] == 0 || sendIndices[bodyIndex + offset] == 3 && ((bodyIndex + offset) < n || (bodyIndex + offset) >= m )) {

                        if (sendIndices[bodyIndex + offset] == 0) {
                            // TODO: this is probably not necessary, since only domain list indices can correspond to another process
                            //if (subDomainKeyTree->key2proc(
                            //        tree->getParticleKey(particles, bodyIndex + offset, MAX_LEVEL, curveType)) !=
                            //    subDomainKeyTree->rank) {
                            //    insert = false;
                            //}
                            // check whether to be inserted index corresponds to a domain list
                            if (insert) {
                                for (int i_domain = 0; i_domain < *domainList->domainListIndex; i_domain++) {
                                    if ((bodyIndex + offset) == domainList->domainListIndices[i_domain]) {
                                        insert = false;
                                        isDomainListNode = true;
                                        break;
                                    }
                                }
                            }

                            if (insert) {
                                sendIndices[bodyIndex + offset] = 3;
                            } else {
                                sendIndices[bodyIndex + offset] = -1;
                            }
                        }

                        // get the particle's level
                        particleLevel = tree->getTreeLevel(particles, bodyIndex + offset, MAX_LEVEL, curveType);
                        /*if (particleLevel == -1) {
                            printf("particleLevel == -1 for index = %i! (%f, %f, %f) %f (treeIndex = %i)\n", bodyIndex + offset,
                                   particles->x[bodyIndex + offset],
                                   particles->y[bodyIndex + offset],
                                   particles->z[bodyIndex + offset],
                                   particles->mass[bodyIndex + offset], *tree->index);
                            printf("particleLevel == -1 for index = %i! index = %i --> (%f, %f, %f) %f (treeIndex = %i)\n", bodyIndex + offset, bodyIndex + offset + 1,
                                   particles->x[bodyIndex + offset + 1],
                                   particles->y[bodyIndex + offset + 1],
                                   particles->z[bodyIndex + offset + 1],
                                   particles->mass[bodyIndex + offset + 1], *tree->index);
                            printf("particleLevel == -1 for index = %i! index = %i --> (%f, %f, %f) %f (treeIndex = %i)\n", bodyIndex + offset, bodyIndex + offset + 2,
                                   particles->x[bodyIndex + offset + 2],
                                   particles->y[bodyIndex + offset + 2],
                                   particles->z[bodyIndex + offset + 2],
                                   particles->mass[bodyIndex + offset + 2], *tree->index);
                            printf("particleLevel == -1 for index = %i! index = %i --> (%f, %f, %f) %f (treeIndex = %i)\n", bodyIndex + offset, bodyIndex + offset + 3,
                                   particles->x[bodyIndex + offset + 3],
                                   particles->y[bodyIndex + offset + 3],
                                   particles->z[bodyIndex + offset + 3],
                                   particles->mass[bodyIndex + offset + 3], *tree->index);
                            printf("particleLevel == -1 for index = %i! index = %i --> (%f, %f, %f) %f (treeIndex = %i)\n", bodyIndex + offset, bodyIndex + offset + 4,
                                   particles->x[bodyIndex + offset + 4],
                                   particles->y[bodyIndex + offset + 4],
                                   particles->z[bodyIndex + offset + 4],
                                   particles->mass[bodyIndex + offset + 4], *tree->index);
                            for (int i_child; i_child < POW_DIM; i_child++) {
                                if (tree->child[POW_DIM * (bodyIndex + offset) + i_child] != -1) {
                                    printf("index = %i, child %i = %i (%f, %f, %f) %f\n",
                                           bodyIndex + offset, i_child, tree->child[POW_DIM * (bodyIndex + offset) + i_child],
                                           particles->x[tree->child[POW_DIM * (bodyIndex + offset) + i_child]],
                                           particles->y[tree->child[POW_DIM * (bodyIndex + offset) + i_child]],
                                           particles->z[tree->child[POW_DIM * (bodyIndex + offset) + i_child]],
                                           particles->mass[tree->child[POW_DIM * (bodyIndex + offset) + i_child]]);
                                }
                            }
                        }*/

                        // get the domain list node's level
                        //domainListLevel = tree->getTreeLevel(particles,
                        //                                     domainList->relevantDomainListIndices[relevantIndex],
                        //                                     MAX_LEVEL, curveType);
                        domainListLevel = domainList->relevantDomainListLevels[relevantIndex];
                        //printf("domainListLevel = %i\n", domainListLevel);
                        if (domainListLevel == -1) {
                            printf("symbolicForce(): domainListLevel == -1 for (relevant) index: %i\n", relevantIndex);
                            assert(0);
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
                        //printf("(%f, %f), (%f, %f), (%f, %f)\n", min_x, max_x, min_y, max_y, min_z, max_z);

                        // determine domain list node's bounding box (in order to determine the distance)
                        //if (domainListLevel != 1) {
                        //    printf("domainListLevel = %i\n", domainListLevel);
                        //    assert(0);
                        //}
                        for (int j = 0; j < domainListLevel; j++) {

                            if (particles->x[domainList->relevantDomainListIndices[relevantIndex]] <= max_x && particles->x[domainList->relevantDomainListIndices[relevantIndex]] >= min_x &&
                                particles->y[domainList->relevantDomainListIndices[relevantIndex]] <= max_y && particles->y[domainList->relevantDomainListIndices[relevantIndex]] >= min_y &&
                                particles->z[domainList->relevantDomainListIndices[relevantIndex]] <= max_z && particles->z[domainList->relevantDomainListIndices[relevantIndex]] >= min_z) {

                            }
                            else {
                                printf("not within box %i, %i (%f, %f, %f) box (%f, %f), (%f, %f), (%f, %f)!\n", relevantIndex, domainList->relevantDomainListIndices[relevantIndex],
                                       particles->x[domainList->relevantDomainListIndices[relevantIndex]], particles->y[domainList->relevantDomainListIndices[relevantIndex]],
                                       particles->z[domainList->relevantDomainListIndices[relevantIndex]],
                                       min_x, max_x, min_y, max_y, min_z, max_z);
                                //assert(0);
                            }
                            childPath = 0;
                            if (particles->x[domainList->relevantDomainListIndices[relevantIndex]] < 0.5 * (min_x + max_x)) {
                                childPath += 1;
                                max_x = 0.5 * (min_x + max_x);
                            } else {
                                min_x = 0.5 * (min_x + max_x);
                            }
#if DIM > 1
                            if (particles->y[domainList->relevantDomainListIndices[relevantIndex]] < 0.5 * (min_y + max_y)) {
                                childPath += 2;
                                max_y = 0.5 * (min_y + max_y);
                            } else {
                                min_y = 0.5 * (min_y + max_y);
                            }
#if DIM == 3
                            if (particles->z[domainList->relevantDomainListIndices[relevantIndex]] < 0.5 * (min_z + max_z)) {
                                childPath += 4;
                                max_z = 0.5 * (min_z + max_z);
                            } else {
                                min_z = 0.5 * (min_z + max_z);
                            }
#endif
#endif
                        }

                        // determine (smallest) distance between domain list box and (pseudo-) particle
                        if (particles->x[bodyIndex + offset] < min_x) { dx = particles->x[bodyIndex + offset] - min_x;
                        } else if (particles->x[bodyIndex + offset] > max_x) { dx = particles->x[bodyIndex + offset] - max_x;
                        } else { dx = 0.f; }
#if DIM > 1
                        if (particles->y[bodyIndex + offset] < min_y) { dy = particles->y[bodyIndex + offset] - min_y;
                        } else if (particles->y[bodyIndex + offset] > max_y) { dy = particles->y[bodyIndex + offset] - max_y;
                        } else { dy = 0.f; }
#if DIM == 3
                        if (particles->z[bodyIndex + offset] < min_z) { dz = particles->z[bodyIndex + offset] - min_z;
                        } else if (particles->z[bodyIndex + offset] > max_z) { dz = particles->z[bodyIndex + offset] - max_z;
                        } else { dz = 0.f; }

#endif
#endif

#if DIM == 1
                        r = sqrtf(dx*dx);
#elif DIM == 2
                        r = sqrtf(dx*dx + dy*dy);
#else
                        r = sqrtf(dx*dx + dy*dy + dz*dz);
#endif

                        //printf("%f >= %f (particleLevel = %i, theta = %f, r = %f)\n", powf(0.5, particleLevel-1) /* * 2*/ * diam, (theta_ * r), particleLevel, theta_, r);
                        if (particleLevel != -1 && ((powf(0.5, particleLevel-1) * 2 * diam) >= (theta_ * r))) {

                            for (int i = 0; i < POW_DIM; i++) {

                                //if (sendIndices[tree->child[POW_DIM * (bodyIndex + offset) + i]] != 1 && tree->child[POW_DIM * (bodyIndex + offset) + i] != -1) {
                                //    sendIndices[tree->child[POW_DIM * (bodyIndex + offset) + i]] = 2;
                                //}

                                if (tree->child[POW_DIM * (bodyIndex + offset) + i] != -1) {
                                    if (sendIndices[tree->child[POW_DIM * (bodyIndex + offset) + i]] != 1) {
                                        sendIndices[tree->child[POW_DIM * (bodyIndex + offset) + i]] = 2;
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
                        subDomainKeyTree->range[i] = helper->keyTypeBuffer[bodyIndex + offset];
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
                    //atomicAdd(&tree->count[childIndex], 1);
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
                        //atomicAdd(&tree->count[temp], 1); // ? do not count, since particles are just temporarily saved on this process
                        childIndex = tree->child[POW_DIM*temp + childPath];

                    }
                    if (childIndex != -1) {
                        printf("ATTENTION: insertReceivedPseudoParticles(): childIndex = %i temp = %i\n", childIndex, temp);
                        printf("[rank %i] (%f, %f, %f) vs (%f, %f, %f)\n", subDomainKeyTree->rank,
                               particles->x[tree->toDeleteNode[0] + bodyIndex + offset],
                               particles->y[tree->toDeleteNode[0] + bodyIndex + offset],
                               particles->z[tree->toDeleteNode[0] + bodyIndex + offset],
                               particles->x[childIndex],
                               particles->y[childIndex],
                               particles->z[childIndex]);
                        assert(0);
                    }

                    insertionLevel++;

                    tree->child[POW_DIM*temp + childPath] = tree->toDeleteNode[0] + bodyIndex + offset;

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
                               tree->toDeleteNode[0] + bodyIndex + offset, bodyIndex + offset, levels[bodyIndex + offset], insertionLevel);
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
                    //atomicAdd(&tree->count[temp], 1); // do not count, since particles are just temporarily saved on this process
                    childIndex = tree->child[POW_DIM*temp + childPath];

                }

                if (childIndex != -1) {
                    printf("ATTENTION: insertReceivedParticles(): childIndex = %i (%i, %i) (%i, %i)\n", childIndex,
                           tree->toDeleteLeaf[0], tree->toDeleteLeaf[1], tree->toDeleteNode[0], tree->toDeleteNode[1]);
                    assert(0);
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

        __global__ void repairTree(Tree *tree, Particles *particles, DomainList *domainList, int n, int m) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            if (bodyIndex + offset == 0) {
                *tree->index = tree->toDeleteNode[0];
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

        real Launch::computeForces(Tree *tree, Particles *particles, integer n, integer m, integer blockSize,
                                   integer warp, integer stackSize, SubDomainKeyTree *subDomainKeyTree) {

            size_t sharedMemory = (sizeof(real)+sizeof(integer))*stackSize*blockSize/warp;
            //size_t sharedMemory = 2*sizeof(real)*stackSize*blockSize/warp;
            ExecutionPolicy executionPolicy(256, 256, sharedMemory);
            //ExecutionPolicy executionPolicy(512, 256, sharedMemory);
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::computeForces, tree, particles, n, m,
                                blockSize, warp, stackSize, subDomainKeyTree);
        }

        real Launch::computeForcesUnsorted(Tree *tree, Particles *particles, integer n, integer m, integer blockSize,
                                   integer warp, integer stackSize, SubDomainKeyTree *subDomainKeyTree) {

            size_t sharedMemory = (sizeof(real)+sizeof(integer))*stackSize*blockSize/warp;
            //size_t sharedMemory = 2*sizeof(real)*stackSize*blockSize/warp;
            ExecutionPolicy executionPolicy(256, 256, sharedMemory);
            //ExecutionPolicy executionPolicy(512, 256, sharedMemory);
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::computeForcesUnsorted, tree, particles, n, m,
                                blockSize, warp, stackSize, subDomainKeyTree);
        }

        real Launch::computeForcesMiluphcuda(Tree *tree, Particles *particles, integer n, integer m,
                                             SubDomainKeyTree *subDomainKeyTree) {
            size_t sharedMemory = sizeof(real) * MAX_DEPTH;
            ExecutionPolicy executionPolicy(256, 256, sharedMemory);
            //ExecutionPolicy executionPolicy(512, 256, sharedMemory);
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::computeForcesMiluphcuda, tree, particles, n, m,
                                subDomainKeyTree);
        }


        real Launch::update(Particles *particles, integer n, real dt, real d) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::update, particles, n, dt, d);
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

        real Launch::repairTree(Tree *tree, Particles *particles, DomainList *domainList, int n, int m) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::repairTree, tree, particles, domainList,
                                n, m);
        }

    }
}
