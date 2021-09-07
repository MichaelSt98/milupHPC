#include "../../include/gravity/gravity.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

namespace Gravity {

    namespace Kernel {

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
                            //printf("lowestDomainIndex = %i: realBuffer = %f, x = %f\n", lowestDomainIndex,
                            //       helper->realBuffer[bodyIndex + offset], particles->x[lowestDomainIndex]);
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
                            //printf("lowestDomainIndex = %i: realBuffer = %f, mass = %f\n", lowestDomainIndex,
                            //       helper->realBuffer[bodyIndex + offset], particles->mass[lowestDomainIndex]);
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
                        //printf("lowestDomainIndex: lowestDomainList->domainListIndices[%i] = %i == %f\n", originalIndex,
                        //       lowestDomainList->domainListIndices[originalIndex], helper->realBuffer[DOMAIN_LIST_SIZE + bodyIndex + offset]);
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

        __global__ void compLowestDomainListNodes(Particles *particles, DomainList *lowestDomainList) {

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
                            //printf("x += %f * %f = %f (%i)\n", particles->x[tree->child[POW_DIM*domainIndex + i]], particles->mass[tree->child[POW_DIM*domainIndex + i]],
                            //       particles->x[tree->child[POW_DIM*domainIndex + i]] * particles->mass[tree->child[POW_DIM*domainIndex + i]],
                            //       tree->child[POW_DIM*domainIndex + i]);
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

        __global__ void computeForces(Tree *tree, Particles *particles, integer n, integer m, integer blockSize,
                                      integer warp, integer stackSize) {

            integer bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
            integer stride = blockDim.x*gridDim.x;
            integer offset = 0;

            //__shared__ float depth[stackSize * blockSize/warp];
            // stack controlled by one thread per warp
            //__shared__ int   stack[stackSize * blockSize/warp];
            extern __shared__ real buffer[];

            real* depth = (real*)buffer;
            real* stack = (real*)&depth[stackSize* blockSize/warp];

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

            // in case that one of the first 8 children are a leaf
            integer jj = -1;
            for (integer i=0; i<POW_DIM; i++) {
                if (tree->child[i] != -1) {
                    jj++;
                }
            }

            integer counter = threadIdx.x % warp;
            integer stackStartIndex = stackSize*(threadIdx.x / warp);

            while ((bodyIndex + offset) < m) {

                integer sortedIndex = tree->sorted[bodyIndex + offset];

                //if ((bodyIndex + offset) % 1000 == 0) {
                //    printf("computeForces: sortedIndex = %i\n", sortedIndex);
                //}

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
                        // if child is not locked
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
                    //debug
                    //if (node > n && node < m) {
                    //    printf("PARALLEL FORCE! (node = %i x = (%f, %f, %f) m = %f)\n", node, x[node], y[node], z[node],
                    //        mass[node]);
                    //}
                    //end: debug
                    real dp = 0.25*depth[top]; // float dp = depth[top];

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

                            real r = dx*dx + 0.05; //0.0025; //NEW: TODO: needed for smoothing
#if DIM > 1
                            r += dy*dy;
#if DIM == 3
                            r += dz*dz;
#endif
#endif

                            //unsigned activeMask = __activemask();

                            //if (ch < n /*is leaf node*/ || !__any_sync(activeMask, dp > r)) {
                            if (ch < n /*is leaf node*/ || __all_sync(__activemask(), dp <= r)) { //NEW: && ch != sortedIndex

                                /*//debug
                                key = getParticleKeyPerParticle(x[ch], y[ch], z[ch], minX, maxX, minY, maxY,
                                                                minZ, maxZ, 21);
                                if (key2proc(key, s) != s->rank) {
                                    printf("Parallel force! child = %i x = (%f, %f, %f) mass = %f\n", ch, x[ch], y[ch], z[ch], mass[ch]);
                                }
                                //end: debug*/

                                // calculate interaction force contribution
                                if (r > 0.f) { //NEW //TODO: how to avoid r = 0?
                                    r = rsqrt(r);
                                }
                                //if (r == 0.f) {
                                //    printf("r = 0!!! x[%i] = (%f, %f, %f) vs x[%i] = (%f, %f, %f)\n", sortedIndex,
                                //           particles->x[sortedIndex], particles->y[sortedIndex], particles->z[sortedIndex],
                                //           ch, particles->x[ch], particles->y[ch], particles->z[ch]);
                                //}
                                real f = particles->mass[ch] * r * r * r;// + 0.0025;



                                acc_x += f*dx; // * 0.0001;
#if DIM > 1
                                acc_y += f*dy; // * 0.0001;
#if DIM == 3
                                acc_z += f*dz; // * 0.0001;
#endif
#endif
                                /*if (acc_x > 500000) {
                                    printf("huge acceleration!!! r = %f acc = (%f, %f, %f) x[%i] = (%f, %f, %f) m = %f vs x[%i] = (%f, %f, %f) m = %f\n",
                                           r, acc_x, acc_y, acc_z, sortedIndex,
                                           particles->x[sortedIndex], particles->y[sortedIndex],
                                           particles->z[sortedIndex], particles->mass[sortedIndex],
                                           ch, particles->x[ch], particles->y[ch], particles->z[ch],
                                           particles->mass[ch]);
                                }*/
                                //if (particles->mass[ch] > 10000) {
                                //    printf("mass is huge for ch=%i with node = %i (mass = %f, r = %f, f = %f -> acc = (%f, %f, %f) f = (%f, %f, %f))!\n", ch, node, particles->mass[ch],
                                //    r, f, acc_x, acc_y, acc_z, f*dx, f*dy, f*dz);
                                //}
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
                        else { /*top = max(stackStartIndex, top-1); */}
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

        __global__ void update(Particles *particles, integer n, real dt, real d) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            while (bodyIndex + offset < n) {

                /*if ((bodyIndex + offset) % 1000 == 0) {
                    printf("index= %i: velocity = = (%f, %f, %f)  acceleration = (%f, %f, %f)\n",
                           bodyIndex + offset,
                           particles->vx[bodyIndex + offset], particles->vy[bodyIndex + offset],
                           particles->vz[bodyIndex + offset],
                           particles->ax[bodyIndex + offset], particles->ay[bodyIndex + offset],
                           particles->az[bodyIndex + offset]);
                }*/

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

        __global__ void symbolicForce(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                      DomainList *domainList, Helper *helper,
                                      real diam, real theta_, integer n, integer m, integer relevantIndex,
                                      Curve::Type curveType) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;
            real r;
            integer insertIndex;
            bool insert;
            integer level;
            integer childIndex;
            //bool redo = false;

            while ((bodyIndex + offset) < *tree->index) {

                insert = true;
                //redo = false;

                for (integer i=0; i<*domainList->domainListIndex; i++) {
                    if ((bodyIndex + offset) == domainList->domainListIndices[i]) {
                        insert = false;
                        break;
                    }
                }

                //if (mass[relevantDomainListIndices[relevantIndex]] == 0) {
                //    insert = false;
                //}

                if (insert && (bodyIndex + offset) != domainList->relevantDomainListIndices[relevantIndex] &&
                    ((bodyIndex + offset) < subDomainKeyTree->procParticleCounter[subDomainKeyTree->rank] || (bodyIndex + offset) > n)) {

                    r = particles->distance(relevantIndex, bodyIndex + offset);
                    //r = smallestDistance(x, y, z, relevantDomainListIndices[relevantIndex], bodyIndex + offset);

                    //calculate tree level by determining the particle's key and traversing the tree until hitting that particle
                    level = tree->getTreeLevel(particles, bodyIndex + offset, MAX_LEVEL, curveType);
                    //level = getTreeLevel(bodyIndex + offset, child, x, y, z, minX, maxX, minY, maxY, minZ, maxZ);

                    if ((powf(0.5, level) * diam) >= (theta_ * r) && level >= 0) {
                        //TODO: insert cell itself or children?

                        /// inserting cell itself
                        /* //check whether node is a domain list node
                        for (int i=0; i<*domainList->domainListIndex; i++) {
                            if ((bodyIndex + offset) == domainList->domainListIndices[i]) {
                                insert = false;
                                break;
                                //printf("domain list nodes do not need to be sent!\n");

                            }
                        }
                        if (insert) {
                            //add to indices to be sent
                            insertIndex = atomicAdd(domainList->domainListCounter, 1);
                            //sendIndices[insertIndex] = bodyIndex + offset;
                            helper->integerBuffer[insertIndex] = bodyIndex + offset;
                        }
                        else {

                        }*/

                        /// inserting children
                        for (int i=0; i<POW_DIM; i++) {
                            childIndex = tree->child[POW_DIM * (bodyIndex + offset) + i];
                            //check whether node is already within the indices to be sent
                            //check whether node is a domain list node
                            for (int i = 0; i < *domainList->domainListIndex; i++) {
                                if (childIndex == domainList->domainListIndices[i]) {
                                    insert = false;
                                    //printf("domain list nodes do not need to be sent!\n");
                                }
                            }
                            if (insert && childIndex != -1) {
                                //add to indices to be sent
                                insertIndex = atomicAdd(domainList->domainListCounter, 1);
                                helper->integerBuffer[insertIndex] = childIndex;
                                //sendIndices[insertIndex] = childIndex;
                            }
                        }
                    }
                }
                else {
                    //no particle to examine...
                }
                offset += stride;
            }

        }

        __global__ void compTheta(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                  DomainList *domainList, Helper *helper, Curve::Type curveType) {

            integer index = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;
            integer bodyIndex;
            keyType key;
            integer domainIndex;

            //"loop" over domain list nodes
            while ((index + offset) < *domainList->domainListIndex) {

                bodyIndex = domainList->domainListIndices[index + offset];
                //calculate key
                key = tree->getParticleKey(particles, bodyIndex, MAX_LEVEL, curveType);
                //key = getParticleKeyPerParticle(x[bodyIndex], y[bodyIndex], z[bodyIndex], minX, maxX, minY, maxY,
                //                                minZ, maxZ, 21);

                //if domain list node belongs to other process: add to relevant domain list indices
                if (subDomainKeyTree->key2proc(key) != subDomainKeyTree->rank) {
                    domainIndex = atomicAdd(domainList->domainListCounter, 1);
                    domainList->relevantDomainListIndices[domainIndex] = bodyIndex;
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

        __global__ void insertReceivedParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
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

            while ((bodyIndex + offset) < tree->toDeleteLeaf[1] && (bodyIndex + offset) > tree->toDeleteLeaf[0]) {

                if ((bodyIndex + offset) % 100 == 0) {
                    printf("Inserting received particle %i: x = (%f, %f, %f) m = %f\n", bodyIndex + offset,
                           particles->x[bodyIndex + offset], particles->y[bodyIndex + offset],
                           particles->z[bodyIndex + offset], particles->mass[bodyIndex + offset]);
                }

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
                while (childIndex >= m /*&& childIndex < (8*m)*/) { //formerly n

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

                    for (int i=0; i<*domainList->domainListIndex; i++) {
                        if (temp == domainList->domainListIndices[i]) {
                            isDomainList = true;
                            break;
                        }
                    }

                    //TODO: !!!
                    if (/*true*/ !isDomainList) {
                        if (particles->mass[bodyIndex + offset] != 0) {
                            atomicAdd(&particles->x[temp], particles->mass[bodyIndex + offset] * particles->x[bodyIndex + offset]);
#if DIM > 1
                            atomicAdd(&particles->y[temp], particles->mass[bodyIndex + offset] * particles->y[bodyIndex + offset]);
#if DIM == 3
                            atomicAdd(&particles->z[temp], particles->mass[bodyIndex + offset] * particles->z[bodyIndex + offset]);
#endif
#endif
                        }
                        atomicAdd(&particles->mass[temp], particles->mass[bodyIndex + offset]);
                        //atomicAdd(&count[temp], 1); // do not count, since particles are just temporarily saved on this process
                    }
                    atomicAdd(&tree->count[temp], 1); // do not count, since particles are just temporarily saved on this process
                    childIndex = tree->child[POW_DIM*temp + childPath];
                }

                // if child is not locked
                if (childIndex != -2) {

                    int locked = temp * POW_DIM + childPath;

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

                                // TODO: remove!
                                // debug
                                if (particles->x[childIndex] == particles->x[bodyIndex + offset] &&
                                        particles->y[childIndex] == particles->y[bodyIndex + offset]) {
                                    printf("[rank %i] ATTENTION!!! %i vs. %i ((%f, %f, %f) vs (%f, %f, %f))\n", subDomainKeyTree->rank,
                                           childIndex, bodyIndex + offset,
                                           particles->x[childIndex], particles->y[childIndex], particles->z[childIndex],
                                           particles->x[bodyIndex+offset], particles->y[bodyIndex+offset], particles->z[bodyIndex+offset]);
                                    break;
                                }
                                // end: debug

                                // insert old/original particle
                                childPath = 0;
                                if (particles->x[childIndex] < 0.5 * (min_x + max_x)) { childPath += 1; }
#if DIM > 1
                                if (particles->y[childIndex] < 0.5 * (min_y + max_y)) { childPath += 2; }
#if DIM == 3
                                if (particles->z[childIndex] < 0.5 * (min_z + max_z)) { childPath += 4; }
#endif
#endif

                                particles->x[cell] += particles->mass[childIndex] * particles->x[childIndex];
#if DIM > 1
                                particles->y[cell] += particles->mass[childIndex] * particles->y[childIndex];
#if DIM == 3
                                particles->z[cell] += particles->mass[childIndex] * particles->z[childIndex];
#endif
#endif

                                particles->mass[cell] += particles->mass[childIndex];
                                // do not count, since particles are just temporarily saved on this process
                                tree->count[cell] += tree->count[childIndex];

                                tree->child[POW_DIM * cell + childPath] = childIndex;

                                tree->start[cell] = -1; //TODO: resetting start needed in insertReceivedParticles()?

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
                                    particles->x[cell] += particles->mass[bodyIndex + offset] * particles->x[bodyIndex + offset];
#if DIM > 1
                                    particles->y[cell] += particles->mass[bodyIndex + offset] * particles->y[bodyIndex + offset];
#if DIM == 3
                                    particles->z[cell] += particles->mass[bodyIndex + offset] * particles->z[bodyIndex + offset];
#endif
#endif
                                    particles->mass[cell] += particles->mass[bodyIndex + offset];
                                }
                                // do not count, since particles are just temporarily saved on this process
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
                    else {

                    }
                }
                else {

                }
                __syncthreads();
            }

        }

        __global__ void centreOfMassReceivedParticles(Particles *particles, integer *startIndex,
                                                            integer *endIndex, int n) {

            integer bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
            integer stride = blockDim.x*gridDim.x;
            integer offset = 0;

            //note: most of it already done within buildTreeKernel
            bodyIndex += *startIndex;

            while ((bodyIndex + offset) < *endIndex) {

                //if (particles->mass[bodyIndex + offset] == 0) {
                //    printf("centreOfMassKernel: mass = 0 (%i)!\n", bodyIndex + offset);
                //}

                if (particles->mass[bodyIndex + offset] != 0) {
                    particles->x[bodyIndex + offset] /= particles->mass[bodyIndex + offset];
                    particles->y[bodyIndex + offset] /= particles->mass[bodyIndex + offset];
                    particles->z[bodyIndex + offset] /= particles->mass[bodyIndex + offset];
                }

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

                particles->x[bodyIndex + offset] = 0;
                particles->vx[bodyIndex + offset] = 0;
                particles->ax[bodyIndex + offset] = 0;
#if DIM > 1
                particles->y[bodyIndex + offset] = 0;
                particles->vy[bodyIndex + offset] = 0;
                particles->ay[bodyIndex + offset] = 0;
#if DIM == 3
                particles->z[bodyIndex + offset] = 0;
                particles->vz[bodyIndex + offset] = 0;
                particles->az[bodyIndex + offset] = 0;
#endif
#endif
                particles->mass[bodyIndex + offset] = 0;
                tree->start[bodyIndex + offset] = -1;
                //sorted[bodyIndex + offset] = 0;

                offset += stride;
            }

            offset = tree->toDeleteNode[0]; //0;
            //delete inserted cells
            while ((bodyIndex + offset) >= tree->toDeleteNode[0] && (bodyIndex + offset) < tree->toDeleteNode[1]) {
                for (int i=0; i<POW_DIM; i++) {
                    tree->child[(bodyIndex + offset)*POW_DIM + i] = -1;
                }
                tree->count[bodyIndex + offset] = 0;
                particles->x[bodyIndex + offset] = 0;
                particles->vx[bodyIndex + offset] = 0;
                particles->ax[bodyIndex + offset] = 0;
#if DIM > 1
                particles->y[bodyIndex + offset] = 0;
                particles->vy[bodyIndex + offset] = 0;
                particles->ay[bodyIndex + offset] = 0;
#if DIM == 3
                particles->z[bodyIndex + offset] = 0;
                particles->vz[bodyIndex + offset] = 0;
                particles->az[bodyIndex + offset] = 0;
#endif
#endif
                particles->mass[bodyIndex + offset] = 0;
                tree->start[bodyIndex + offset] = -1;
                //sorted[bodyIndex + offset] = 0;

                offset += stride;
            }
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

        real Launch::compLowestDomainListNodes(Particles *particles, DomainList *lowestDomainList) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::compLowestDomainListNodes, particles,
                                lowestDomainList);
        }

        real Launch::compLocalPseudoParticles(Tree *tree, Particles *particles, DomainList *domainList, int n) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::compLocalPseudoParticles, tree, particles,
                                domainList, n);
        }

        real Launch::compDomainListPseudoParticles(Tree *tree, Particles *particles, DomainList *domainList,
                                                   DomainList *lowestDomainList, int n) {
            ExecutionPolicy executionPolicy(256, 1);
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::compDomainListPseudoParticles, tree,
                                particles, domainList, lowestDomainList, n);
        }

        real Launch::computeForces(Tree *tree, Particles *particles, integer n, integer m, integer blockSize,
                                   integer warp, integer stackSize) {

            //TODO: check shared memory size
            //size_t sharedMemory = (sizeof(real)+sizeof(integer))*stackSize*blockSize/warp;
            size_t sharedMemory = 2*sizeof(real)*stackSize*blockSize/warp;
            ExecutionPolicy executionPolicy(256, 256, sharedMemory);
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::computeForces, tree, particles, n, m,
                                blockSize, warp, stackSize);
        }

        real Launch::update(Particles *particles, integer n, real dt, real d) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::update, particles, n, dt, d);
        }

        real Launch::symbolicForce(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                           DomainList *domainList, Helper *helper,
                           real diam, real theta_, integer n, integer m, integer relevantIndex,
                           Curve::Type curveType) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::symbolicForce, subDomainKeyTree, tree,
                                particles, domainList, helper, diam, theta_, n, m, relevantIndex, curveType);
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

        real Launch::insertReceivedParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                     DomainList *domainList, DomainList *lowestDomainList, int n, int m) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::insertReceivedParticles, subDomainKeyTree,
                                tree, particles, domainList, lowestDomainList, n, m);
        }

        real Launch::centreOfMassReceivedParticles(Particles *particles, integer *startIndex,
                                           integer *endIndex, int n) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::centreOfMassReceivedParticles,
                                particles, startIndex, endIndex, n);
        }

        real Launch::repairTree(Tree *tree, Particles *particles, DomainList *domainList, int n, int m) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::Gravity::Kernel::repairTree, tree, particles, domainList,
                                n, m);
        }

    }
}
