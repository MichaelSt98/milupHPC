#include "../../include/subdomain_key_tree/subdomain.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

CUDA_CALLABLE_MEMBER void KeyNS::key2Char(keyType key, integer maxLevel, char *keyAsChar) {
    int level[21];
    for (int i=0; i<maxLevel; i++) {
        level[i] = (int)(key >> (maxLevel*DIM - DIM*(i+1)) & (int)(POW_DIM - 1));
    }
    for (int i=0; i<=maxLevel; i++) {
        keyAsChar[2*i] = level[i] + '0';
        keyAsChar[2*i+1] = '|';
    }
    keyAsChar[2*maxLevel+3] = '\0';
}

CUDA_CALLABLE_MEMBER integer KeyNS::key2proc(keyType key, SubDomainKeyTree *subDomainKeyTree/*, Curve::Type curveType*/) {
    return subDomainKeyTree->key2proc(key/*, curveType*/);
}

CUDA_CALLABLE_MEMBER SubDomainKeyTree::SubDomainKeyTree() {

}

CUDA_CALLABLE_MEMBER SubDomainKeyTree::SubDomainKeyTree(integer rank, integer numProcesses, keyType *range,
                                                        integer *procParticleCounter) : rank(rank),
                                                        numProcesses(numProcesses), range(range),
                                                        procParticleCounter(procParticleCounter) {

}

CUDA_CALLABLE_MEMBER SubDomainKeyTree::~SubDomainKeyTree() {

}

CUDA_CALLABLE_MEMBER void SubDomainKeyTree::set(integer rank, integer numProcesses, keyType *range,
                                                integer *procParticleCounter) {
    this->rank = rank;
    this->numProcesses = numProcesses;
    this->range = range;
    this->procParticleCounter = procParticleCounter;
}

CUDA_CALLABLE_MEMBER integer SubDomainKeyTree::key2proc(keyType key/*, Curve::Type curveType*/) {

    for (integer proc = 0; proc < numProcesses; proc++) {
        if (key >= range[proc] && key < range[proc + 1]) {
            return proc;
        }
    }
    //switch (curveType) {
    //    case Curve::lebesgue: {
    //        for (integer proc = 0; proc < numProcesses; proc++) {
    //            if (key >= range[proc] && key < range[proc + 1]) {
    //                return proc;
    //            }
    //        }
    //    }
    //    case Curve::hilbert: {
    //
    //        keyType hilbert = Lebesgue2Hilbert(key, 21);
    //        for (int proc = 0; proc < s->numProcesses; proc++) {
    //            if (hilbert >= s->range[proc] && hilbert < s->range[proc + 1]) {
    //                return proc;
    //            }
    //        }
    //
    //    }
    //    default: {
    //        printf("Curve type not available!\n");
    //    }
    //}
    printf("ERROR: key2proc(k=%lu): -1!", key);
    return -1; // error
}

CUDA_CALLABLE_MEMBER bool SubDomainKeyTree::isDomainListNode(keyType key, integer maxLevel, integer level,
                                                             Curve::Type curveType) {
    integer p1, p2;
    //p1 = key2proc(key);
    //p2 = key2proc(key | ~(~0UL << DIM * (maxLevel - level)));
    //TODO: necessary? always lebesgue key?
    switch (curveType) {
        case Curve::lebesgue: {
            p1 = key2proc(key);
            p2 = key2proc(key | ~(~0UL << DIM * (maxLevel - level)));
            break;
        }
        case Curve::hilbert: {
            //p1 = key2proc(KeyNS::lebesgue2hilbert(key, maxLevel));
            //p2 = key2proc(KeyNS::lebesgue2hilbert(key | ~(~0UL << DIM * (maxLevel - level)), maxLevel));

            p1 = key2proc(KeyNS::lebesgue2hilbert(key, maxLevel, level));
            //p2 = key2proc(KeyNS::lebesgue2hilbert(key | ~(~0UL << DIM * (maxLevel - level)), maxLevel, maxLevel));
            keyType hilbert = KeyNS::lebesgue2hilbert(key, maxLevel, level);
            p2 = key2proc(hilbert | (KEY_MAX >> (DIM*level+1)));

            //printf("lebesgue: %lu vs %lu < ? : %i\n", key, key | ~(~0UL << DIM * (maxLevel - level)), key < (key | ~(~0UL << DIM * (maxLevel - level))));
            //printf("hilbert: %lu vs %lu < ? : %i\n", KeyNS::lebesgue2hilbert(key, maxLevel, maxLevel), hilbert | (KEY_MAX >> (DIM*level+1)),
                   //KeyNS::lebesgue2hilbert(key, maxLevel, maxLevel) < (hilbert | (KEY_MAX >> (DIM*level+1))));

            break;
        }
        default: {
            printf("Curve type not available!\n");
        }
    }
    //printf("p1 = %i, p2 = %i\n", p1, p2);
    if (p1 != p2) {
        return true;
    }
    return false;
}

namespace SubDomainKeyTreeNS {

    namespace Kernel {

        __global__ void set(SubDomainKeyTree *subDomainKeyTree, integer rank, integer numProcesses, keyType *range,
                            integer *procParticleCounter) {
            subDomainKeyTree->set(rank, numProcesses, range, procParticleCounter);
        }

        __global__ void test(SubDomainKeyTree *subDomainKeyTree) {
            printf("device: subDomainKeyTree->rank = %i\n", subDomainKeyTree->rank);
            printf("device: subDomainKeyTree->numProcesses = %i\n", subDomainKeyTree->numProcesses);
        }

        __global__ void buildDomainTree(Tree *tree, Particles *particles, DomainList *domainList, integer n, integer m) {

            integer domainListCounter = 0;

            integer path[MAX_LEVEL];

            real min_x, max_x;
#if DIM > 1
            real min_y, max_y;
#if DIM == 3
            real min_z, max_z;
#endif
#endif
            integer currentChild;
            integer childPath;
            bool insert = true;

            integer childIndex;
            integer temp;

            // loop over domain list indices (over the keys found/generated by createDomainListKernel)
            for (int i = 0; i < *domainList->domainListIndex; i++) {
                //printf("domainListKey[%i] = %lu\n", i, domainList->domainListKeys[i]);
                childIndex = 0;
                // iterate through levels (of corresponding domainListIndex)
                for (int j = 0; j < domainList->domainListLevels[i]; j++) {
                    path[j] = (integer) (domainList->domainListKeys[i] >> (MAX_LEVEL * DIM - DIM * (j + 1)) &
                                    (integer)(POW_DIM - 1));
                    temp = childIndex;
                    childIndex = tree->child[POW_DIM*childIndex + path[j]];
                    if (childIndex < n) {
                        if (childIndex == -1) {
                            // no child at all here, thus add node
                            integer cell = atomicAdd(tree->index, 1);
                            tree->child[POW_DIM * temp + path[j]] = cell;
                            childIndex = cell;
                            domainList->domainListIndices[domainListCounter] = childIndex; //cell;
#if DIM == 3
                            printf("adding node index %i  cell = %i (childPath = %i,  j = %i)! x = (%f, %f, %f)\n",
                                   childIndex, cell, path[j], j, particles->x[childIndex], particles->y[childIndex], particles->z[childIndex]);
#endif
                            domainListCounter++;
                        } else {
                            // child is a leaf, thus add node in between
                            integer cell = atomicAdd(tree->index, 1);
                            tree->child[POW_DIM * temp + path[j]] = cell;

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

                            for (int k=0; k<=j; k++) {

                                currentChild = path[k];

                                if (currentChild % 2 != 0) {
                                    max_x = 0.5 * (min_x + max_x);
                                    currentChild -= 1;
                                }
                                else {
                                    min_x = 0.5 * (min_x + max_x);
                                }
#if DIM > 1
                                if (currentChild % 2 == 0 && currentChild % 4 != 0) {
                                    max_y = 0.5 * (min_y + max_y);
                                    currentChild -= 2;
                                }
                                else {
                                    min_y = 0.5 * (min_y + max_y);
                                }
#if DIM == 3
                                if (currentChild == 4) {
                                    max_z = 0.5 * (min_z + max_z);
                                    currentChild -= 4;
                                }
                                else {
                                    min_z = 0.5 * (min_z + max_z);
                                }
#endif
#endif
                            }
                            // insert old/original particle
                            childPath = 0; //(int) (domainListKeys[i] >> (21 * 3 - 3 * ((j+1) + 1)) & (int)7); //0; //currentChild; //0;
                            if (particles->x[childIndex] < 0.5 * (min_x + max_x)) {
                                childPath += 1;
                                //max_x = 0.5 * (min_x + max_x);
                            }
                            //else { min_x = 0.5 * (min_x + max_x); }
#if DIM > 1
                            if (particles->y[childIndex] < 0.5 * (min_y + max_y)) {
                                childPath += 2;
                                //max_y = 0.5 * (min_y + max_y);
                            }
                            //else { min_y = 0.5 * (min_y + max_y); }
#if DIM == 3
                            if (particles->z[childIndex] < 0.5 * (min_z + max_z)) {
                                childPath += 4;
                                //max_z = 0.5 * (min_z + max_z);
                            }
                            //else { min_z = 0.5 * (min_z + max_z); }
#endif
#endif

                            particles->x[cell] += particles->weightedEntry(childIndex, Entry::x);
#if DIM > 1
                            particles->y[cell] += particles->weightedEntry(childIndex, Entry::y);
#if DIM == 3
                            particles->z[cell] += particles->weightedEntry(childIndex, Entry::z);
#endif
#endif
                            particles->mass[cell] += particles->mass[childIndex];
#if DIM == 3
                            printf("adding node in between for index %i  cell = %i (childPath = %i,  j = %i)! x = (%f, %f, %f)\n",
                                   childIndex, cell, childPath, j, particles->x[childIndex], particles->y[childIndex], particles->z[childIndex]);
#endif

                            tree->child[POW_DIM * cell + childPath] = childIndex;
                            //printf("child[8 * %i + %i] = %i\n", cell, childPath, childIndex);

                            childIndex = cell;
                            domainList->domainListIndices[domainListCounter] = childIndex; //temp;
                            domainListCounter++;
                        }
                    }
                    else {
                        insert = true;
                        // check whether node already marked as domain list node
                        for (int k=0; k<domainListCounter; k++) {
                            if (childIndex == domainList->domainListIndices[k]) {
                                insert = false;
                                break;
                            }
                        }
                        if (insert) {
                            // mark/save node as domain list node
                            domainList->domainListIndices[domainListCounter] = childIndex; //temp;
                            domainListCounter++;
                        }
                    }
                }
            }
        }

        __global__ void getParticleKeys(SubDomainKeyTree *subDomainKeyTree, Tree *tree,
                                        Particles *particles, keyType *keys, integer maxLevel,
                                        integer n, Curve::Type curveType) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            keyType particleKey;
            keyType hilbertParticleKey;

            char keyAsChar[21 * 2 + 3];
            integer proc;

            while (bodyIndex + offset < n) {

                //particleKey = 0UL;
                particleKey = tree->getParticleKey(particles, bodyIndex + offset, maxLevel, curveType);

                // DEBUG
                //KeyNS::key2Char(particleKey, 21, keyAsChar);
                //printf("keyMax: %lu = %s\n", particleKey, keyAsChar);
                //proc = subDomainKeyTree->key2proc(particleKey);
                //if (proc == 0) {
                //    atomicAdd(tree->index, 1);
                //}
                //if ((bodyIndex + offset) % 1000 == 0) {
                //    printf("[rank %i] proc = %i, particleKey = %s = %lu\n", subDomainKeyTree->rank, proc,
                //           keyAsChar, particleKey);
                    //printf("[rank %i] particleKey = %lu, proc = %i\n", subDomainKeyTree->rank, particleKey,
                    //       proc);
                //}
                //if (subDomainKeyTree->rank != proc) {
                //    printf("[rank %i] particleKey = %lu, proc = %i\n", subDomainKeyTree->rank, particleKey,
                //           proc);
                //}

                keys[bodyIndex + offset] = particleKey; //hilbertParticleKey;

                //if ((bodyIndex + offset) % 100 == 0) {
                //    KeyNS::key2Char(particleKey, 21, keyAsChar);
                //    printf("key = %s = %lu\n", keyAsChar, particleKey);
                //}

                offset += stride;
            }
        }

        __global__ void particlesPerProcess(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                            integer n, integer m, Curve::Type curveType) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            keyType key;
            integer proc;

            while ((bodyIndex + offset) < n) {

                // calculate particle key from particle's position
                key = tree->getParticleKey(particles, bodyIndex + offset, MAX_LEVEL, curveType);
                // get corresponding process
                proc = subDomainKeyTree->key2proc(key);
                // increment corresponding counter
                atomicAdd(&subDomainKeyTree->procParticleCounter[proc], 1);

                offset += stride;
            }

        }

        __global__ void markParticlesProcess(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                             integer n, integer m, integer *sortArray, Curve::Type curveType) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            keyType key;
            integer proc;
            integer counter;

            while ((bodyIndex + offset) < n) {

                // calculate particle key from particle's position
                key = tree->getParticleKey(particles, bodyIndex + offset, MAX_LEVEL, curveType);
                // get corresponding process
                proc = subDomainKeyTree->key2proc(key);
                // mark particle with corresponding process
                sortArray[bodyIndex + offset] = proc;

                offset += stride;

            }
        }

        void Launch::set(SubDomainKeyTree *subDomainKeyTree, integer rank, integer numProcesses, keyType *range,
                         integer *procParticleCounter) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::SubDomainKeyTreeNS::Kernel::set, subDomainKeyTree, rank,
                         numProcesses, range, procParticleCounter);
        }

        void Launch::test(SubDomainKeyTree *subDomainKeyTree) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::SubDomainKeyTreeNS::Kernel::test, subDomainKeyTree);
        }

        real Launch::buildDomainTree(Tree *tree, Particles *particles, DomainList *domainList, integer n, integer m) {
            //TODO: is there any possibility to call kernel with more than one thread?
            ExecutionPolicy executionPolicy(1, 1);
            return cuda::launch(true, executionPolicy, ::SubDomainKeyTreeNS::Kernel::buildDomainTree, tree, particles,
                         domainList, n, m);
        }

        real Launch::getParticleKeys(SubDomainKeyTree *subDomainKeyTree, Tree *tree,
                             Particles *particles, keyType *keys, integer maxLevel,
                             integer n, Curve::Type curveType) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SubDomainKeyTreeNS::Kernel::getParticleKeys, subDomainKeyTree,
                                tree, particles, keys, maxLevel, n, curveType);
        }

        real Launch::particlesPerProcess(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                         integer n, integer m, Curve::Type curveType) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SubDomainKeyTreeNS::Kernel::particlesPerProcess,
                                subDomainKeyTree, tree, particles, n, m, curveType);
        }

        real Launch::markParticlesProcess(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                          integer n, integer m, integer *sortArray,
                                          Curve::Type curveType) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SubDomainKeyTreeNS::Kernel::markParticlesProcess,
                                subDomainKeyTree, tree, particles, n, m, sortArray, curveType);
        }

    }

}

CUDA_CALLABLE_MEMBER DomainList::DomainList() {

}

CUDA_CALLABLE_MEMBER DomainList::DomainList(integer *domainListIndices, integer *domainListLevels,
                                            integer *domainListIndex, integer *domainListCounter,
                                            keyType *domainListKeys, keyType *sortedDomainListKeys,
                                            integer *relevantDomainListIndices, integer *relevantDomainListLevels,
                                            integer *relevantDomainListProcess) :
                                            domainListIndices(domainListIndices), domainListLevels(domainListLevels),
                                            domainListIndex(domainListIndex), domainListCounter(domainListCounter),
                                            domainListKeys(domainListKeys), sortedDomainListKeys(sortedDomainListKeys),
                                            relevantDomainListIndices(relevantDomainListIndices),
                                            relevantDomainListLevels(relevantDomainListLevels),
                                            relevantDomainListProcess(relevantDomainListProcess) {

}

CUDA_CALLABLE_MEMBER DomainList::~DomainList() {

}

CUDA_CALLABLE_MEMBER void DomainList::set(integer *domainListIndices, integer *domainListLevels,
                                          integer *domainListIndex, integer *domainListCounter, keyType *domainListKeys,
                                          keyType *sortedDomainListKeys, integer *relevantDomainListIndices,
                                          integer *relevantDomainListLevels, integer *relevantDomainListProcess) {

    this->domainListIndices = domainListIndices;
    this->domainListLevels = domainListLevels;
    this->domainListIndex = domainListIndex;
    this->domainListCounter = domainListCounter;
    this->domainListKeys = domainListKeys;
    this->sortedDomainListKeys = sortedDomainListKeys;
    this->relevantDomainListIndices = relevantDomainListIndices;
    this->relevantDomainListLevels = relevantDomainListLevels;
    this->relevantDomainListProcess = relevantDomainListProcess;

    *domainListIndex = 0;
}

namespace DomainListNS {

    namespace Kernel {

        __global__ void set(DomainList *domainList, integer *domainListIndices, integer *domainListLevels,
                            integer *domainListIndex, integer *domainListCounter, keyType *domainListKeys,
                            keyType *sortedDomainListKeys, integer *relevantDomainListIndices,
                            integer *relevantDomainListLevels, integer *relevantDomainListProcess) {

            domainList->set(domainListIndices, domainListLevels, domainListIndex, domainListCounter, domainListKeys,
                            sortedDomainListKeys, relevantDomainListIndices, relevantDomainListLevels,
                            relevantDomainListProcess);
        }

        __global__ void info(Particles *particles, DomainList *domainList) {

            integer index = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            integer domainListIndex;

            /*if (index == 0) {
                printf("domainListIndices = [");
                for (int i=0; i<*domainList->domainListIndex; i++) {
                    printf("%i, ", domainList->domainListIndices[i]);
                }
                printf("]\n");
            }*/

#if DIM == 3
            while ((index + offset) < *domainList->domainListIndex) {

                domainListIndex = domainList->domainListIndices[index + offset];

                if (true) {
                    printf("domainListIndices[%i] = %i, level = %i, x = (%f, %f, %f) mass = %f\n", index + offset,
                           domainListIndex, domainList->domainListLevels[index + offset], particles->x[domainListIndex],
                           particles->y[domainListIndex], particles->z[domainListIndex],
                           particles->mass[domainListIndex]);
                }

                offset += stride;
            }
#endif

            /*if (index == 0) {
                while ((index + offset) < *domainList->domainListIndex) {
                    domainListIndex = domainList->domainListIndices[index + offset];
                }
            }*/

        }

        __global__ void info(Particles *particles, DomainList *domainList, DomainList *lowestDomainList) {

            integer index = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            integer domainListIndex;

            /*if (index == 0) {
                printf("domainListIndices = [");
                for (int i=0; i<*domainList->domainListIndex; i++) {
                    printf("%i, ", domainList->domainListIndices[i]);
                }
                printf("]\n");
            }*/

            bool show;

            while ((index + offset) < *domainList->domainListIndex) {

                show = true;
                domainListIndex = domainList->domainListIndices[index + offset];

                /*for (int i=0; i<*lowestDomainList->domainListIndex; i++) {
                    if (lowestDomainList->domainListIndices[i] == domainListIndex) {
                        printf("domainListIndices[%i] = %i, x = (%f, %f, %f) mass = %f\n", index + offset,
                               domainListIndex, particles->x[domainListIndex],
                               particles->y[domainListIndex], particles->z[domainListIndex], particles->mass[domainListIndex]);
                    }
                }*/

                for (int i=0; i<*lowestDomainList->domainListIndex; i++) {
                    if (lowestDomainList->domainListIndices[i] == domainListIndex) {
                        show = false;
                    }
                }

                if (show) {
#if DIM == 1
                    printf("domainListIndices[%i] = %i, x = (%f) mass = %f\n", index + offset,
                           domainListIndex, particles->x[domainListIndex], particles->mass[domainListIndex]);
#elif DIM == 2
                    printf("domainListIndices[%i] = %i, x = (%f, %f) mass = %f\n", index + offset,
                           domainListIndex, particles->x[domainListIndex],
                           particles->y[domainListIndex], particles->mass[domainListIndex]);
#else
                    printf("domainListIndices[%i] = %i, x = (%f, %f, %f) mass = %f\n", index + offset,
                           domainListIndex, particles->x[domainListIndex],
                           particles->y[domainListIndex], particles->z[domainListIndex], particles->mass[domainListIndex]);
#endif
                }

                offset += stride;
            }

        }

        __global__ void createDomainList(SubDomainKeyTree *subDomainKeyTree, DomainList *domainList,
                                         integer maxLevel, Curve::Type curveType) {

            char keyAsChar[21 * 2 + 3];
            // workaround for fixing bug... in principle: unsigned long keyMax = (1 << 63) - 1;
#if DIM == 1
            keyType shiftValue = 1;
            keyType toShift = 21;
            keyType keyMax = (shiftValue << toShift) - 1; // 1 << 63 not working!
#elif DIM == 2
            keyType shiftValue = 1;
            keyType toShift = 42;
            keyType keyMax = (shiftValue << toShift) - 1; // 1 << 63 not working!
#else
            keyType shiftValue = 1;
            keyType toShift = 63;
            keyType keyMax = (shiftValue << toShift) - 1; // 1 << 63 not working!
#endif
            //key2Char(keyMax, 21, keyAsChar); //printf("keyMax: %lu = %s\n", keyMax, keyAsChar);

            keyType key2test = 0UL;
            integer level = 1;
            //level++;

            // in principle: traversing a (non-existent) octree by walking the 1D spacefilling curve (keys of the tree nodes)
            //int counter = 0;
            while (key2test < keyMax) {
                //counter++;
                if (subDomainKeyTree->isDomainListNode(key2test & (~0UL << (DIM * (maxLevel - level + 1))),
                                                      maxLevel, level-1, curveType)) {
                    domainList->domainListKeys[*domainList->domainListIndex] = key2test;
                    // TODO: which key to save (always lebesgue???)
                    // add domain list key
                    //switch (curveType) {
                    //    case Curve::lebesgue:
                    //        domainList->domainListKeys[*domainList->domainListIndex] = key2test;
                    //        break;
                    //    case Curve::hilbert:
                    //        domainList->domainListKeys[*domainList->domainListIndex] = KeyNS::lebesgue2hilbert(key2test, maxLevel);
                    //        break;
                    //    default:
                    //        printf("Curve type not available!\n");
                    //}
                    //printf("[rank %i] Adding domain list with key = %lu\n", subDomainKeyTree->rank, key2test);
                    // add domain list level
                    domainList->domainListLevels[*domainList->domainListIndex] = level;
                    *domainList->domainListIndex += 1;
                    if (subDomainKeyTree->isDomainListNode(key2test, maxLevel, level, curveType)) {
                        level++;
                    }
                    else {
                        key2test = key2test + (1UL << DIM * (maxLevel - level));
                    }
                } else {
                    level--;
                    // not necessary: key2test = keyMaxLevel(key2test & (~0UL << (3 * (maxLevel - level))), maxLevel, level, s) + 1 - (1UL << (3 * (maxLevel - level)));
                }
                //if (counter == 100) {
                //    break;
                //}
            }
            //for (int i=0; i < *index; i++) {
            //    key2Char(domainListKeys[i], 21, keyAsChar);
            //}
            //KeyNS::key2Char(key2test, 21, keyAsChar);
            //printf("key2test = %s = %lu\n", keyAsChar, key2test);
        }

        __global__ void lowestDomainList(SubDomainKeyTree *subDomainKeyTree, Tree *tree, DomainList *domainList,
                                                       DomainList *lowestDomainList, integer n, integer m) {

            integer index = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            bool lowestDomainListNode;
            integer domainIndex;
            integer lowestDomainIndex;
            integer childIndex;

            // check all domain list nodes
            while ((index + offset) < *domainList->domainListIndex) {
                lowestDomainListNode = true;
                // get domain list index of current domain list node
                domainIndex = domainList->domainListIndices[index + offset];
                // check all children
                for (int i=0; i<POW_DIM; i++) {
                    childIndex = tree->child[POW_DIM * domainIndex + i];
                    // check whether child exists
                    if (childIndex != -1) {
                        // check whether child is a node
                        if (childIndex >= n) {
                            // check if this node is a domain list node
                            for (int k=0; k<*domainList->domainListIndex; k++) {
                                if (childIndex == domainList->domainListIndices[k]) {
                                    //printf("domainIndex = %i  childIndex: %i  domainListIndices: %i\n", domainIndex,
                                    //       childIndex, domainListIndices[k]);
                                    lowestDomainListNode = false;
                                    break;
                                }
                            }
                            // one child being a domain list node is sufficient for not being a lowest domain list node
                            if (!lowestDomainListNode) {
                                break;
                            }
                        }
                    }
                }

                if (lowestDomainListNode) {
                    // increment lowest domain list counter/index
                    lowestDomainIndex = atomicAdd(lowestDomainList->domainListIndex, 1);
                    // add/save index of lowest domain list node
                    lowestDomainList->domainListIndices[lowestDomainIndex] = domainIndex;
                    // add/save key of lowest domain list node
                    lowestDomainList->domainListKeys[lowestDomainIndex] = domainList->domainListKeys[index + offset];
                    // add/save level of lowest domain list node
                    lowestDomainList->domainListLevels[lowestDomainIndex] = domainList->domainListLevels[index + offset];
                    // debugging
                    //printf("Adding lowest domain list node #%i (key = %lu)\n", lowestDomainIndex,
                    //  lowestDomainListKeys[lowestDomainIndex]);
                }
                offset += stride;
            }

        }

        void Launch::set(DomainList *domainList, integer *domainListIndices, integer *domainListLevels,
                             integer *domainListIndex, integer *domainListCounter, keyType *domainListKeys,
                             keyType *sortedDomainListKeys, integer *relevantDomainListIndices,
                             integer *relevantDomainListLevels, integer *relevantDomainListProcess) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::DomainListNS::Kernel::set, domainList, domainListIndices, domainListLevels,
                         domainListIndex, domainListCounter, domainListKeys, sortedDomainListKeys,
                         relevantDomainListIndices, relevantDomainListLevels, relevantDomainListProcess);
        }

        real Launch::info(Particles *particles, DomainList *domainList) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::DomainListNS::Kernel::info, particles, domainList);
        }

        real Launch::info(Particles *particles, DomainList *domainList, DomainList *lowestDomainList) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::DomainListNS::Kernel::info, particles, domainList, lowestDomainList);
        }

        real Launch::createDomainList(SubDomainKeyTree *subDomainKeyTree, DomainList *domainList, integer maxLevel,
                                      Curve::Type curveType) {
            //TODO: is there any possibility to call kernel with more than one thread?
            ExecutionPolicy executionPolicy(1,1);
            return cuda::launch(true, executionPolicy, ::DomainListNS::Kernel::createDomainList, subDomainKeyTree,
                                domainList, maxLevel, curveType);
        }

        real Launch::lowestDomainList(SubDomainKeyTree *subDomainKeyTree, Tree *tree, DomainList *domainList,
                              DomainList *lowestDomainList, integer n, integer m) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::DomainListNS::Kernel::lowestDomainList, subDomainKeyTree,
                                tree, domainList, lowestDomainList, n, m);
        }
    }

}

namespace ParticlesNS {
    __device__ bool applyCriterion(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles, int index) {

#if DIM == 1
        if (sqrtf(particles->x[index] * particles->x[index]) < 0.95) {
#elif DIM == 2
        if (sqrtf(particles->x[index] * particles->x[index] +
                particles->y[index] * particles->y[index]) < 0.95) {
#else
        if (sqrtf(particles->x[index] * particles->x[index] + particles->y[index] * particles->y[index] +
                  particles->z[index] * particles->z[index]) < 0.95) {
#endif
            return false;
        } else {
            return true;
        }
    }

    namespace Kernel {

        __global__ void mark2remove(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                    int *particles2remove, int *counter, int numParticles) {

            int bodyIndex = threadIdx.x + blockDim.x * blockIdx.x;
            int stride = blockDim.x * gridDim.x;
            int offset = 0;

            while (bodyIndex + offset < numParticles) {

                if (::ParticlesNS::applyCriterion(subDomainKeyTree, tree, particles, bodyIndex + offset)) {
                    particles2remove[bodyIndex + offset] = 1;
                    atomicAdd(counter, 1);
                } else {
                    particles2remove[bodyIndex + offset] = 0;
                }

                offset += stride;
            }

        }

        real Launch::mark2remove(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                 int *particles2remove, int *counter, int numParticles) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::ParticlesNS::Kernel::mark2remove, subDomainKeyTree,
                                tree, particles, particles2remove, counter, numParticles);
        }

    }
}

namespace CudaUtils {

    namespace Kernel {

        template<typename T, typename U>
        __global__ void markDuplicatesTemp(Tree *tree, DomainList *domainList, T *array, U *entry1, U *entry2, U *entry3, integer *duplicateCounter, integer *child, int length) {

            integer index = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;
            integer maxIndex;

            bool isChild;
            //remark: check only x, but in principle check all
            while ((index + offset) < length) {
                if (array[index + offset] != -1) {
                    for (integer i = 0; i < length; i++) {
                        if (i != (index + offset)) {
                            if (array[i] != -1 && (array[index + offset] == array[i] || (entry1[array[i]] == entry1[array[index + offset]] &&
                                                                                         entry2[array[i]] == entry2[array[index + offset]] &&
                                                                                         entry3[array[i]] == entry3[array[index + offset]]))) {
                                isChild = false;

                                if (true/*array[index + offset] == array[i]*/) {
                                    printf("DUPLICATE: %i vs %i | (%f, %f, %f) vs (%f, %f, %f):\n",
                                           array[index + offset], array[i],
                                           entry1[array[index + offset]], entry2[array[index + offset]], entry3[array[index + offset]],
                                           entry1[array[i]], entry2[array[i]], entry3[array[i]]);

                                    for (int k=0; k<*domainList->domainListIndex; k++) {
                                        if (array[index + offset] == domainList->domainListIndices[k] ||
                                                array[i] == domainList->domainListIndices[k]) {
                                            printf("DUPLICATE is domainList!\n");
                                        }
                                    }

                                    for (int k=0; k<POW_DIM; k++) {
                                        if (child[POW_DIM*array[index + offset] + k] == array[i]) {
                                            printf("isChild: index = %i: child %i == i = %i\n", array[index + offset],
                                                   k, array[i]);
                                            isChild = true;
                                        }

                                        if (child[8*array[i] + k] == array[index + offset]) {
                                            printf("isChild: index = %i: child %i == index = %i\n", array[i],
                                                   k, array[index + offset]);
                                            isChild = true;
                                        }
                                    }

                                    if (!isChild) {
                                        printf("isChild: Duplicate NOT a child: %i vs %i | (%f, %f, %f) vs (%f, %f, %f):\n",
                                               array[index + offset], array[i],
                                               entry1[array[index + offset]], entry2[array[index + offset]], entry3[array[index + offset]],
                                               entry1[array[i]], entry2[array[i]], entry3[array[i]]);
                                        //for (int k=0; k<POW_DIM; k++) {
                                        //        printf("isChild: Duplicate NOT a child: children index = %i: child %i == i = %i\n", array[index + offset],
                                        //               k, array[i]);
                                        //
                                        //        printf("isChild: Duplicate NOT a child: children index = %i: child %i == index = %i\n", array[i],
                                        //               k, array[index + offset]);
                                        //
                                        //}
                                    }

                                }

                                //maxIndex = max(index + offset, i);
                                // mark larger index with -1 (thus a duplicate)
                                //array[maxIndex] = -1;
                                //atomicAdd(duplicateCounter, 1);
                            }
                        }

                    }
                }
                offset += stride;
            }
        }

        namespace Launch {
            template<typename T, typename U>
            real markDuplicatesTemp(Tree *tree, DomainList *domainList, T *array, U *entry1, U *entry2, U *entry3, integer *duplicateCounter, integer *child, int length) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::CudaUtils::Kernel::markDuplicatesTemp<T, U>, tree, domainList, array, entry1, entry2,
                                    entry3, duplicateCounter, child, length);
            }
            template real markDuplicatesTemp<integer, real>(Tree *tree, DomainList *domainList, integer *array, real *entry1, real *entry2, real *entry3, integer *duplicateCounter, integer *child, int length);
        }
    }
}