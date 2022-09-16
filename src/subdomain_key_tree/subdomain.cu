#include "../../include/subdomain_key_tree/subdomain.cuh"
#if TARGET_GPU
#include "../../include/cuda_utils/cuda_launcher.cuh"
#include <cub/cub.cuh>
#endif // TARGET_GPU

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
    return subDomainKeyTree->key2proc(key); //, curveType);
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

// ATTENTION: independent of lebesgue/hilbert, thus provide appropriate key!
CUDA_CALLABLE_MEMBER integer SubDomainKeyTree::key2proc(keyType key/*, Curve::Type curveType*/) {

    for (integer proc = 0; proc < numProcesses; proc++) {
        if (key >= range[proc] && key < range[proc + 1]) {
            return proc;
        }
    }
    printf("ERROR: key2proc(k=%lu): -1!\n", key);
    return -1;
}

CUDA_CALLABLE_MEMBER bool SubDomainKeyTree::isDomainListNode(keyType key, integer maxLevel, integer level,
                                                             Curve::Type curveType) {
    integer p1, p2;
    //p1 = key2proc(key);
    //p2 = key2proc(key | ~(~0UL << DIM * (maxLevel - level)));

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
#if DIM == 1
            keyType shiftValue = 1;
            keyType toShift = 21;
            keyType keyMax = (shiftValue << toShift);
#elif DIM == 2
            keyType shiftValue = 1;
            keyType toShift = 42;
            keyType keyMax = (shiftValue << toShift);
#else
            //keyType shiftValue = 1;
            //keyType toShift = 63;
            //keyType keyMax = (shiftValue << toShift);
            keyType keyMax = ~0UL; //KEY_MAX;
#endif
            //p2 = key2proc(hilbert | (KEY_MAX >> (DIM*level+1)));
            p2 = key2proc(hilbert | (keyMax >> (DIM*level+1)));

            //printf("lebesgue: %lu vs %lu < ? : %i\n", key, key | ~(~0UL << DIM * (maxLevel - level)),
            // key < (key | ~(~0UL << DIM * (maxLevel - level))));
            //printf("hilbert: %lu vs %lu < ? : %i\n", KeyNS::lebesgue2hilbert(key, maxLevel, maxLevel),
            // hilbert | (KEY_MAX >> (DIM*level+1)),
            //KeyNS::lebesgue2hilbert(key, maxLevel, maxLevel) < (hilbert | (KEY_MAX >> (DIM*level+1))));

            break;
        }
        default: {
            printf("Curve type not available!\n");
        }
    }
    if (p1 != p2) {
        return true;
    }
    return false;
}

void SubDomainKeyTree::buildDomainTree(Tree *tree, Particles *particles, DomainList *domainList, integer numParticles, Curve::Type curveType) {


    //Logger(TRACE) << "buildDomainTree()";

    *domainList->domainListIndex = 0;
    keyType key;

    for (int i=0; i<POW_DIM; i++) {

#if DIM == 1
        Box box {*tree->minX, *tree->maxX};
        Box sonBox {*tree->minX, *tree->maxX};
#elif DIM == 2
        Box box {*tree->minX, *tree->maxX, *tree->minY, *tree->maxY};
        Box sonBox {*tree->minX, *tree->maxX, *tree->minY, *tree->maxY};
#else
        Box box {*tree->minX, *tree->maxX, *tree->minY, *tree->maxY, *tree->minZ, *tree->maxZ};
        Box sonBox {*tree->minX, *tree->maxX, *tree->minY, *tree->maxY, *tree->minZ, *tree->maxZ};
#endif

        if (i & 1) {
            sonBox.maxX = 0.5 * (box.minX + box.maxX);
        } else {
            sonBox.minX = 0.5 * (box.minX + box.maxX);
        }
#if DIM > 1
        if ((i >> 1) & 1) {
            sonBox.maxY = 0.5 * (box.minY + box.maxY);
            //path -= 2;
        } else {
            sonBox.minY = 0.5 * (box.minY + box.maxY);
        }
#if DIM == 3
        if ((i >> 2) & 1) {
            sonBox.maxZ = 0.5 * (box.minZ + box.maxZ);
        } else {
            sonBox.minZ = 0.5 * (box.minZ + box.maxZ);
        }
#endif
#endif

        key = (keyType)((i * 1UL) << (DIM * (MAX_LEVEL - 1)));
        domainList->domainListKeys[*domainList->domainListIndex] = key;
        domainList->domainListLevels[*domainList->domainListIndex] = 1;

        int nodeIndex = tree->child[i];

        if (nodeIndex < numParticles) {
            if (tree->child[i] == -1) {
                // nothing there, thus create pseudo-particle!
                //Logger(TRACE) << "i: " << i <<"... nothing here create: " << *tree->index;
                tree->child[i] = *tree->index;
                nodeIndex = *tree->index;

            }
            else {
                // particle, thus create pseudo-particle in between!
                int particleIndex = tree->child[i];
                tree->child[i] = *tree->index;
                //(*tree->index)++;

                //Logger(TRACE) << "i: " << i <<"... already pseudo: " << *tree->index;

                //Box box {*tree->minX, *tree->maxX, *tree->minY, *tree->maxY, *tree->minZ, *tree->maxZ};
                int childPath = 0;

                if (particles->x[particleIndex] < 0.5 * (box.minX + box.maxX)) { // x direction
                    childPath += 1;
                    sonBox.maxX = 0.5 * (box.minX + box.maxX);
                    sonBox.minX = box.minX;
                }
                else {
                    sonBox.minX = 0.5 * (box.minX + box.maxX);
                    sonBox.maxX = box.maxX;
                }
#if DIM > 1
                // y direction
                if (particles->y[particleIndex] < 0.5 * (box.minY + box.maxY)) { // y direction
                    childPath += 2;
                    sonBox.maxY = 0.5 * (box.minY + box.maxY);
                    sonBox.minY = box.minY;
                }
                else {
                    sonBox.minY = 0.5 * (box.minY + box.maxY);
                    sonBox.maxY = box.maxY;
                }
#if DIM == 3
                // z direction
                if (particles->z[particleIndex] < 0.5 * (box.minZ + box.maxZ)) {  // z direction
                    childPath += 4;
                    sonBox.maxZ = 0.5 * (box.minZ + box.maxZ);
                    sonBox.minZ = box.minZ;
                }
                else {
                    sonBox.minZ = 0.5 * (box.minZ + box.maxZ);
                    sonBox.maxZ = box.maxZ;
                }
#endif
#endif

                tree->child[POW_DIM * (*tree->index) + childPath] = particleIndex;
                nodeIndex = *tree->index;
            }

            (*tree->index)++;
        }
        //else {
        //    Logger(TRACE) << "i: " << i <<"... good to go: " << nodeIndex;
        //}

        domainList->domainListIndices[*domainList->domainListIndex] = nodeIndex;
        particles->nodeType[nodeIndex] = 1;

        int domainListIndex = *domainList->domainListIndex;
        domainList->borders[domainListIndex * 2 * DIM] = sonBox.minX;
        domainList->borders[domainListIndex * 2 * DIM + 1] = sonBox.maxX;
#if DIM > 1
        domainList->borders[domainListIndex * 2 * DIM + 2] = sonBox.minY;
        domainList->borders[domainListIndex * 2 * DIM + 3] = sonBox.maxY;
#if DIM == 3
        domainList->borders[domainListIndex * 2 * DIM + 4] = sonBox.minZ;
        domainList->borders[domainListIndex * 2 * DIM + 5] = sonBox.maxZ;
#endif
#endif


        (*domainList->domainListIndex)++;

        buildDomainTree(tree, particles, domainList, 2, key, numParticles, nodeIndex, sonBox, curveType);
        //buildDomainTree(tree, particles, domainList, 2, key, numParticles, i, sonBox, curveType);
    }

    //for (int i=0; i<POW_DIM; ++i) {
        //buildDomainTree(tree, domainList, 2, domainList->domainListKeys[i]);
    //}

    //for (int i=0; i< *domainList->domainListIndex; ++i) {
    //    Logger(TRACE) << "domainList[" << i << "] (level: " << domainList->domainListLevels[i] << ")= "
    //    << domainList->domainListKeys[i] << "  " << domainList->domainListIndices[i] << "  type: " << particles->nodeType[domainList->domainListIndices[i]];
    //}

}

// TODO: pass box
void SubDomainKeyTree::buildDomainTree(Tree *tree, Particles *particles, DomainList *domainList, int level, keyType key2test,
                                       integer numParticles, int childIndex, Box &box, Curve::Type curveType) {



    if (isDomainListNode(key2test & (~0UL << (DIM * (MAX_LEVEL - level - 1))),
                                           MAX_LEVEL, level-1, curveType)) {

        //Logger(TRACE) << "-----------";
        //Logger(TRACE) << "Testing key: " << key2test << " | childIndex: " << childIndex << " | level = "
        //              << level << " | " << isDomainListNode(key2test & (~0UL << (DIM * (MAX_LEVEL - level - 1))), MAX_LEVEL, level-1, curveType);

        for (int i=0; i<POW_DIM; ++i) {

            //Logger(TRACE) << "-- " << i;

            domainList->domainListKeys[*domainList->domainListIndex] = key2test | ((i * 1UL) << DIM * (MAX_LEVEL - level)); //key2test & (~0UL << (DIM * (MAX_LEVEL - level - 1))); //key2test;
            domainList->domainListLevels[*domainList->domainListIndex] = level;

            int nodeIndex = tree->child[POW_DIM * childIndex + i];

            Box sonBox;
            sonBox.minX = box.minX;
            sonBox.maxX = box.maxX;
#if DIM > 1
            sonBox.minY = box.minY;
            sonBox.maxY = box.maxY;
#if DIM == 3
            sonBox.minZ = box.minZ;
            sonBox.maxZ = box.maxZ;
#endif
#endif

            if (i & 1) {
                sonBox.maxX = 0.5 * (box.minX + box.maxX);
            } else {
                sonBox.minX = 0.5 * (box.minX + box.maxX);
            }
#if DIM > 1
            if ((i >> 1) & 1) {
                sonBox.maxY = 0.5 * (box.minY + box.maxY);
                //path -= 2;
            } else {
                sonBox.minY = 0.5 * (box.minY + box.maxY);
            }
#if DIM == 3
            if ((i >> 2) & 1) {
                sonBox.maxZ = 0.5 * (box.minZ + box.maxZ);
            } else {
                sonBox.minZ = 0.5 * (box.minZ + box.maxZ);
            }
#endif
#endif
            if (nodeIndex < numParticles) {
                if (nodeIndex == -1) {
                    // nothing there, thus create pseudo-particle!
                    //Logger(TRACE) << "nothing here for " << childIndex << "and " << i << ": create pseudo-particle: tree->child[8 * " << childIndex << " + " << i << "] = " << *tree->index;
                    tree->child[POW_DIM * childIndex + i] = *tree->index;
                    nodeIndex = *tree->index;

                }
                else {
                    // particle, thus create pseudo-particle in between!
                    int particleIndex = tree->child[POW_DIM * childIndex + i];
                    tree->child[POW_DIM * childIndex + i] = *tree->index;
                    //(*tree->index)++;

                    //Box box {*tree->minX, *tree->maxX, *tree->minY, *tree->maxY, *tree->minZ, *tree->maxZ};
                    int childPath = 0;

                    if (particles->x[particleIndex] < 0.5 * (box.minX + box.maxX)) { // x direction
                        childPath += 1;
                        sonBox.maxX = 0.5 * (box.minX + box.maxX);
                        sonBox.minX = box.minX;
                    }
                    else {
                        sonBox.minX = 0.5 * (box.minX + box.maxX);
                        sonBox.maxX = box.maxX;
                    }
#if DIM > 1
                    // y direction
                    if (particles->y[particleIndex] < 0.5 * (box.minY + box.maxY)) { // y direction
                        childPath += 2;
                        sonBox.maxY = 0.5 * (box.minY + box.maxY);
                        sonBox.minY = box.minY;
                    }
                    else {
                        sonBox.minY = 0.5 * (box.minY + box.maxY);
                        sonBox.maxY = box.maxY;
                    }
#if DIM == 3
                    // z direction
                    if (particles->z[particleIndex] < 0.5 * (box.minZ + box.maxZ)) {  // z direction
                        childPath += 4;
                        sonBox.maxZ = 0.5 * (box.minZ + box.maxZ);
                        sonBox.minZ = box.minZ;
                    }
                    else {
                        sonBox.minZ = 0.5 * (box.minZ + box.maxZ);
                        sonBox.maxZ = box.maxZ;
                    }
#endif
#endif

                    //Logger(TRACE) << "particle here child[8 * " << childIndex << " + " << i << " = " << nodeIndex << "create pseudo-particle: tree->child[8 * " << childIndex << " + " << i << "] = " << *tree->index;
                    //Logger(TRACE) << "    and: insert particle at          : tree->child[8 * " << (*tree->index) << " + " << childPath << "] = " << particleIndex;

                    tree->child[POW_DIM * (*tree->index) + childPath] = particleIndex;
                    nodeIndex = *tree->index;
                }

                (*tree->index)++;
            }
            //else {
            //    //Logger(TRACE) << "already a pseudo-particle here: " << childIndex;
            //}

            //Logger(TRACE) << "particles->nodeType[" << nodeIndex << "] = 1 (numParticles = " << numParticles << ")";

            domainList->domainListIndices[*domainList->domainListIndex] = nodeIndex; //childIndex;
            int domainListIndex = *domainList->domainListIndex;
            domainList->borders[domainListIndex * 2 * DIM] = sonBox.minX;
            domainList->borders[domainListIndex * 2 * DIM + 1] = sonBox.maxX;
#if DIM > 1
            domainList->borders[domainListIndex * 2 * DIM + 2] = sonBox.minY;
            domainList->borders[domainListIndex * 2 * DIM + 3] = sonBox.maxY;
#if DIM == 3
            domainList->borders[domainListIndex * 2 * DIM + 4] = sonBox.minZ;
            domainList->borders[domainListIndex * 2 * DIM + 5] = sonBox.maxZ;
#endif
#endif
            (*domainList->domainListIndex)++;
            particles->nodeType[nodeIndex] = 1;

            buildDomainTree(tree, particles, domainList, level + 1, key2test + ((i * 1UL) << DIM * (MAX_LEVEL - level)),
                            numParticles, nodeIndex, sonBox, curveType);
        }

    }

}

/*
void SubDomain::createDomainList(TreeNode &t, int level, keyType k) {
    t.node = TreeNode::domainList;

    //Logger(INFO) << "keyType k = " << k;

    int proc1;
    int proc2;
    if (curve == hilbert) {
        keyType hilbert = keyType::Lebesgue2Hilbert(k, level);
        proc1 = key2proc(hilbert, level, true);
        proc2 = key2proc(hilbert | (keyType{ keyType::KEY_MAX } >> (DIM * level + 1)), level, true);
    }
    else {
        proc1 = key2proc(k, level);
        proc2 = key2proc(k | ~(~keyType(0L) << DIM * (k.maxLevel - level)), level);
    }
    if (proc1 != proc2) {
        for (int i=0; i<POWDIM; i++) {
            if (t.son[i] == NULL) {
                t.son[i] = new TreeNode;
            }
            else if (t.son[i]->isLeaf() && t.son[i]->node == TreeNode::particle) {
                t.son[i]->node = TreeNode::domainList;
                t.insert(t.son[i]->p);
                continue;
            }
            createDomainList(*t.son[i], level + 1,
                             k | (keyType{ i } << (DIM*(k.maxLevel-level-1))));
        }
    }
}
*/

#if TARGET_GPU
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

        // "serial" version
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
                childIndex = 0;
                // iterate through levels (of corresponding domainListIndex)
                for (int j = 0; j < domainList->domainListLevels[i]; j++) {
                    path[j] = (integer) (domainList->domainListKeys[i] >> (MAX_LEVEL * DIM - DIM * (j + 1)) &
                                         (integer)(POW_DIM - 1));
                    temp = childIndex;
                    childIndex = tree->child[POW_DIM*childIndex+path[j]];//tree->child[POW_DIM*childIndex + path[j]];
                    if (childIndex < n) {
                        if (childIndex == -1) {
                            // no child at all here, thus add node
                            integer cell = atomicAdd(tree->index, 1);
                            tree->child[POW_DIM * temp + path[j]] = cell;
                            particles->level[cell] = j + 1; //particles->level[temp] + 1;
                            childIndex = cell;
                            domainList->domainListIndices[domainListCounter] = childIndex; //cell;
                            particles->nodeType[childIndex] = 1;
#if DEBUGGING
#if DIM == 3
                            printf("adding node index %i  cell = %i (childPath = %i,  j = %i)! x = (%e, %e, %e) level = %i\n",
                                   temp, cell, path[j], j, particles->x[childIndex], particles->y[childIndex], particles->z[childIndex],
                                   particles->level[cell]);
#endif
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

                            //particles->x[cell] += particles->weightedEntry(childIndex, Entry::x);
                            particles->x[cell] = particles->x[childIndex];
#if DIM > 1
                            //particles->y[cell] += particles->weightedEntry(childIndex, Entry::y);
                            particles->y[cell] = particles->y[childIndex];
#if DIM == 3
                            //particles->z[cell] += particles->weightedEntry(childIndex, Entry::z);
                            particles->z[cell] = particles->z[childIndex];
#endif
#endif
                            //particles->mass[cell] += particles->mass[childIndex];
                            particles->mass[cell] = particles->mass[childIndex];

                            particles->level[cell] = particles->level[childIndex];
                            particles->level[childIndex] += 1;
#if DEBUGGING
#if DIM == 3
                            printf("adding node in between for index %i, temp = %i  cell = %i (childPath = %i,  j = %i)! x = (%e, %e, %e) level = %i vs %i\n",
                                   childIndex, temp, cell, childPath, j, particles->x[childIndex], particles->y[childIndex], particles->z[childIndex],
                                   particles->level[cell], particles->level[childIndex]);
#endif
#endif

                            tree->child[POW_DIM * cell + childPath] = childIndex;

                            childIndex = cell;
                            domainList->domainListIndices[domainListCounter] = childIndex; //temp;
                            particles->nodeType[childIndex] = 1;
                            domainListCounter++;
                        }
                    }
                    else {
                        insert = true;
                        // check whether node already marked as domain list node
                        if (particles->nodeType[childIndex] >= 1) {
                            insert = false;
                        }
                        //for (int k=0; k<domainListCounter; k++) {
                        //    if (childIndex == domainList->domainListIndices[k]) {
                        //        insert = false;
                        //        break;
                        //    }
                        //}
                        if (insert) {
                            // mark/save node as domain list node
                            domainList->domainListIndices[domainListCounter] = childIndex; //temp;
                            particles->nodeType[childIndex] = 1;
                            domainListCounter++;
                        }
                    }
                }
            }
        }

        /*
        // parallel version
        __global__ void buildDomainTree(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                        DomainList *domainList, integer n, integer m, integer level) {

            integer index = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            integer domainListCounter;

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
            int j;

            // loop over domain list indices (over the keys found/generated by createDomainListKernel)
            while ((index + offset) < *domainList->domainListIndex) {

                for (j = 0; j < MAX_LEVEL; j++) {
                    path[j] = 0;
                }
                    if (domainList->domainListLevels[index + offset] == level) {
                    //printf("domainListKey[%i] = %lu\n", i, domainList->domainListKeys[i]);
                    childIndex = 0;
                    temp = 0;
                    // iterate through levels (of corresponding domainListIndex)
                    for (j = 0; j < domainList->domainListLevels[index + offset]; j++) {
                        path[j] = (integer) (
                                domainList->domainListKeys[index + offset] >> (MAX_LEVEL * DIM - DIM * (j + 1)) &
                                (integer) (POW_DIM - 1));
                        temp = childIndex;
                        childIndex = tree->child[POW_DIM * childIndex + path[j]];
                    }
                    if (childIndex < n) {
                        if (childIndex == -1) {
                            // no child at all here, thus add node
                            printf("build domain tree, adding node ...\n");
                            integer cell = atomicAdd(tree->index, 1);
                            tree->child[POW_DIM * temp + path[j-1]] = cell;
                            particles->level[cell] = j;
                            childIndex = cell;
                            //domainListCounter = atomicAdd(domainList->domainListCounter, 1);
                            domainList->domainListIndices[index + offset] = childIndex; //cell;
                            particles->nodeType[childIndex] = 1;

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

                            // new
                            int k;
                            for (k = 0; k < j; k++) {

                                currentChild = path[k];
                                //printf("[rank %i] currentChild: %i\n", subDomainKeyTree->rank, currentChild);
                                if (currentChild & 1) {
                                    max_x = 0.5 * (min_x + max_x);
                                    //path -= 1;
                                } else {
                                    min_x = 0.5 * (min_x + max_x);
                                }
#if DIM > 1
                                if ((currentChild >> 1) & 1) {
                                    //if (path % 2 == 0 && path % 4 != 0) {
                                    max_y = 0.5 * (min_y + max_y);
                                    //path -= 2;
                                } else {
                                    min_y = 0.5 * (min_y + max_y);
                                }
#if DIM == 3
                                if ((currentChild >> 2) & 1) {
                                    //if (path == 4) {
                                    max_z = 0.5 * (min_z + max_z);
                                    //path -= 4;
                                } else {
                                    min_z = 0.5 * (min_z + max_z);
                                }
#endif
#endif

                                //if (currentChild % 2 != 0) {
                                //    max_x = 0.5 * (min_x + max_x);
                                //    currentChild -= 1;
                                //} else {
                                //    min_x = 0.5 * (min_x + max_x);
                                //}
                                //#if DIM > 1
                                //if (currentChild % 2 == 0 && currentChild % 4 != 0) {
                                //    max_y = 0.5 * (min_y + max_y);
                                //    currentChild -= 2;
                                //} else {
                                //    min_y = 0.5 * (min_y + max_y);
                                //}
                                //#if DIM == 3
                                //if (currentChild == 4) {
                                //    max_z = 0.5 * (min_z + max_z);
                                //    currentChild -= 4;
                                //} else {
                                //    min_z = 0.5 * (min_z + max_z);
                                //}
                                //#endif
                                //#endif
                            //}

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
                            // end: new
#if DEBUGGING
#if DIM == 3
                            printf("adding node index %i  cell = %i (childPath = %i,  j = %i)! x = (%e, %e, %e) level = %i\n",
                                   temp, cell, path[j], j, particles->x[childIndex], particles->y[childIndex], particles->z[childIndex],
                                   particles->level[cell]);
#endif
#endif
#if DIM == 3
#if DEBUGGING
                            printf("[rank %i] adding domainListIndices[%i] = %i, childIndex = %i, path = %i\n", subDomainKeyTree->rank,
                                   index + offset, domainList->domainListIndices[index + offset], childIndex, path[j-1]);
                            //printf("adding node index %i  cell = %i (childPath = %i,  j = %i)! x = (%f, %f, %f)\n",
                            //       childIndex, cell, path[j], j, particles->x[childIndex], particles->y[childIndex],
                            //       particles->z[childIndex]);
#endif
#endif
                        } else {
                            printf("build domain tree, adding node in between...\n");
                            // child is a leaf, thus add node in between
                            integer cell = atomicAdd(tree->index, 1);
                            tree->child[POW_DIM * temp + path[j - 1]] = cell;

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
                            int k;
                            for (k = 0; k < j; k++) {

                                currentChild = path[k];

                                //if (currentChild % 2 != 0) {
                                if (currentChild & 1) {
                                    max_x = 0.5 * (min_x + max_x);
                                    //currentChild -= 1;
                                } else {
                                    min_x = 0.5 * (min_x + max_x);
                                }
#if DIM > 1
                                //if (currentChild % 2 == 0 && currentChild % 4 != 0) {
                                if ((currentChild >> 1) & 1) {
                                    max_y = 0.5 * (min_y + max_y);
                                    //currentChild -= 2;
                                } else {
                                    min_y = 0.5 * (min_y + max_y);
                                }
#if DIM == 3
                                //if (currentChild == 4) {
                                if ((currentChild >> 2) & 1) {
                                    max_z = 0.5 * (min_z + max_z);
                                    //currentChild -= 4;
                                } else {
                                    min_z = 0.5 * (min_z + max_z);
                                }
#endif
#endif
                            }

                            //TODO: was particles->x[cell] = 0.5 * (min_x + max_x);
                            particles->x[cell] = particles->x[childIndex];
                            //particles->x[cell] = 0.5 * (min_x + max_x);
#if DIM > 1
                            particles->y[cell] = particles->y[childIndex];
                            //particles->y[cell] = 0.5 * (min_y + max_y);
#if DIM == 3
                            particles->z[cell] = particles->z[childIndex];
                            //particles->z[cell] = 0.5 * (min_z + max_z);
#endif
#endif
                            //particles->mass[cell] = particles->mass[childIndex];


                            // //particles->x[cell] += particles->weightedEntry(childIndex, Entry::x);
                            //particles->x[cell] = particles->x[childIndex];
                            //#if DIM > 1
                            // //particles->y[cell] += particles->weightedEntry(childIndex, Entry::y);
                            //particles->y[cell] = particles->y[childIndex];
                            //#if DIM == 3
                            // //particles->z[cell] += particles->weightedEntry(childIndex, Entry::z);
                            //particles->z[cell] = particles->z[childIndex];
                            //#endif
                            //#endif
                            // //particles->mass[cell] += particles->mass[childIndex];
                            //particles->mass[cell] = particles->mass[childIndex];

                            particles->level[cell] = particles->level[childIndex];
                            //particles->level[childIndex] += 1;
                            atomicAdd(&particles->level[childIndex], 1);

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



#if DEBUGGING
#if DIM == 3
                            printf("adding node in between for index %i, temp = %i  cell = %i (childPath = %i,  j = %i)! x = (%e, %e, %e) level = %i vs %i\n",
                                   childIndex, temp, cell, childPath, j, particles->x[childIndex], particles->y[childIndex], particles->z[childIndex],
                                   particles->level[cell], particles->level[childIndex]);
#endif
#endif


                            tree->child[POW_DIM * cell + childPath] = childIndex;
                            printf("child[8 * %i + %i] = %i\n", cell, childPath, childIndex);

                            //particles->nodeType[childIndex] = -10; // just some testing

                            childIndex = cell;
                            //domainListCounter = atomicAdd(domainList->domainListCounter, 1);
                            domainList->domainListIndices[index + offset] = childIndex; //temp;
                            particles->nodeType[childIndex] = 1;
                        }
                    } else {
                        // mark/save node as domain list node
                        //domainListCounter = atomicAdd(domainList->domainListCounter, 1);
#if DEBUGGING
                        printf("[rank %i] Mark already available node %i: %i, path = %i (level = %i)\n",
                               subDomainKeyTree->rank, index + offset, childIndex, path[j-1], level);
#endif
                        domainList->domainListIndices[index + offset] = childIndex; //temp;
                        particles->nodeType[childIndex] = 1;

                    }
                    __threadfence();
                }
                __syncthreads();

                offset += stride;
            }
        }*/

        // parallel version
        __global__ void buildDomainTree(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                        DomainList *domainList, integer n, integer m, integer level) {

            integer index = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            integer domainListCounter;

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
            int j;

            // loop over domain list indices (over the keys found/generated by createDomainListKernel)
            while ((index + offset) < *domainList->domainListIndex) {

                #pragma unroll
                for (j = 0; j < MAX_LEVEL; j++) {
                    path[j] = 0;
                }
                if (domainList->domainListLevels[index + offset] == level) {
                    //printf("domainListKey[%i] = %lu\n", i, domainList->domainListKeys[i]);
                    childIndex = 0;
                    temp = 0;

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

                    // iterate through levels (of corresponding domainListIndex)
                    for (j = 0; j < domainList->domainListLevels[index + offset]; j++) {
                        path[j] = (integer) (
                                domainList->domainListKeys[index + offset] >> (MAX_LEVEL * DIM - DIM * (j + 1)) &
                                (integer) (POW_DIM - 1));
                        temp = childIndex;
                        childIndex = tree->child[POW_DIM * childIndex + path[j]];
                    }

                    int k;
                    for (k = 0; k < j; k++) {

                        currentChild = path[k];
                        if (currentChild & 1) {
                            max_x = 0.5 * (min_x + max_x);
                        } else {
                            min_x = 0.5 * (min_x + max_x);
                        }
#if DIM > 1
                        if ((currentChild >> 1) & 1) {
                            max_y = 0.5 * (min_y + max_y);
                            //path -= 2;
                        } else {
                            min_y = 0.5 * (min_y + max_y);
                        }
#if DIM == 3
                        if ((currentChild >> 2) & 1) {
                            max_z = 0.5 * (min_z + max_z);
                        } else {
                            min_z = 0.5 * (min_z + max_z);
                        }
#endif
#endif
                    }

                    if (childIndex < n) {
                        if (childIndex == -1) {
                            // no child at all here, thus add node
                            //printf("build domain tree, adding node ...\n");
                            integer cell = atomicAdd(tree->index, 1);
                            tree->child[POW_DIM * temp + path[j-1]] = cell;
                            particles->level[cell] = j;
                            childIndex = cell;

                            domainList->domainListIndices[index + offset] = childIndex;
                            particles->nodeType[childIndex] = 1;

                            domainList->borders[(index + offset) * 2 * DIM] = min_x;
                            domainList->borders[(index + offset) * 2 * DIM + 1] = max_x;
#if DIM > 1
                            domainList->borders[(index + offset) * 2 * DIM + 2] = min_y;
                            domainList->borders[(index + offset) * 2 * DIM + 3] = max_y;
#if DIM == 3
                            domainList->borders[(index + offset) * 2 * DIM + 4] = min_z;
                            domainList->borders[(index + offset) * 2 * DIM + 5] = max_z;
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
                            // end: new
#if DEBUGGING
                            #if DIM == 3
                            printf("adding node index %i  cell = %i (childPath = %i,  j = %i)! x = (%e, %e, %e) level = %i\n",
                                   temp, cell, path[j], j, particles->x[childIndex], particles->y[childIndex], particles->z[childIndex],
                                   particles->level[cell]);
#endif
#endif
#if DIM == 3
#if DEBUGGING
                            printf("[rank %i] adding domainListIndices[%i] = %i, childIndex = %i, path = %i\n", subDomainKeyTree->rank,
                                   index + offset, domainList->domainListIndices[index + offset], childIndex, path[j-1]);
                            //printf("adding node index %i  cell = %i (childPath = %i,  j = %i)! x = (%f, %f, %f)\n",
                            //       childIndex, cell, path[j], j, particles->x[childIndex], particles->y[childIndex],
                            //       particles->z[childIndex]);
#endif
#endif
                        } else {
                            //printf("build domain tree, adding node in between...\n");
                            // child is a leaf, thus add node in between
                            integer cell = atomicAdd(tree->index, 1);
                            tree->child[POW_DIM * temp + path[j - 1]] = cell;

                            //TODO: was particles->x[cell] = 0.5 * (min_x + max_x);
                            particles->x[cell] = particles->x[childIndex];
                            //particles->x[cell] = 0.5 * (min_x + max_x);
#if DIM > 1
                            particles->y[cell] = particles->y[childIndex];
                            //particles->y[cell] = 0.5 * (min_y + max_y);
#if DIM == 3
                            particles->z[cell] = particles->z[childIndex];
                            //particles->z[cell] = 0.5 * (min_z + max_z);
#endif
#endif

                            domainList->borders[(index + offset) * 2 * DIM] = min_x;
                            domainList->borders[(index + offset) * 2 * DIM + 1] = max_x;
#if DIM > 1
                            domainList->borders[(index + offset) * 2 * DIM + 2] = min_y;
                            domainList->borders[(index + offset) * 2 * DIM + 3] = max_y;
#if DIM == 3
                            domainList->borders[(index + offset) * 2 * DIM + 4] = min_z;
                            domainList->borders[(index + offset) * 2 * DIM + 5] = max_z;
#endif
#endif

                            particles->level[cell] = particles->level[childIndex];
                            //particles->level[childIndex] += 1;
                            atomicAdd(&particles->level[childIndex], 1);

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



#if DEBUGGING
                            #if DIM == 3
                            printf("adding node in between for index %i, temp = %i  cell = %i (childPath = %i,  j = %i)! x = (%e, %e, %e) level = %i vs %i\n",
                                   childIndex, temp, cell, childPath, j, particles->x[childIndex], particles->y[childIndex], particles->z[childIndex],
                                   particles->level[cell], particles->level[childIndex]);
#endif
#endif


                            tree->child[POW_DIM * cell + childPath] = childIndex;
                            //printf("child[8 * %i + %i] = %i\n", cell, childPath, childIndex);

                            //particles->nodeType[childIndex] = -10; // just some testing

                            childIndex = cell;
                            //domainListCounter = atomicAdd(domainList->domainListCounter, 1);
                            domainList->domainListIndices[index + offset] = childIndex; //temp;
                            particles->nodeType[childIndex] = 1;
                        }
                    } else {
                        // mark/save node as domain list node
                        //domainListCounter = atomicAdd(domainList->domainListCounter, 1);
#if DEBUGGING
                        printf("[rank %i] Mark already available node %i: %i, path = %i (level = %i)\n",
                               subDomainKeyTree->rank, index + offset, childIndex, path[j-1], level);
#endif

                        domainList->borders[(index + offset) * 2 * DIM] = min_x;
                        domainList->borders[(index + offset) * 2 * DIM + 1] = max_x;
#if DIM > 1
                        domainList->borders[(index + offset) * 2 * DIM + 2] = min_y;
                        domainList->borders[(index + offset) * 2 * DIM + 3] = max_y;
#if DIM == 3
                        domainList->borders[(index + offset) * 2 * DIM + 4] = min_z;
                        domainList->borders[(index + offset) * 2 * DIM + 5] = max_z;
#endif
#endif
                        domainList->domainListIndices[index + offset] = childIndex; //temp;
                        particles->nodeType[childIndex] = 1;

                    }
                    __threadfence();
                }
                __syncthreads();

                offset += stride;
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
                    particles->x[domainIndex] = (real)0;
#if DIM > 1
                    particles->y[domainIndex] = (real)0;
#if DIM == 3
                    particles->z[domainIndex] = (real)0;
#endif
#endif
                    particles->mass[domainIndex] = (real)0;
                }
                else {
                //    //printf("domainIndex = %i *= mass = %f\n", domainIndex, particles->mass[domainIndex]);
                    particles->x[domainIndex] *= particles->mass[domainIndex];
#if DIM > 1
                    particles->y[domainIndex] *= particles->mass[domainIndex];
#if DIM == 3
                    particles->z[domainIndex] *= particles->mass[domainIndex];
#endif
#endif
                }

                offset += stride;
            }
        }

        template <typename T>
        __global__ void prepareLowestDomainExchange(Particles *particles, DomainList *lowestDomainList,
                                                    T *buffer, Entry::Name entry) {

            integer bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
            integer stride = blockDim.x*gridDim.x;
            integer offset = 0;
            integer lowestDomainIndex;

            //copy x, y, z, mass of lowest domain list nodes into arrays
            //sorting using cub (not here)
            while ((bodyIndex + offset) < *lowestDomainList->domainListIndex) {
                lowestDomainIndex = lowestDomainList->domainListIndices[bodyIndex + offset];
                if (lowestDomainIndex >= 0) {
                    switch (entry) {
                        case Entry::x: {
                            buffer[bodyIndex + offset] = particles->x[lowestDomainIndex];
                        } break;
#if DIM > 1
                        case Entry::y: {
                            buffer[bodyIndex + offset] = particles->y[lowestDomainIndex];
                        } break;
#if DIM == 3
                        case Entry::z: {
                            buffer[bodyIndex + offset] = particles->z[lowestDomainIndex];
                        } break;
#endif
#endif
                        case Entry::mass: {
                            buffer[bodyIndex + offset] = particles->mass[lowestDomainIndex];
                        } break;
                        default:
                            printf("prepareLowestDomainExchange(): Not available!\n");
                    }
                }
                offset += stride;
            }
        }

        template <typename T>
        __global__ void updateLowestDomainListNodes(Particles *particles, DomainList *lowestDomainList,
                                                    T *buffer, Entry::Name entry) {

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
                                buffer[bodyIndex + offset];
                    } break;
#if DIM > 1
                    case Entry::y: {
                        particles->y[lowestDomainList->domainListIndices[originalIndex]] =
                                buffer[bodyIndex + offset];
                    } break;
#if DIM == 3
                    case Entry::z: {
                        particles->z[lowestDomainList->domainListIndices[originalIndex]] =
                                buffer[bodyIndex + offset];
                    } break;
#endif
#endif
                    case Entry::mass: {
                        particles->mass[lowestDomainList->domainListIndices[originalIndex]] =
                                buffer[bodyIndex + offset];
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

                    particles->x[lowestDomainIndex] /= particles->mass[lowestDomainIndex];
#if DIM > 1
                    particles->y[lowestDomainIndex] /= particles->mass[lowestDomainIndex];
#if DIM == 3
                    particles->z[lowestDomainIndex] /= particles->mass[lowestDomainIndex];
#endif
#endif

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
                //        }
                //        break;
                //    }
                //}
                if (compute && domainList->domainListLevels[bodyIndex + offset] == level) {
                    // do the calculation
                    /*
                    particles->x[domainIndex] += 0.;
#if DIM > 1
                    particles->y[domainIndex] += 0.;
#if DIM == 3
                    particles->z[domainIndex] += 0.;
#endif
#endif
                    particles->mass[domainIndex] += 0.;
                    */
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

                    if (particles->mass[domainIndex] > 0.) {
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

        // delete the children of (lowest) domain list nodes if
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
            // alternatively:
            //while ((bodyIndex + offset) < *lowestDomainList->domainListIndex) {
            //    domainIndex = lowestDomainList->domainListIndices[bodyIndex + offset];
            //    //key = tree->getParticleKey(particles, domainIndex, MAX_LEVEL, curveType); // working version
            //    //proc = subDomainKeyTree->key2proc(key);
            //    // //printf("[rank %i] deleting: proc = %i\n", subDomainKeyTree->rank, proc);
            //    //if (proc != subDomainKeyTree->rank) {
            //    //    for (int i=0; i<POW_DIM; i++) {
            //    //        //printf("[rank %i] deleting: POWDIM * %i + %i = %i\n", subDomainKeyTree->rank, domainIndex, i, tree->child[POW_DIM * domainIndex + i]);
            //    //        tree->child[POW_DIM * domainIndex + i] = -1;
            //    //    }
            //    //}
            //    offset += stride;
            //}

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

            offset = tree->toDeleteNode[0];
            //delete inserted cells
            while ((bodyIndex + offset) >= tree->toDeleteNode[0] && (bodyIndex + offset) < tree->toDeleteNode[1]) {
                for (int i=0; i<POW_DIM; i++) {
                    tree->child[(bodyIndex + offset)*POW_DIM + i] = -1;
                }
                tree->count[bodyIndex + offset] = 0;
                particles->x[bodyIndex + offset] = 0.;
                //particles->vx[bodyIndex + offset] = 0.;
                //particles->ax[bodyIndex + offset] = 0.;
#if DIM > 1
                particles->y[bodyIndex + offset] = 0.;
                //particles->vy[bodyIndex + offset] = 0.;
                //particles->ay[bodyIndex + offset] = 0.;
#if DIM == 3
                particles->z[bodyIndex + offset] = 0.;
                //particles->vz[bodyIndex + offset] = 0.;
                //particles->az[bodyIndex + offset] = 0.;
#endif
#endif
                particles->mass[bodyIndex + offset] = 0.;
                //tree->start[bodyIndex + offset] = -1;
                //tree->sorted[bodyIndex + offset] = 0;

                offset += stride;
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

        __global__ void keyHistCounter(Tree *tree, Particles *particles, SubDomainKeyTree *subDomainKeyTree,
                                       Helper *helper, /*keyType *keyHistRanges, integer *keyHistCounts,*/
                                       int bins, int n, Curve::Type curveType) {

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
                                          /*keyType *keyHistRanges, integer *keyHistCounts,*/
                                          int bins, int n, Curve::Type curveType) {

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

                        subDomainKeyTree->range[i] = (helper->keyTypeBuffer[bodyIndex + offset] >> (1*DIM)) << (1*DIM);
                    }
                }
                //printf("[rank %i] keyHistCounts[%i] = %i\n", s->rank, bodyIndex+offset, keyHistCounts[bodyIndex+offset]);
                atomicAdd(helper->integerVal, helper->integerBuffer[bodyIndex+offset]);
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
            // is there any possibility to call kernel with more than one thread?: YES see corresponding function below
            ExecutionPolicy executionPolicy(1, 1);
            return cuda::launch(true, executionPolicy, ::SubDomainKeyTreeNS::Kernel::buildDomainTree, tree, particles,
                         domainList, n, m);
        }

        real Launch::buildDomainTree(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles, DomainList *domainList, integer n, integer m, integer level) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SubDomainKeyTreeNS::Kernel::buildDomainTree,
                                subDomainKeyTree, tree, particles,
                                domainList, n, m, level);
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

        real Launch::zeroDomainListNodes(Particles *particles, DomainList *domainList, DomainList *lowestDomainList) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SubDomainKeyTreeNS::Kernel::zeroDomainListNodes, particles, domainList,
                                lowestDomainList);
        }

        template <typename T>
        real Launch::prepareLowestDomainExchange(Particles *particles, DomainList *lowestDomainList,
                                                 T *buffer, Entry::Name entry) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SubDomainKeyTreeNS::Kernel::prepareLowestDomainExchange, particles,
                                lowestDomainList, buffer, entry);
        }

        template real Launch::prepareLowestDomainExchange<real>(Particles *particles, DomainList *lowestDomainList,
                real *buffer, Entry::Name entry);

        template <typename T>
        real Launch::updateLowestDomainListNodes(Particles *particles, DomainList *lowestDomainList,
                                                    T *buffer, Entry::Name entry) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SubDomainKeyTreeNS::Kernel::updateLowestDomainListNodes, particles,
                                lowestDomainList, buffer, entry);

        }

        template real Launch::updateLowestDomainListNodes<real>(Particles *particles, DomainList *lowestDomainList,
                real *buffer, Entry::Name entry);

        real Launch::compLowestDomainListNodes(Tree *tree, Particles *particles, DomainList *lowestDomainList) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SubDomainKeyTreeNS::Kernel::compLowestDomainListNodes, tree, particles,
                                lowestDomainList);
        }

        real Launch::compLocalPseudoParticles(Tree *tree, Particles *particles, DomainList *domainList, int n) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SubDomainKeyTreeNS::Kernel::compLocalPseudoParticles, tree, particles,
                                domainList, n);
        }

        real Launch::compDomainListPseudoParticlesPerLevel(Tree *tree, Particles *particles, DomainList *domainList,
                                                           DomainList *lowestDomainList, int n, int level) {
            ExecutionPolicy executionPolicy(256, 1);
            return cuda::launch(true, executionPolicy, ::SubDomainKeyTreeNS::Kernel::compDomainListPseudoParticlesPerLevel, tree,
                                particles, domainList, lowestDomainList, n, level);
        }

        real Launch::compDomainListPseudoParticles(Tree *tree, Particles *particles, DomainList *domainList,
                                                   DomainList *lowestDomainList, int n) {
            ExecutionPolicy executionPolicy(256, 1);
            return cuda::launch(true, executionPolicy, ::SubDomainKeyTreeNS::Kernel::compDomainListPseudoParticles, tree,
                                particles, domainList, lowestDomainList, n);
        }

        real Launch::repairTree(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                DomainList *domainList, DomainList *lowestDomainList,
                                int n, int m, Curve::Type curveType) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SubDomainKeyTreeNS::Kernel::repairTree, subDomainKeyTree, tree,
                                particles, domainList, lowestDomainList, n, m, curveType);
        }

        real Launch::createKeyHistRanges(Helper *helper, integer bins) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SubDomainKeyTreeNS::Kernel::createKeyHistRanges, helper, bins);
        }

        real Launch::keyHistCounter(Tree *tree, Particles *particles, SubDomainKeyTree *subDomainKeyTree,
                                    Helper *helper, int bins, int n, Curve::Type curveType) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SubDomainKeyTreeNS::Kernel::keyHistCounter, tree, particles,
                                subDomainKeyTree, helper, bins, n, curveType);
        }

        real Launch::calculateNewRange(SubDomainKeyTree *subDomainKeyTree, Helper *helper, int bins, int n,
                                       Curve::Type curveType) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SubDomainKeyTreeNS::Kernel::calculateNewRange, subDomainKeyTree, helper,
                                bins, n, curveType);
        }


    }

}

#endif // TARGET_GPU

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

CUDA_CALLABLE_MEMBER void DomainList::setBorders(real *borders, integer *relevantDomainListOriginalIndex) {
    this->borders = borders;
    this->relevantDomainListOriginalIndex = relevantDomainListOriginalIndex;
}

#if TARGET_GPU
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

        __global__ void setBorders(DomainList *domainList, real *borders, integer *relevantDomainListOriginalIndex) {
            domainList->setBorders(borders, relevantDomainListOriginalIndex);
        }

        __global__ void info(Particles *particles, DomainList *domainList) {

            integer index = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            integer domainListIndex;

            //if (index == 0) {
            //    printf("domainListIndices = [");
            //    for (int i=0; i<*domainList->domainListIndex; i++) {
            //        printf("%i, ", domainList->domainListIndices[i]);
            //    }
            //    printf("]\n");
            //}

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

            //if (index == 0) {
            //    while ((index + offset) < *domainList->domainListIndex) {
            //        domainListIndex = domainList->domainListIndices[index + offset];
            //    }
            //}

        }

        __global__ void info(Particles *particles, DomainList *domainList, DomainList *lowestDomainList) {

            integer index = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            integer domainListIndex;

            //if (index == 0) {
            //    printf("domainListIndices = [");
            //    for (int i=0; i<*domainList->domainListIndex; i++) {
            //        printf("%i, ", domainList->domainListIndices[i]);
            //    }
            //    printf("]\n");
            //}

            bool show;

            while ((index + offset) < *domainList->domainListIndex) {

                show = true;
                domainListIndex = domainList->domainListIndices[index + offset];

                //for (int i=0; i<*lowestDomainList->domainListIndex; i++) {
                //    if (lowestDomainList->domainListIndices[i] == domainListIndex) {
                //        printf("domainListIndices[%i] = %i, x = (%f, %f, %f) mass = %f\n", index + offset,
                //               domainListIndex, particles->x[domainListIndex],
                //               particles->y[domainListIndex], particles->z[domainListIndex], particles->mass[domainListIndex]);
                //    }
                //}

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

        /*
         * //TODO: parallel version
         *  * either on CPU
         *  * or distribute key2test = 0UL < keyMax between threads
         */
        /*
        __global__ void createDomainList(SubDomainKeyTree *subDomainKeyTree, DomainList *domainList,
                                         integer maxLevel, Curve::Type curveType) {

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
            //keyType keyMax = KEY_MAX;
#endif

            keyType key2test = 0UL;
            integer level = 1;

            // in principle: traversing a (non-existent) octree by walking the 1D spacefilling curve (keys of the tree nodes)
            while (key2test < keyMax) { // TODO: createDomainList(): key2test < or <= keyMax
                if (subDomainKeyTree->isDomainListNode(key2test & (~0UL << (DIM * (maxLevel - level + 1))),
                                                       maxLevel, level-1, curveType)) {
                    domainList->domainListKeys[*domainList->domainListIndex] = key2test;
                    // add domain list level
                    domainList->domainListLevels[*domainList->domainListIndex] = level;
                    *domainList->domainListIndex += 1;
                    if (subDomainKeyTree->isDomainListNode(key2test, maxLevel, level, curveType)) {
                        level++;
                    }
                    else {
                        key2test = key2test + (1UL << DIM * (maxLevel - level));
                        while (((key2test >> (DIM * (maxLevel - level))) & (keyType)(POW_DIM - 1)) == 0UL) {
                            level--;
                        }
                    }
                } else {
                    level--;
                }

            }
        }*/

        // Parallel version: testing for now ...
        /*
        __global__ void createDomainList(SubDomainKeyTree *subDomainKeyTree, DomainList *domainList,
                                         integer maxLevel, Curve::Type curveType) {

            integer index = threadIdx.x + blockIdx.x * blockDim.x;

            //printf("index: %i, threadIdx.x: %i, blockIdx.x: %i, blockDim.x: %i\n", index, threadIdx.x, blockIdx.x, blockDim.x);

            keyType key2test = 0UL;
            keyType keyMax;
            int domainListIndex;

            key2test = (keyType)blockIdx.x << (DIM * (maxLevel - 1));

            if (threadIdx.x == 0) {

                domainListIndex = atomicAdd(domainList->domainListIndex, 1);
                domainList->domainListKeys[domainListIndex] = key2test;
                domainList->domainListLevels[domainListIndex] = 1;
            }

            __syncthreads();

            integer level = 2;

            key2test += threadIdx.x << (DIM * maxLevel - 2);
            if (threadIdx.x == (POW_DIM - 1) && blockIdx.x == (POW_DIM - 1)) {
#if DIM == 1
                keyType shiftValue = 1;
                keyType toShift = 21;
                keyMax = (shiftValue << toShift) - 1; // 1 << 63 not working!
#elif DIM == 2
                keyType shiftValue = 1;
                keyType toShift = 42;
                keyMax = (shiftValue << toShift) - 1; // 1 << 63 not working!
#else
                keyType shiftValue = 1;
                keyType toShift = 63;
                keyMax = (shiftValue << toShift) - 1; // 1 << 63 not working!
                //keyType keyMax = KEY_MAX;
#endif
            }
            else {
                keyMax = key2test + 1 << (DIM * maxLevel - 2);
            }

            // in principle: traversing a (non-existent) octree by walking the 1D spacefilling curve (keys of the tree nodes)
            while (key2test < keyMax) { // TODO: createDomainList(): key2test < or <= keyMax
                if (subDomainKeyTree->isDomainListNode(key2test & (~0UL << (DIM * (maxLevel - level + 1))),
                                                       maxLevel, level-1, curveType)) {

                    //if (level > 1) {
                    domainListIndex = atomicAdd(domainList->domainListIndex, 1);
                    printf("adding key2test: %lu | level = %i\n", key2test, level);
                    domainList->domainListKeys[domainListIndex] = key2test;
                    // add domain list level
                    domainList->domainListLevels[domainListIndex] = level;
                    //}
                    //*domainList->domainListIndex += 1;
                    if (subDomainKeyTree->isDomainListNode(key2test, maxLevel, level, curveType)) {
                        level++;
                    }
                    else {
                        key2test = key2test + (1UL << DIM * (maxLevel - level));
                        while (((key2test >> (DIM * (maxLevel - level))) & (keyType)(POW_DIM - 1)) == 0UL) {
                            level--;
                        }
                    }
                } else {
                    level--;
                }

            }
        }*/

        __global__ void createDomainList(SubDomainKeyTree *subDomainKeyTree, DomainList *domainList,
                                         integer maxLevel, Curve::Type curveType) {

            integer index = threadIdx.x + blockIdx.x * blockDim.x;

            //printf("index: %i, threadIdx.x: %i, blockIdx.x: %i, blockDim.x: %i\n", index, threadIdx.x, blockIdx.x, blockDim.x);

            keyType key2test = 0UL;
            keyType keyMax;
            int domainListIndex;

            key2test = (keyType)blockIdx.x << (DIM * (maxLevel - 1));

            if (threadIdx.x == 0) {

                domainListIndex = atomicAdd(domainList->domainListIndex, 1);
                domainList->domainListKeys[domainListIndex] = key2test;
                domainList->domainListLevels[domainListIndex] = 1;
            }

            __syncthreads();

            integer level = 2;

            key2test += (keyType)threadIdx.x << (DIM * (maxLevel - 2));
            if (threadIdx.x == (POW_DIM - 1) && blockIdx.x == (POW_DIM - 1)) {
#if DIM == 1
                keyType shiftValue = 1;
                keyType toShift = 21;
                keyMax = (shiftValue << toShift) - 1; // 1 << 63 not working!
#elif DIM == 2
                keyType shiftValue = 1;
                keyType toShift = 42;
                keyMax = (shiftValue << toShift) - 1; // 1 << 63 not working!
#else
                keyType shiftValue = 1;
                keyType toShift = 63;
                keyMax = (shiftValue << toShift) - 1; // 1 << 63 not working!
                //keyType keyMax = KEY_MAX;
#endif
            }
            else {
                keyMax = key2test + (1UL << (DIM * (maxLevel - 2)));
            }

            //printf("threadIdx.x = %i, blockIdx.x = %i, key2test = %lu, keyMax = %lu\n", threadIdx.x, blockIdx.x, key2test, keyMax);

            // in principle: traversing a (non-existent) octree by walking the 1D spacefilling curve (keys of the tree nodes)
            while (key2test < keyMax && level > 1) { // TODO: createDomainList(): key2test < or <= keyMax
                if (subDomainKeyTree->isDomainListNode(key2test & (~0UL << (DIM * (maxLevel - level + 1))),
                                                       maxLevel, level-1, curveType)) {

                    //if (level > 1) {
                    domainListIndex = atomicAdd(domainList->domainListIndex, 1);
                    //printf("adding key2test: %lu | level = %i\n", key2test, level);
                    domainList->domainListKeys[domainListIndex] = key2test;
                    // add domain list level
                    domainList->domainListLevels[domainListIndex] = level;
                    //}
                    //*domainList->domainListIndex += 1;
                    if (subDomainKeyTree->isDomainListNode(key2test, maxLevel, level, curveType)) {
                        level++;
                    }
                    else {
                        key2test = key2test + (1UL << DIM * (maxLevel - level));
                        while (((key2test >> (DIM * (maxLevel - level))) & (keyType)(POW_DIM - 1)) == 0UL) {
                            level--;
                        }
                    }
                } else {
                    level--;
                }

            }
        }


        __global__ void lowestDomainList(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                         DomainList *domainList, DomainList *lowestDomainList,
                                         integer n, integer m) {

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
                            if (particles->nodeType[childIndex] >= 1) {
                                lowestDomainListNode = false;
                            }
                            //for (int k=0; k<*domainList->domainListIndex; k++) {
                            //    if (childIndex == domainList->domainListIndices[k]) {
                            //        //printf("domainIndex = %i  childIndex: %i  domainListIndices: %i\n", domainIndex,
                            //        //       childIndex, domainListIndices[k]);
                            //        lowestDomainListNode = false;
                            //        break;
                            //    }
                            //}
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
                    particles->nodeType[domainIndex] = 2;
                    // add/save key of lowest domain list node
                    lowestDomainList->domainListKeys[lowestDomainIndex] = domainList->domainListKeys[index + offset];
                    // add/save level of lowest domain list node
                    lowestDomainList->domainListLevels[lowestDomainIndex] = domainList->domainListLevels[index + offset];

                    lowestDomainList->borders[lowestDomainIndex * 2 * DIM] = domainList->borders[(index + offset) * 2 * DIM];
                    lowestDomainList->borders[lowestDomainIndex * 2 * DIM + 1] = domainList->borders[(index + offset) * 2 * DIM + 1];
#if DIM > 1
                    lowestDomainList->borders[lowestDomainIndex * 2 * DIM + 2] = domainList->borders[(index + offset) * 2 * DIM + 2];
                    lowestDomainList->borders[lowestDomainIndex * 2 * DIM + 3] = domainList->borders[(index + offset) * 2 * DIM + 3];
#if DIM == 3
                    lowestDomainList->borders[lowestDomainIndex * 2 * DIM + 4] = domainList->borders[(index + offset) * 2 * DIM + 4];
                    lowestDomainList->borders[lowestDomainIndex * 2 * DIM + 5] = domainList->borders[(index + offset) * 2 * DIM + 5];
#endif
#endif

                    // debugging
                    //printf("[rank %i] Adding lowest domain list node #%i : %i (key = %lu)\n", subDomainKeyTree->rank, lowestDomainIndex, domainIndex,
                    //      lowestDomainList->domainListKeys[lowestDomainIndex]);
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

        void Launch::setBorders(DomainList *domainList, real *borders, integer *relevantDomainListOriginalIndex) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::DomainListNS::Kernel::setBorders, domainList, borders,
                         relevantDomainListOriginalIndex);
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
            //TODO: is there any possibility to call kernel createDomainList() with more than one thread?
            ExecutionPolicy executionPolicy(POW_DIM, POW_DIM);
            return cuda::launch(true, executionPolicy, ::DomainListNS::Kernel::createDomainList, subDomainKeyTree,
                                domainList, maxLevel, curveType);
        }

        real Launch::lowestDomainList(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                      DomainList *domainList, DomainList *lowestDomainList, integer n, integer m) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::DomainListNS::Kernel::lowestDomainList, subDomainKeyTree,
                                tree, particles, domainList, lowestDomainList, n, m);
        }

    }

}

namespace ParticlesNS {
    __device__ bool applySphericalCriterion(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                            real d, int index) {

#if DIM == 1
        if (cuda::math::sqrt(particles->x[index] * particles->x[index]) < d) {
#elif DIM == 2
        if (cuda::math::sqrt(particles->x[index] * particles->x[index] +
                particles->y[index] * particles->y[index]) < d) {
#else
        if (cuda::math::sqrt(particles->x[index] * particles->x[index] + particles->y[index] * particles->y[index] +
                  particles->z[index] * particles->z[index]) < d) {
#endif
            return false;
        } else {
            return true;
        }
    }

    __device__ bool applyCubicCriterion(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                        real d, int index) {

#if DIM == 1
        if (cuda::math::abs(particles->x[index]) < d) {
#elif DIM == 2
        if (cuda::math::abs(particles->x[index]) < d &&
            cuda::math::abs(particles->y[index]) < d) {
#else
        if (cuda::math::abs(particles->x[index]) < d &&
            cuda::math::abs(particles->y[index]) < d &&
            cuda::math::abs(particles->z[index]) < d) {
#endif
            return false;
        } else {
            return true;
        }
    }

    namespace Kernel {

        __global__ void mark2remove(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                    int *particles2remove, int *counter, int criterion, real d, int numParticles) {

            int bodyIndex = threadIdx.x + blockDim.x * blockIdx.x;
            int stride = blockDim.x * gridDim.x;
            int offset = 0;
            bool remove;

            while (bodyIndex + offset < numParticles) {

                switch (criterion) {
                    case 0: {
                        remove = applySphericalCriterion(subDomainKeyTree, tree, particles, d, bodyIndex + offset);
                    } break;
                    case 1: {
                        remove = applyCubicCriterion(subDomainKeyTree, tree, particles, d, bodyIndex + offset);
                    } break;
                    default: {
                        cudaTerminate("Criterion for removing particles not available! Exiting...\n");
                    }
                }
                if (remove) {
                    particles2remove[bodyIndex + offset] = 1;
                    atomicAdd(counter, 1);
                } else {
                    particles2remove[bodyIndex + offset] = 0;
                }

                offset += stride;
            }

        }

        real Launch::mark2remove(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                 int *particles2remove, int *counter, int criterion, real d,
                                 int numParticles) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::ParticlesNS::Kernel::mark2remove, subDomainKeyTree,
                                tree, particles, particles2remove, counter, criterion, d, numParticles);
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

        template <typename T, unsigned int blockSize>
        __global__ void reduceBlockwise(T *array, T *outputData, int n) {

            extern __shared__ T sdata[];

            unsigned int tid = threadIdx.x;
            unsigned int i = blockIdx.x*(blockSize*2) + tid;
            unsigned int gridSize = blockSize*2*gridDim.x;

            sdata[tid] = 0;

            while (i < n)
            {
                sdata[tid] += array[i] + array[i+blockSize];
                i += gridSize;
            }

            __syncthreads();

            if (blockSize >= 512) {
                if (tid < 256) {
                    array[tid] += array[tid + 256];
                }
                __syncthreads();
            }
            if (blockSize >= 256) {
                if (tid < 128) {
                    array[tid] += array[tid + 128];
                }
                __syncthreads();
            }
            if (blockSize >= 128) {
                if (tid < 64) {
                    array[tid] += array[tid + 64];
                }
                __syncthreads();
            }

            if (tid < 32)
            {
                if (blockSize >= 64) array[tid] += array[tid + 32];
                if (blockSize >= 32) array[tid] += array[tid + 16];
                if (blockSize >= 16) array[tid] += array[tid + 8];
                if (blockSize >= 8) array[tid] += array[tid + 4];
                if (blockSize >= 4) array[tid] += array[tid + 2];
                if (blockSize >= 2) array[tid] += array[tid + 1];
            }

            if (tid == 0) {
                outputData[blockIdx.x] = array[0];
            }
        }

        //template __global__ void reduceBlockwise<real, 256>(real *array, real *outputData, int n);

        template <typename T, unsigned int blockSize>
        __global__ void blockReduction(const T *indata, T *outdata) {
            integer index = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            //extern __shared__ real *buff;

            while ((index + offset) < blockSize) {
                atomicAdd(&outdata[0], indata[index + offset]);
                __threadfence();
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

            template <typename T, unsigned int blockSize>
            real reduceBlockwise(T *array, T *outputData, int n) {
                ExecutionPolicy executionPolicy(256, blockSize, blockSize * sizeof(T));
                return cuda::launch(true, executionPolicy, ::CudaUtils::Kernel::reduceBlockwise<T, blockSize>, array, outputData, n);
            }
            template real reduceBlockwise<real, 256>(real *array, real *outputData, int n);

            template <typename T, unsigned int blockSize>
            real blockReduction(const T *indata, T *outdata) {
                ExecutionPolicy executionPolicy(256, blockSize);
                return cuda::launch(true, executionPolicy, ::CudaUtils::Kernel::blockReduction<T, blockSize>,
                                    indata, outdata);
            }
            template real blockReduction<real, 256>(const real *indata, real *outdata);

        }
    }
}

namespace Physics {
    namespace Kernel {

        // see: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
        template <unsigned int blockSize>
        __global__ void calculateAngularMomentumBlockwise(Particles *particles, real *outputData, int n) {

            extern __shared__ real sdata[];
            real *lx = sdata;
#if DIM > 1
            real *ly = &sdata[blockSize];
#if DIM > 2
            real *lz = &sdata[2 * blockSize];
#endif
#endif
            unsigned int tid = threadIdx.x;
            unsigned int i = blockIdx.x*(blockSize*2) + tid;
            unsigned int gridSize = blockSize*2*gridDim.x;

            lx[tid] = 0.;
#if DIM > 1
            ly[tid] = 0.;
#if DIM > 2
            lz[tid] = 0.;
#endif
#endif

            while (i < n)
            {
// TODO: implementation for DIM == 2
#if DIM == 3
                lx[tid] += particles->mass[i] * (particles->y[i]*particles->vz[i] - particles->z[i]*particles->vy[i]) +
                            particles->mass[i+blockSize] * (particles->y[i+blockSize]*particles->vz[i+blockSize] - particles->z[i+blockSize]*particles->vy[i+blockSize]);

                ly[tid] += particles->mass[i] * (particles->z[i]*particles->vx[i] - particles->x[i]*particles->vz[i]) +
                            particles->mass[i+blockSize] * (particles->z[i+blockSize]*particles->vx[i+blockSize] - particles->x[i+blockSize]*particles->vz[i+blockSize]);

                lz[tid] += particles->mass[i] * (particles->x[i]*particles->vy[i] - particles->y[i]*particles->vx[i]) +
                            particles->mass[i+blockSize] * (particles->x[i+blockSize]*particles->vy[i+blockSize] - particles->y[i+blockSize]*particles->vx[i+blockSize]);
#endif

                i += gridSize;
            }

            __syncthreads();

            if (blockSize >= 512) {
                if (tid < 256) {
                    lx[tid] += lx[tid + 256];
#if DIM > 1
                    ly[tid] += ly[tid + 256];
#if DIM > 2
                    lz[tid] += lz[tid + 256];
#endif
#endif
                }
                __syncthreads();
            }
            if (blockSize >= 256) {
                if (tid < 128) {
                    lx[tid] += lx[tid + 128];
#if DIM > 1
                    ly[tid] += ly[tid + 128];
#if DIM > 2
                    lz[tid] += lz[tid + 128];
#endif
#endif
                }
                __syncthreads();
            }
            if (blockSize >= 128) {
                if (tid < 64) {
                    lx[tid] += lx[tid + 64];
#if DIM > 1
                    ly[tid] += ly[tid + 64];
#if DIM > 2
                    lz[tid] += lz[tid + 64];
#endif
#endif
                }
                __syncthreads();
            }

            if (tid < 32)
            {
                if (blockSize >= 64) {
                    lx[tid] += lx[tid + 32];
#if DIM > 1
                    ly[tid] += ly[tid + 32];
#if DIM > 2
                    lz[tid] += lz[tid + 32];
#endif
#endif
                }
                if (blockSize >= 32) {
                    lx[tid] += lx[tid + 16];
#if DIM > 1
                    ly[tid] += ly[tid + 16];
#if DIM > 2
                    lz[tid] += lz[tid + 16];
#endif
#endif
                }
                if (blockSize >= 16) {
                    lx[tid] += lx[tid + 8];
#if DIM > 1
                    ly[tid] += ly[tid + 8];
#if DIM > 2
                    lz[tid] += lz[tid + 8];
#endif
#endif
                }
                if (blockSize >= 8) {
                    lx[tid] += lx[tid + 4];
#if DIM > 1
                    ly[tid] += ly[tid + 4];
#if DIM > 2
                    lz[tid] += lz[tid + 4];
#endif
#endif
                }
                if (blockSize >= 4) {
                    lx[tid] += lx[tid + 2];
#if DIM > 1
                    ly[tid] += ly[tid + 2];
#if DIM > 2
                    lz[tid] += lz[tid + 2];
#endif
#endif
                }
                if (blockSize >= 2) {
                    lx[tid] += lx[tid + 1];
#if DIM > 1
                    ly[tid] += ly[tid + 1];
#if DIM > 2
                    lz[tid] += lz[tid + 1];
#endif
#endif
                }
            }

            if (tid == 0) {
                outputData[blockIdx.x] = lx[0];
#if DIM > 1
                outputData[blockSize + blockIdx.x] = ly[0];
#if DIM > 2
                outputData[2 * blockSize + blockIdx.x] = lz[0];
#endif
#endif
            }
        }

        template <unsigned int blockSize>
        __global__ void sumAngularMomentum(const real *indata, real *outdata) {

            integer index = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            //extern __shared__ real *buff;

            while ((index + offset) < blockSize) {
                atomicAdd(&outdata[0], indata[index + offset]);
#if DIM > 1
                atomicAdd(&outdata[1], indata[blockSize + index + offset]);
#if DIM > 2
                atomicAdd(&outdata[2], indata[2 * blockSize + index + offset]);
#endif
#endif
                __threadfence();
                offset += stride;
            }

        }

        __global__ void kineticEnergy(Particles *particles, int n) {
            integer index = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;
            real vel;

            while ((index + offset) < n) {
#if DIM == 1
                vel = abs(particles->vx[index + offset]);
#elif DIM == 2
                vel = cuda::math::sqrt(particles->vx[index + offset] * particles->vx[index + offset] +
                        particles->vy[index + offset] * particles->vy[index + offset]);
#else
                vel = cuda::math::sqrt(particles->vx[index + offset] * particles->vx[index + offset] +
                            particles->vy[index + offset] * particles->vy[index + offset] +
                            particles->vz[index + offset] * particles->vz[index + offset]);
#endif

                particles->u[index + offset] += 0.5 * particles->mass[index + offset] * vel * vel;
                offset += stride;
            }
        }

        namespace Launch {
            template <unsigned int blockSize>
            real calculateAngularMomentumBlockwise(Particles *particles, real *outputData, int n) {
                ExecutionPolicy executionPolicy(256, blockSize, DIM * blockSize * sizeof(real));
                return cuda::launch(true, executionPolicy, ::Physics::Kernel::calculateAngularMomentumBlockwise<blockSize>,
                                    particles, outputData, n);
            }
            template real calculateAngularMomentumBlockwise<256>(Particles *particles, real *outputData, int n);

            template <unsigned int blockSize>
            real sumAngularMomentum(const real *indata, real *outdata) {
                ExecutionPolicy executionPolicy(256, blockSize); //, DIM * sizeof(real));
                return cuda::launch(true, executionPolicy, ::Physics::Kernel::sumAngularMomentum<blockSize>,
                        indata, outdata);
            }
            template real sumAngularMomentum<256>(const real *indata, real *outdata);

            real kineticEnergy(Particles *particles, int n) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::Physics::Kernel::kineticEnergy, particles, n);
            }
        }
    }
}

#endif // TARGET_GPU

namespace TreeNS {

    void compPseudoParticles(Tree *tree, Particles *particles, DomainList *domainList, int numParticles,
                            int nodeIndex) {

        for (int i=0; i<POW_DIM; i++) {
            //if (son[i] != NULL) {
            if (tree->child[POW_DIM * nodeIndex + i] >= numParticles) {
                //if (tree->child[POW_DIM * nodeIndex + i] == 0) {
                //    Logger(TRACE) << " = 0 for nodeIndex: " << nodeIndex << " and i: " << i;
                //}
                //Logger(TRACE) << "new nodeIndex: " << tree->child[POW_DIM * nodeIndex + i];
                compPseudoParticles(tree, particles, domainList, numParticles,
                                    tree->child[POW_DIM * nodeIndex + i]);
            }
        }

        if (nodeIndex >= numParticles) { //(!isLeaf()) {

            particles->mass[nodeIndex] = 0.;
            particles->x[nodeIndex] = 0.;
#if DIM > 1
            particles->y[nodeIndex] = 0.;
#if DIM == 3
            particles->z[nodeIndex] = 0.;
#endif
#endif

            int childIndex = -1;
            for (int j=0; j<POW_DIM; j++) {
                childIndex = tree->child[POW_DIM * nodeIndex + j];
                if (childIndex >= 0) {//if (son[j] != NULL) {
                    particles->mass[nodeIndex] += particles->mass[childIndex]; //son[j]->p.m;
                    particles->x[nodeIndex] += particles->mass[childIndex] * particles->x[childIndex];
#if DIM > 1
                    particles->y[nodeIndex] += particles->mass[childIndex] * particles->y[childIndex];
#if DIM == 3
                    particles->z[nodeIndex] += particles->mass[childIndex] * particles->z[childIndex];
#endif
#endif
                }
            }
            if (particles->mass[nodeIndex] > 0.) {
                particles->x[nodeIndex] /= particles->mass[nodeIndex];
#if DIM > 1
                particles->y[nodeIndex] /= particles->mass[nodeIndex];
#if DIM == 3
                particles->z[nodeIndex] /= particles->mass[nodeIndex];
#endif
#endif
            }
        }

    }

    void lowestDomainListNodes(Tree *tree, Particles *particles, DomainList *domainList, DomainList *lowestDomainList,
                               int numParticles) {
        int domainIndex;
        int childIndex;
        bool lowestDomainListNode;

        *lowestDomainList->domainListIndex = 0;

        Logger(TRACE) << "lowestDomainListNodes: domainListIndex: " << *domainList->domainListIndex;

        for (int i_domainList=0; i_domainList < *domainList->domainListIndex; ++i_domainList) {
            lowestDomainListNode = true;
            domainIndex = domainList->domainListIndices[i_domainList];
            for (int i=0; i<POW_DIM; ++i) {
                childIndex = tree->child[POW_DIM * domainIndex + i];
                if (childIndex != -1) {
                    //Logger(TRACE) << "i: " << i_domainList << "  particles->nodeType[" << childIndex << "] = " << particles->nodeType[childIndex];
                    if (particles->nodeType[childIndex] == 1) {
                        //Logger(TRACE) << "This is not a lowest domain list! (child[8 * " << domainIndex << " + " << i
                        //              << " = childIndex: " << childIndex
                        //              << ") " << domainList->domainListKeys[i_domainList] << " type: "
                        //              << particles->nodeType[childIndex];
                        lowestDomainListNode = false;
                        break;
                    }
                }
                //if (childIndex != -1) {
                    //if (childIndex >= numParticles) {
                    //    for (int k=0; k<*domainList->domainListIndex; ++k) {
                    //        if (childIndex == domainList->domainListIndices[k]) {
                    //            Logger(TRACE) << "This is not a lowest domain list! (childIndex: " << childIndex
                    //                << ") " << domainList->domainListKeys[i_domainList];
                    //            lowestDomainListNode = false;
                    //            break;
                    //        }
                    //    }
                    //}
                //}
                //if (!lowestDomainListNode) {
                //    break;
                //}
            }

            if (lowestDomainListNode) {

                Logger(TRACE) << "Adding lowest domain list node! ...";

                lowestDomainList->domainListIndices[*lowestDomainList->domainListIndex] = domainIndex;
                particles->nodeType[domainIndex] = 2;
                lowestDomainList->domainListKeys[*lowestDomainList->domainListIndex] = domainList->domainListKeys[i_domainList];
                lowestDomainList->domainListLevels[*lowestDomainList->domainListIndex] = domainList->domainListLevels[i_domainList];


                lowestDomainList->borders[(*lowestDomainList->domainListIndex) * 2 * DIM] = domainList->borders[i_domainList * 2 * DIM];
                lowestDomainList->borders[(*lowestDomainList->domainListIndex) * 2 * DIM + 1] = domainList->borders[i_domainList * 2 * DIM + 1];
#if DIM > 1
                lowestDomainList->borders[(*lowestDomainList->domainListIndex) * 2 * DIM + 2] = domainList->borders[i_domainList * 2 * DIM + 2];
                lowestDomainList->borders[(*lowestDomainList->domainListIndex) * 2 * DIM + 3] = domainList->borders[i_domainList * 2 * DIM + 3];
#if DIM == 3
                lowestDomainList->borders[(*lowestDomainList->domainListIndex) * 2 * DIM + 4] = domainList->borders[i_domainList * 2 * DIM + 4];
                lowestDomainList->borders[(*lowestDomainList->domainListIndex)* 2 * DIM + 5] = domainList->borders[i_domainList * 2 * DIM + 5];
#endif
#endif
                (*lowestDomainList->domainListIndex)++;
            }
            else {
                Logger(TRACE) << "NOT Adding lowest domain list node! ...";
            }
        }
    }

    void zeroDomainListNodes(Tree *tree, Particles *particles, DomainList *domainList) {

        int domainIndex;
        for (int i=0; i<*domainList->domainListIndex; ++i) {
            domainIndex = domainList->domainListIndices[i];
            if (particles->nodeType[domainIndex] == 2) {
                //Logger(TRACE) << "*= particles->mass: " << domainIndex;
                particles->x[domainIndex] *= particles->mass[domainIndex];
#if DIM > 1
                particles->y[domainIndex] *= particles->mass[domainIndex];
#if DIM == 3
                particles->z[domainIndex] *= particles->mass[domainIndex];
#endif
#endif
            }

            else {
                //Logger(TRACE) << "[" << domainIndex << "] = 0: " << i << " before x = " << particles->x[domainIndex];
                particles->x[domainIndex] = (real)0;
#if DIM > 1
                particles->y[domainIndex] = (real)0;
#if DIM == 3
                particles->z[domainIndex] = (real)0;
#endif
#endif
                particles->mass[domainIndex] = (real)0;
            }

        }

    }

    void compDomainListPseudoParticlesPerLevel(Tree *tree, Particles *particles, DomainList *domainList,
                                               DomainList *lowestDomainList, int level) {

        int childIndex;
        for (int i=0; i<*domainList->domainListIndex; ++i) {
            if (particles->nodeType[domainList->domainListIndices[i]] == 1) {
                if (domainList->domainListLevels[i] == level) {
                    for (int i_child=0; i_child < POW_DIM; ++i_child) {
                        childIndex = tree->child[POW_DIM * domainList->domainListIndices[i] + i_child];
                        if (childIndex != -1) {
                            particles->mass[domainList->domainListIndices[i]] += particles->mass[childIndex];
                            particles->x[domainList->domainListIndices[i]] += particles->x[childIndex] * particles->mass[childIndex];
#if DIM > 1
                            particles->y[domainList->domainListIndices[i]] += particles->y[childIndex] * particles->mass[childIndex];
#if DIM == 3
                            particles->z[domainList->domainListIndices[i]] += particles->z[childIndex] * particles->mass[childIndex];
#endif
#endif
                        }
                    }
                    if (particles->mass[childIndex] > 0) {
                        particles->x[domainList->domainListIndices[i]] /= particles->mass[childIndex];
#if DIM > 1
                        particles->y[domainList->domainListIndices[i]] /= particles->mass[childIndex];
#if DIM == 3
                        particles->z[domainList->domainListIndices[i]] /= particles->mass[childIndex];
#endif
#endif
                    }
                }
            }
        }

    }

    void newLoadDistribution(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles, int nodeIndex,
                             int numParticles, int numParticlesLocal, Curve::Type curveType) {

        //int numParticles = root.getParticleCount();

        boost::mpi::communicator comm;

        int *particleCounts = new int[subDomainKeyTree->numProcesses];
        boost::mpi::all_gather(comm, &numParticlesLocal, 1, particleCounts);

        //for (int i=0; i < subDomainKeyTree->numProcesses; ++i) {
        //    Logger(TRACE) << "local numP: " << particleCounts[i];
        //}

        int *oldDist = new int[subDomainKeyTree->numProcesses + 1];
        int *newDist = new int [subDomainKeyTree->numProcesses + 1];
        keyType *range = new keyType [subDomainKeyTree->numProcesses + 1];

        oldDist[0] = 0;
        for (int i=0; i < subDomainKeyTree->numProcesses; i++) {
            oldDist[i + 1] = oldDist[i] + particleCounts[i];
        }

        for (int i=0; i <= subDomainKeyTree->numProcesses; i++) {
            newDist[i] = (i * oldDist[subDomainKeyTree->numProcesses]) / subDomainKeyTree->numProcesses;
        }

        for (int i=0; i <= subDomainKeyTree->numProcesses; i++) {
            subDomainKeyTree->range[i] = 0UL;
        }

        int p = 0;
        int n = oldDist[subDomainKeyTree->rank];

        while (n > newDist[p]) {
            p++;
        }

        //updateRange(n, p, newDist);
        updateRange(subDomainKeyTree, tree, particles, nodeIndex, n, p, range, newDist, numParticles, curveType);

        subDomainKeyTree->range[0] = 0UL;
        subDomainKeyTree->range[subDomainKeyTree->numProcesses] = (keyType)KEY_MAX;

        //keyType sendRange[subDomainKeyTree->numProcesses + 1];
        //std::copy(range, range + numProcesses+1, sendRange);

        // //boost::mpi::all_reduce(comm, sendRange, numProcesses+1, range, boost::mpi::maximum<keyType>());
        //boost::mpi::all_reduce(comm, sendRange, numProcesses+1, range, boost::mpi::KeyMaximum<keyType>());

        all_reduce(comm, boost::mpi::inplace_t<keyType*>(&subDomainKeyTree->range[1]), subDomainKeyTree->numProcesses-1, boost::mpi::maximum<keyType>()); //boost::mpi::maximum<keyType>());

        /*for (int i=0; i <= numProcesses; i++){
            Logger(DEBUG) << "Load balancing: NEW range[" << i << "] = " << range[i];
        }*/

        delete [] range;
        delete [] particleCounts;
        delete [] oldDist;
        delete [] newDist;

    }

    void updateRange(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles, int nodeIndex, int &n, int &p, keyType *range,
                     int *newDist, int numParticles, Curve::Type curveType) {
        switch (curveType) {
            case 0: {
                updateRange(subDomainKeyTree, tree, particles, nodeIndex, n, p, range, newDist, 0UL, 1, numParticles); // level : 0
                break;
            }
            //case 1: {
            //    updateRangeHilbert(root, n, p, newDist);
            //    break;
            //}
            default: {
                Logger(ERROR) << "updateRange() not implemented for curve type: " << curveType;
            }
        }
    }

    // TODO: updateRangeHilbert ...
    void updateRange(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles, int nodeIndex, int &n,
                     int &p, keyType *range, int *newDist, keyType k, int level, int numParticles) {

        for (int i=0; i<POW_DIM; i++) {
            //Logger(TRACE) << "nodeIndex: " << nodeIndex << " | childIndex: " << tree->child[POW_DIM * nodeIndex + i];
            if (/*tree->child[POW_DIM * nodeIndex + i] != -1 &&*/tree->child[POW_DIM * nodeIndex + i] > numParticles) { // != -1) {
                //Logger(TRACE) << "nodeIndex: " << nodeIndex << " | childIndex: " << tree->child[POW_DIM * nodeIndex + i] << " | level: " << level << " | i: " << i;
                updateRange(subDomainKeyTree, tree, particles, tree->child[POW_DIM * nodeIndex + i], n, p,
                            range, newDist, k | (keyType)((i * 1UL)  << (DIM*(MAX_LEVEL-level - 1))), level + 1, numParticles);
            // (MAX_LEVEL-level-1)
            // tree->child[POW_DIM * nodeIndex + i]
            // | (((keyType)i)  << (DIM*(MAX_LEVEL-level-1)))
            }
            else {
                if (tree->child[POW_DIM * nodeIndex + i] != -1) {
                    while (n >= newDist[p]) {
                        subDomainKeyTree->range[p] = (k | (keyType)((i * 1UL)  << (DIM*(MAX_LEVEL-level - 1)))); // or k | (keyType)((i * 1UL)  << (DIM*(MAX_LEVEL-level-1))
                        Logger(TRACE) << "p: " << p << " | k = " << (k | (keyType)((i * 1UL)  << (DIM*(MAX_LEVEL-level - 1)))); //k;
                        p++;
                    }
                }
            }
        }
        //if (isLeaf() && !isDomainList()) {
        //if (nodeIndex < numParticles) { // && particles->nodeType[nodeIndex] < 1) {
            //Logger(TRACE) << "nodeIndex: " << nodeIndex;
            //while (n >= newDist[p]) {
                //subDomainKeyTree->range[p] = k;
                //Logger(INFO) << "k = " << k;
            //    p++;
            //}
            //n++;
        //}
    }

    void markParticlesProcess(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                              int *particleProcess, int numParticles) {
        int childIndex;
        for (int i=0; i<POW_DIM; ++i) {
            childIndex = tree->child[i];
            if (childIndex != -1) {
                Logger(TRACE) << "initial childIndex: " << childIndex;
                markParticlesProcess(subDomainKeyTree, tree, particles, childIndex,
                                     (i * 1UL) << (DIM * (MAX_LEVEL)), 2, particleProcess, numParticles);
            }
        }
    }

    void markParticlesProcess(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles, int nodeIndex,
                              keyType key, int level, int *particleProcess, int numParticles) {

        int childIndex, proc;
        for (int i=0; i<POW_DIM; ++i) {
            childIndex = tree->child[POW_DIM * nodeIndex + i];
            if (childIndex != -1) {
                if (childIndex >= numParticles) {
                    markParticlesProcess(subDomainKeyTree, tree, particles, childIndex,
                                         key | (keyType) ((i * 1UL) << (DIM * (MAX_LEVEL - level))),
                                         level + 1, particleProcess, numParticles);
                }
                else {
                    // TODO: Hilbert key
                    proc = subDomainKeyTree->key2proc(key);
                    subDomainKeyTree->procParticleCounter[proc] += 1;
                    particleProcess[childIndex] = proc;
                }
            }
            //if (t.son[i] != NULL) {
                //compTheta(*t.son[i], pMap, diam, k | KeyType{ i << (DIM * (k.maxLevel - level - 1)) },
                //          level + 1);
            //}
        }
    }

}
