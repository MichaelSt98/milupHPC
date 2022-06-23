#include "../../include/subdomain_key_tree/tree.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

CUDA_CALLABLE_MEMBER keyType KeyNS::lebesgue2hilbert(keyType lebesgue, integer maxLevel) {

    //keyType hilbert = 0UL;
    //integer dir = 0;
    //for (integer lvl=maxLevel; lvl>0; lvl--) {
    //    keyType cell = (lebesgue >> ((lvl-1)*DIM)) & (keyType)((1<<DIM)-1);
    //    hilbert = hilbert << DIM;
    //    if (lvl > 0) {
    //        hilbert += HilbertTable[dir][cell];
    //    }
    //    dir = DirTable[dir][cell];
    //}
    //return hilbert;

    keyType hilbert = 1UL;
    int level = 0, dir = 0;
    for (keyType tmp=lebesgue; tmp>1; level++) {
        tmp>>=DIM;
    }
    if (level == 0) {
        hilbert = 0UL;
    }
    for (; level>0; level--) {
        int cell = (lebesgue >> ((level-1)*DIM)) & ((1<<DIM)-1);
        hilbert = (hilbert<<DIM) + HilbertTable[dir][cell];
        dir = DirTable[dir][cell];
    }

    //if (lebesgue == 0UL) {
    //    printf("HERE: lebesgue = %lu --> level = %i, hilbert = %lu\n", lebesgue, rememberLevel, hilbert);
    //}

    return hilbert;

}

CUDA_CALLABLE_MEMBER keyType KeyNS::lebesgue2hilbert(keyType lebesgue, int maxLevel, int level) {

    keyType hilbert = 0UL; // 0UL is our root, placeholder bit omitted
    int dir = 0;
    for (int lvl=maxLevel; lvl>0; lvl--) {
        int cell = (lebesgue >> ((lvl-1)*DIM)) & (keyType)((1<<DIM)-1);
        hilbert = hilbert<<DIM;
        if (lvl>maxLevel-level) {
            hilbert += HilbertTable[dir][cell];
        }
        dir = DirTable[dir][cell];
    }
    return hilbert;
}

CUDA_CALLABLE_MEMBER Tree::Tree() {

}

CUDA_CALLABLE_MEMBER Tree::Tree(integer *count, integer *start, integer *child, integer *sorted, integer *index,
                                integer *toDeleteLeaf, integer *toDeleteNode, real *minX, real *maxX) : count(count),
                                start(start), child(child), sorted(sorted), index(index), toDeleteLeaf(toDeleteLeaf),
                                toDeleteNode(toDeleteNode), minX(minX), maxX(maxX) {

}
CUDA_CALLABLE_MEMBER void Tree::set(integer *count, integer *start, integer *child, integer *sorted,
                                        integer *index, integer *toDeleteLeaf, integer *toDeleteNode,
                                        real *minX, real *maxX) {
    this->count = count;
    this->start = start;
    this->child = child;
    this->sorted = sorted;
    this->index = index;
    this->toDeleteNode = toDeleteNode;
    this->toDeleteLeaf = toDeleteLeaf;
    this->minX = minX;
    this->maxX = maxX;
}

#if DIM > 1
CUDA_CALLABLE_MEMBER Tree::Tree(integer *count, integer *start, integer *child, integer *sorted, integer *index,
                                integer *toDeleteLeaf, integer *toDeleteNode, real *minX, real *maxX, real *minY,
                                real *maxY) : count(count), start(start), child(child), sorted(sorted), index(index),
                                toDeleteLeaf(toDeleteLeaf), toDeleteNode(toDeleteNode), minX(minX), maxX(maxX),
                                minY(minY), maxY(maxY) {

}
CUDA_CALLABLE_MEMBER void Tree::set(integer *count, integer *start, integer *child, integer *sorted,
                                        integer *index, integer *toDeleteLeaf, integer *toDeleteNode, real *minX,
                                        real *maxX, real *minY, real *maxY) {
    this->count = count;
    this->start = start;
    this->child = child;
    this->sorted = sorted;
    this->index = index;
    this->toDeleteNode = toDeleteNode;
    this->toDeleteLeaf = toDeleteLeaf;
    this->minX = minX;
    this->maxX = maxX;
    this->minY = minY;
    this->maxY = maxY;
}

#if DIM == 3
CUDA_CALLABLE_MEMBER Tree::Tree(integer *count, integer *start, integer *child, integer *sorted, integer *index,
                                integer *toDeleteLeaf, integer *toDeleteNode,
                                real *minX, real *maxX, real *minY, real *maxY, real *minZ, real *maxZ) : count(count),
                                start(start), child(child), sorted(sorted), index(index), toDeleteLeaf(toDeleteLeaf),
                                toDeleteNode(toDeleteNode), minX(minX), maxX(maxX), minY(minY), maxY(maxY), minZ(minZ),
                                maxZ(maxZ) {

}
CUDA_CALLABLE_MEMBER void Tree::set(integer *count, integer *start, integer *child, integer *sorted,
                                        integer *index, integer *toDeleteLeaf, integer *toDeleteNode,
                                        real *minX, real *maxX, real *minY, real *maxY,
                                        real *minZ, real *maxZ) {
    this->count = count;
    this->start = start;
    this->child = child;
    this->sorted = sorted;
    this->index = index;
    this->toDeleteNode = toDeleteNode;
    this->toDeleteLeaf = toDeleteLeaf;
    this->minX = minX;
    this->maxX = maxX;
    this->minY = minY;
    this->maxY = maxY;
    this->minZ = minZ;
    this->maxZ = maxZ;
}
#endif
#endif

CUDA_CALLABLE_MEMBER void Tree::reset(integer index, integer n) {
#if DIM == 1
    #pragma unroll 2
#elif DIM == 2
    #pragma unroll 4
#else
    #pragma unroll 8
#endif
    for (integer i=0; i<POW_DIM; i++) {
        // reset child indices
        child[index * POW_DIM + i] = -1;
    }
    // reset counter in dependence of being a node or a leaf
    if (index < n) {
        count[index] = 1;
    }
    else {
        count[index] = 0;
    }
    // reset start
    start[index] = -1;
    sorted[index] = 0;
}

CUDA_CALLABLE_MEMBER keyType Tree::getParticleKey(Particles *particles, integer index, integer maxLevel,
                                                  Curve::Type curveType) {

    integer level = 0;
    keyType particleKey = (keyType)0;

    integer sonBox;
    real min_x = *minX;
    real max_x = *maxX;
#if DIM > 1
    real min_y = *minY;
    real max_y = *maxY;
#if DIM == 3
    real min_z = *minZ;
    real max_z = *maxZ;
#endif
#endif

    integer particleLevel;
    integer particleLevelTemp = 0;
    integer childIndex = 0;
    // calculate path to the particle's position assuming an (oct-)tree with above bounding boxes
    while (level <= maxLevel) {
        sonBox = 0;
        // find insertion point for body
        if (particles->x[index] < 0.5 * (min_x + max_x)) {
            sonBox += 1;
            max_x = 0.5 * (min_x + max_x);
        }
        else { min_x = 0.5 * (min_x + max_x); }
#if DIM > 1
        if (particles->y[index] < 0.5 * (min_y+max_y)) {
            sonBox += 2;
            max_y = 0.5 * (min_y + max_y);
        }
        else { min_y = 0.5 * (min_y + max_y); }
#if DIM == 3
        if (particles->z[index] < 0.5 * (min_z+max_z)) {
            sonBox += 4;
            max_z = 0.5 * (min_z + max_z);
        }
        else { min_z =  0.5 * (min_z + max_z); }
#endif
#endif
        particleKey = particleKey | ((keyType)sonBox << (keyType)(DIM * (maxLevel-level-1)));
        level++;

        particleLevelTemp++;
        if (childIndex == index) {
            particleLevel = particleLevelTemp;
        }
        //for (int i_child = 0; i_child < POW_DIM; i_child++) {
        //    if (child[POW_DIM * childIndex + i_child] == index) {
        //        printf("found index = %i for child[8 * %i + %i] = %i\n", index, childIndex, i_child, child[POW_DIM * childIndex + i_child]);
        //        break;
        //    }
        //}
        childIndex = child[POW_DIM * childIndex + sonBox];
    }

    //if (particleLevel == 0) {
    //    printf("particleLevel = %i particleLevelTemp = %i index = %i (%f, %f, %f)\n", particleLevel, particleLevelTemp, index,
    //           particles->x[index], particles->y[index], particles->z[index]);
    //}

    //if (particleKey == 0UL) {
    //    printf("Why key = %lu? x = (%f, %f, %f) min = (%f, %f, %f), max = (%f, %f, %f)\n", particleKey,
    //           particles->x[index], particles->y[index], particles->z[index],
    //           *minX, *minY, *minZ, *maxX, *maxY, *maxZ);
    //}

    switch (curveType) {
        case Curve::lebesgue: {
            return particleKey;
        }
        case Curve::hilbert: {
            return KeyNS::lebesgue2hilbert(particleKey, maxLevel, maxLevel);
            //return KeyNS::lebesgue2hilbert(particleKey, maxLevel);
        }
        default:
            printf("Curve type not available!\n");
            return (keyType)0;
    }
}

CUDA_CALLABLE_MEMBER integer Tree::getTreeLevel(Particles *particles, integer index, integer maxLevel,
                                                Curve::Type curveType) {

    keyType key = getParticleKey(particles, index, maxLevel); //, curveType); //TODO: hilbert working for lebesgue: why???
    integer level = 0;
    integer childIndex;

    integer path[MAX_LEVEL];
    for (integer i=0; i<maxLevel; i++) {
        path[i] = (integer) (key >> (maxLevel * DIM - DIM * (i + 1)) & (integer)(POW_DIM - 1));
        //printf("path[%i] = %i\n", i, path[i]);
    }

    childIndex = 0;

    for (integer i=0; i<maxLevel; i++) {
        level++;
        if (childIndex == index) {
            return level;
        }
        childIndex = child[POW_DIM * childIndex + path[i]];
    }

#if DIM == 3
#if SAFETY_LEVEL > 1
    printf("ATTENTION: level = -1 (index = %i x = (%f, %f, %f) %f) tree index = %i\n",
           index, particles->x[index], particles->y[index], particles->z[index], particles->mass[index], *this->index);
#endif
#endif

    //for (integer i=0; i<maxLevel; i++) {
    //    childIndex = child[POW_DIM * childIndex + path[i]];
    //    for (int k=0; k<8; k++) {
    //        if (child[POW_DIM * childIndex + k] == index) {
    //            printf("FOUND index = %i in level %i for child = %i x = (%f, %f, %f) ((%i, %i), (%i, %i))\n", index, i, k,
    //                   particles->x[index], particles->y[index], particles->z[index],
    //                   toDeleteLeaf[0], toDeleteLeaf[1], toDeleteNode[0], toDeleteNode[1]);
    //        }
    //    }
    //    //printf("index = %i, path[%i] = %i, childIndex = %i\n", index, i, path[i], childIndex);
    //}

    return -1;
}

//TODO: is this still working? since count only used within buildTree (probably yes)
CUDA_CALLABLE_MEMBER integer Tree::sumParticles() {
    integer sumParticles = 0;
    // sum over first level tree count values
    for (integer i=0; i<POW_DIM; i++) {
        sumParticles += count[child[i]];
    }
    printf("sumParticles = %i\n", sumParticles);
    return sumParticles;
}

CUDA_CALLABLE_MEMBER Tree::~Tree() {

}

__global__ void TreeNS::Kernel::computeBoundingBox(Tree *tree, Particles *particles, integer *mutex, integer n,
                                                 integer blockSize) {

    integer index = threadIdx.x + blockDim.x * blockIdx.x;
    integer stride = blockDim.x * gridDim.x;

    real x_min = particles->x[index];
    real x_max = particles->x[index];
#if DIM > 1
    real y_min = particles->y[index];
    real y_max = particles->y[index];
#if DIM == 3
    real z_min = particles->z[index];
    real z_max = particles->z[index];
#endif
#endif

    extern __shared__ real buffer[];

    real* x_min_buffer = (real*)buffer;
    real* x_max_buffer = (real*)&x_min_buffer[blockSize];
#if DIM > 1
    real* y_min_buffer = (real*)&x_max_buffer[blockSize];
    real* y_max_buffer = (real*)&y_min_buffer[blockSize];
#if DIM == 3
    real* z_min_buffer = (real*)&y_max_buffer[blockSize];
    real* z_max_buffer = (real*)&z_min_buffer[blockSize];
#endif
#endif

    integer offset = stride;

    // find (local) min/max
    while (index + offset < n) {

        x_min = cuda::math::min(x_min, particles->x[index + offset]);
        x_max = cuda::math::max(x_max, particles->x[index + offset]);
#if DIM > 1
        y_min = cuda::math::min(y_min, particles->y[index + offset]);
        y_max = cuda::math::max(y_max, particles->y[index + offset]);
#if DIM == 3
        z_min = cuda::math::min(z_min, particles->z[index + offset]);
        z_max = cuda::math::max(z_max, particles->z[index + offset]);
#endif
#endif

        offset += stride;
    }

    // save value in corresponding buffer
    x_min_buffer[threadIdx.x] = x_min;
    x_max_buffer[threadIdx.x] = x_max;
#if DIM > 1
    y_min_buffer[threadIdx.x] = y_min;
    y_max_buffer[threadIdx.x] = y_max;
#if DIM == 3
    z_min_buffer[threadIdx.x] = z_min;
    z_max_buffer[threadIdx.x] = z_max;
#endif
#endif

    // synchronize threads / wait for unfinished threads
    __syncthreads();

    integer i = blockDim.x/2; // assuming blockDim.x is a power of 2!

    // reduction within block
    while (i != 0) {
        if (threadIdx.x < i) {
            x_min_buffer[threadIdx.x] = cuda::math::min(x_min_buffer[threadIdx.x], x_min_buffer[threadIdx.x + i]);
            x_max_buffer[threadIdx.x] = cuda::math::max(x_max_buffer[threadIdx.x], x_max_buffer[threadIdx.x + i]);
#if DIM > 1
            y_min_buffer[threadIdx.x] = cuda::math::min(y_min_buffer[threadIdx.x], y_min_buffer[threadIdx.x + i]);
            y_max_buffer[threadIdx.x] = cuda::math::max(y_max_buffer[threadIdx.x], y_max_buffer[threadIdx.x + i]);
#if DIM == 3
            z_min_buffer[threadIdx.x] = cuda::math::min(z_min_buffer[threadIdx.x], z_min_buffer[threadIdx.x + i]);
            z_max_buffer[threadIdx.x] = cuda::math::max(z_max_buffer[threadIdx.x], z_max_buffer[threadIdx.x + i]);
#endif
#endif

        }
        __syncthreads();
        i /= 2;
    }

    // combining the results and generate the root cell
    if (threadIdx.x == 0) {
        while (atomicCAS(mutex, 0 ,1) != 0); // lock

        *tree->minX = cuda::math::min(*tree->minX, x_min_buffer[0]);
        *tree->maxX = cuda::math::max(*tree->maxX, x_max_buffer[0]);
#if DIM > 1
        *tree->minY = cuda::math::min(*tree->minY, y_min_buffer[0]);
        *tree->maxY = cuda::math::max(*tree->maxY, y_max_buffer[0]);

#if CUBIC_DOMAINS
        if (*tree->minY < *tree->minX) {
            *tree->minX = *tree->minY;
        }
        else {
            *tree->minY = *tree->minX;
        }
        if (*tree->maxY > *tree->maxX) {
            *tree->maxX = *tree->maxY;
        }
        else {
            *tree->maxY = *tree->maxX;
        }
#endif

#if DIM == 3
        *tree->minZ = cuda::math::min(*tree->minZ, z_min_buffer[0]);
        *tree->maxZ = cuda::math::max(*tree->maxZ, z_max_buffer[0]);

#if CUBIC_DOMAINS
        if (*tree->minZ < *tree->minX) {
            *tree->minX = *tree->minZ;
            *tree->minY = *tree->minZ;
        }
        else {
            *tree->minZ = *tree->minX;
        }
        if (*tree->maxZ > *tree->maxX) {
            *tree->maxX = *tree->maxZ;
            *tree->maxY = *tree->maxZ;
        }
        else {
            *tree->maxZ = *tree->maxX;
        }
#endif

#endif
#endif
        atomicExch(mutex, 0); // unlock
    }
}

// just a wrapper for the member function
__global__ void TreeNS::Kernel::sumParticles(Tree *tree) {

    integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    integer stride = blockDim.x * gridDim.x;
    integer offset = 0;

    if (bodyIndex == 0) {
        integer sumParticles = tree->sumParticles();
        printf("sumParticles = %i\n", sumParticles);
    }
}

#define COMPUTE_DIRECTLY 0

/*
__global__ void TreeNS::Kernel::buildTree(Tree *tree, Particles *particles, integer n, integer m) {

    integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    integer stride = blockDim.x * gridDim.x;

    //note: -1 used as "null pointer"
    //note: -2 used to lock a child (pointer)

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

    offset = 0;

    while ((bodyIndex + offset) < n) {

        if (newBody) {

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
            // x direction
            if (x < 0.5 * (min_x + max_x)) { // x direction
                childPath += 1;
                max_x = 0.5 * (min_x + max_x);
            }
            else {
                min_x = 0.5 * (min_x + max_x);
            }
#if DIM > 1
            // y direction
            if (y < 0.5 * (min_y + max_y)) { // y direction
                childPath += 2;
                max_y = 0.5 * (min_y + max_y);
            }
            else {
                min_y = 0.5 * (min_y + max_y);
            }
#if DIM == 3
            // z direction
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

        // traverse tree until hitting leaf node
        while (childIndex >= m) { //n

            temp = childIndex;
            level++;

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
                            tree->child[POW_DIM * temp + childPath] = cell;
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

                        tree->child[POW_DIM * cell + childPath] = childIndex;
                        particles->level[cell] = level;
                        particles->level[childIndex] += 1;
                        tree->start[cell] = -1;

#if DEBUGGING
                        if (particles->level[cell] >= particles->level[childIndex]) {
                            cudaAssert("lvl: %i vs. %i\n", particles->level[cell], particles->level[childIndex]);
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
                        childIndex = tree->child[POW_DIM * temp + childPath];
                    }

                    tree->child[POW_DIM * temp + childPath] = bodyIndex + offset;
                    particles->level[bodyIndex + offset] = level + 1;

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
*/

__global__ void TreeNS::Kernel::buildTree(Tree *tree, Particles *particles, integer n, integer m) {

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

    offset = 0;

    while ((bodyIndex + offset) < n) {

        if (newBody) {

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
            // x direction
            if (x < 0.5 * (min_x + max_x)) { // x direction
                childPath += 1;
                max_x = 0.5 * (min_x + max_x);
            }
            else {
                min_x = 0.5 * (min_x + max_x);
            }
#if DIM > 1
            // y direction
            if (y < 0.5 * (min_y + max_y)) { // y direction
                childPath += 2;
                max_y = 0.5 * (min_y + max_y);
            }
            else {
                min_y = 0.5 * (min_y + max_y);
            }
#if DIM == 3
            // z direction
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

            if (atomicCAS((int *) &childList[locked], childIndex, -2) == childIndex) {

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
                            cudaAssert("lvl: %i vs. %i\n", particles->level[cell], particles->level[childIndex]);
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
        __syncthreads();
    }
}

/*
__global__ void TreeNS::Kernel::buildTree(Tree *tree, Particles *particles, integer n, integer m)
{
    register int inc = blockDim.x * gridDim.x;
    register int i = threadIdx.x + blockIdx.x * blockDim.x;
    register int k;
    register int childIndex, child;
    register int lockedIndex;
    register double x;
    register double min_x, max_x;
#if DIM > 1
    register double y;
    register double min_y, max_y;
#endif
    register double r;
    register double dx;
#if DIM > 1
    register double dy;
#endif
    register double rootX = 0.5 * (*tree->maxX + *tree->minX);
    register double rootRadius = 0.5 * (*tree->maxX - *tree->minX);
#if DIM > 1
    register double rootY = 0.5 * (*tree->maxY + *tree->minY);
    rootRadius = cuda::math::max(rootRadius, 0.5 * (*tree->maxY - *tree->minY));
#endif
    register int depth = 0;
    register bool isNewParticle = true;
    register int currentNodeIndex;
    register int newNodeIndex;
    register int subtreeNodeIndex;
#if DIM == 3
    register double z;
    register double min_z, max_z;
    register double dz;
    register double rootZ = 0.5 * (*tree->maxZ + *tree->minZ);
    rootRadius = cuda::math::max(rootRadius, 0.5 * (*tree->maxZ - *tree->minZ));
#endif

    volatile double *px, *pm;
#if DIM > 1
    volatile double *py;
#if DIM == 3
    volatile double *pz;
#endif
#endif

    px  = particles->x;
    pm = particles->mass;
#if DIM > 1
    py = particles->y;
#if DIM == 3
    pz = particles->z;
#endif
#endif

    while (i < n) {
        depth = 0;

        if (isNewParticle) {
            isNewParticle = false;
            // cache particle data
            x = px[i];
            min_x = *tree->minX;
            max_x = *tree->maxX;
            //p.ax[i] = 0.0;
#if DIM > 1
            y = py[i];
            min_y = *tree->minY;
            max_y = *tree->maxY;
            //p.ay[i] = 0.0;
#if DIM == 3
            z = pz[i];
            min_z = *tree->minZ;
            max_z = *tree->maxZ;
            //p.az[i] = 0.0;
#endif
#endif

            // start at root
            currentNodeIndex = 0;
            r = rootRadius;
            childIndex = 0;

//            if (x > rootX) childIndex = 1;
//#if DIM > 1
//            if (y > rootY) childIndex += 2;
//#if DIM == 3
//            if (z > rootZ) childIndex += 4;
//#endif
//#endif


            // find insertion point for body
            // x direction
            if (x < 0.5 * (min_x + max_x)) { // x direction
                childIndex += 1;
                max_x = 0.5 * (min_x + max_x);
            }
            else {
                min_x = 0.5 * (min_x + max_x);
            }
#if DIM > 1
            // y direction
            if (y < 0.5 * (min_y + max_y)) { // y direction
                childIndex += 2;
                max_y = 0.5 * (min_y + max_y);
            }
            else {
                min_y = 0.5 * (min_y + max_y);
            }
#if DIM == 3
            // z direction
            if (z < 0.5 * (min_z + max_z)) {  // z direction
                childIndex += 4;
                max_z = 0.5 * (min_z + max_z);
            }
            else {
                min_z = 0.5 * (min_z + max_z);
            }
#endif
#endif

        }

        // follow path to leaf
        child = tree->child[POW_DIM * currentNodeIndex + childIndex]; //childList[childListIndex(currentNodeIndex, childIndex)];
        // leaves are 0 ... numParticles
        while (child >= m) {
            currentNodeIndex = child;
            depth++;
            r *= 0.5;
            // which child?
            childIndex = 0;

//            if (x > px[currentNodeIndex]) childIndex = 1;
//#if DIM > 1
//            if (y > py[currentNodeIndex]) childIndex += 2;
//#if DIM > 2
//            if (z > pz[currentNodeIndex]) childIndex += 4;
//#endif
//#endif

            // find insertion point for body
            if (x < 0.5 * (min_x + max_x)) { // x direction
                childIndex += 1;
                max_x = 0.5 * (min_x + max_x);
            }
            else {
                min_x = 0.5 * (min_x + max_x);
            }
#if DIM > 1
            if (y < 0.5 * (min_y + max_y)) { // y direction
                childIndex += 2;
                max_y = 0.5 * (min_y + max_y);
            }
            else {
                min_y = 0.5 * (min_y + max_y);
            }
#if DIM == 3
            if (z < 0.5 * (min_z + max_z)) { // z direction
                childIndex += 4;
                max_z = 0.5 * (min_z + max_z);
            }
            else {
                min_z = 0.5 * (min_z + max_z);
            }
#endif
#endif

            child = tree->child[POW_DIM * currentNodeIndex + childIndex]; //childList[childListIndex(currentNodeIndex, childIndex)];
        }

        // we want to insert the current particle i into currentNodeIndex's child at position childIndex
        // where child is now empty, locked or a particle
        // if empty -> simply insert, if particle -> create new subtree
        if (child != -2) {
            // the position where we want to place the particle gets locked
            lockedIndex = tree->child[POW_DIM * currentNodeIndex + childIndex]; //childListIndex(currentNodeIndex, childIndex);
            // atomic compare and save: compare if child is still the current value of childlist at the index lockedIndex, if so, lock it
            // atomicCAS returns the old value of child
            if (child == atomicCAS(&tree->child[lockedIndex], child, -2)) { //&childList[lockedIndex]
                // if the destination is empty, insert particle
                if (child == -1) {
                    // insert the particle into this leaf
                    tree->child[lockedIndex] = i; //childList[lockedIndex] = i;
                } else {
                    // there is already a particle, create new inner node
                    subtreeNodeIndex = POW_DIM * m;
                    do {
                        // get the next free nodeIndex
                        newNodeIndex = atomicAdd(tree->index, 1); //atomicSub((int * ) &maxNodeIndex, 1) - 1;

                        // throw error if there aren't enough node indices available
                        //if (newNodeIndex > m) {
                            //printf("(thread %d): error during tree creation: not enough nodes. newNodeIndex %d, maxNodeIndex %d, numParticles: %d\n", threadIdx.x, newNodeIndex, maxNodeIndex, numParticles);
                            //assert(0);
                        //}

                        // the first available free nodeIndex will be the subtree node
                        subtreeNodeIndex = min(subtreeNodeIndex, newNodeIndex);

                        dx = (childIndex & 1) * r;
#if DIM > 1
                        dy = ((childIndex >> 1) & 1) * r;
#if DIM == 3
                        dz = ((childIndex >> 2) & 1) * r;
#endif
#endif
                        depth++;
                        r *= 0.5;

                        // we save the radius here, so we can use it during neighboursearch. we have to set it to EMPTY after the neighboursearch
                        pm[newNodeIndex] = r;

//                        dx = px[newNodeIndex] = px[currentNodeIndex] - r + dx;
//#if DIM > 1
//                        dy = py[newNodeIndex] = py[currentNodeIndex] - r + dy;
//#if DIM == 3
//                        dz = pz[newNodeIndex] = pz[currentNodeIndex] - r + dz;
//#endif
//#endif


                        dx = px[newNodeIndex] = (0.5 * (min_x + max_x)) - r + dx;
#if DIM > 1
                        dy = py[newNodeIndex] = (0.5 * (min_y + max_y)) - r + dy;
#if DIM == 3
                        dz = pz[newNodeIndex] = (0.5 * (min_z + max_z)) - r + dz;
#endif
#endif

                        //for (k = 0; k < POW_DIM; k++) {
                        //    tree->child[POW_DIM * newNodeIndex + k] = -1; //childList[childListIndex(newNodeIndex, k)] = EMPTY;
                        //}

                        if (subtreeNodeIndex != newNodeIndex) {
                            // this condition is true when the two particles are so close to each other, that they are
                            // again put into the same node, so we have to create another new inner node.
                            // in this case, currentNodeIndex is the previous newNodeIndex
                            // and childIndex is the place where the particle i belongs to, relative to the previous newNodeIndex
                            tree->child[POW_DIM * currentNodeIndex + childIndex] = newNodeIndex; //childList[childListIndex(currentNodeIndex, childIndex)] = newNodeIndex;
                        }

                        childIndex = 0;

//                        if (px[child] > dx) childIndex = 1;
//#if DIM > 1
//                        if (py[child] > dy) childIndex += 2;
//#if DIM == 3
//                        if (pz[child] > dz) childIndex += 4;
//#endif
//#endif

                        //if (particles->x[bodyIndex + offset] < 0.5 * (min_x + max_x)) {
                        if (px[child] < 0.5 * (min_x + max_x)) {
                            childIndex += 1;
                            max_x = 0.5 * (min_x + max_x);
                        } else {
                            min_x = 0.5 * (min_x + max_x);
                        }
#if DIM > 1
                        //if (particles->y[bodyIndex + offset] < 0.5 * (min_y + max_y)) {
                        if (py[child] < 0.5 * (min_y + max_y)) {
                            childIndex += 2;
                            max_y = 0.5 * (min_y + max_y);
                        } else {
                            min_y = 0.5 * (min_y + max_y);
                        }
#if DIM == 3
                        //if (particles->z[bodyIndex + offset] < 0.5 * (min_z + max_z)) {
                        if (pz[child] < 0.5 * (min_z + max_z)) {
                            childIndex += 4;
                            max_z = 0.5 * (min_z + max_z);
                        } else {
                            min_z = 0.5 * (min_z + max_z);
                        }
#endif
#endif
                        tree->child[POW_DIM * newNodeIndex + childIndex] = child; //childList[childListIndex(newNodeIndex, childIndex)] = child;

                        // compare positions of particle i to the new node
                        currentNodeIndex = newNodeIndex;
                        childIndex = 0;

//                        if (x > dx) childIndex = 1;
//#if DIM > 1
//                        if (y > dy) childIndex += 2;
//#if DIM == 3
//                        if (z > dz) childIndex += 4;
//#endif
//#endif

                        if (x < 0.5 * (min_x + max_x)) { childIndex += 1; }
#if DIM > 1
                        if (y < 0.5 * (min_y + max_y)) { childIndex += 2; }
#if DIM == 3
                        if (z < 0.5 * (min_z + max_z)) { childIndex += 4; }
#endif
#endif
                        child = tree->child[POW_DIM * currentNodeIndex + childIndex]; //child = childList[childListIndex(currentNodeIndex, childIndex)];
                        // continue creating new nodes (with half radius each) until the other particle is not in the same spot in the tree
                    } while (child >= 0 && child < n);

                    tree->child[POW_DIM * currentNodeIndex + childIndex] = i; //childList[childListIndex(currentNodeIndex, childIndex)] = i;
                    __threadfence();
                    //__threadfence() is used to halt the current thread until all previous writes to shared and global memory are visible
                    // by other threads. It does not halt nor affect the position of other threads though!
                    tree->child[lockedIndex] = subtreeNodeIndex; //childList[lockedIndex] = subtreeNodeIndex;
                }
                //p.depth[i] = depth;
                // continue with next particle
                i += inc;
                isNewParticle = true;
            }
        }
        __syncthreads(); // child was locked, wait for other threads to unlock
    }
}
*/

__global__ void TreeNS::Kernel::prepareSorting(Tree *tree, Particles *particles, integer n, integer m) {
    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    while ((bodyIndex + offset) < n) {
        tree->start[bodyIndex + offset] = bodyIndex + offset;
        offset += stride;
    }
}


__global__ void TreeNS::Kernel::calculateCentersOfMass(Tree *tree, Particles *particles, integer n, integer level) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int offset = n;

    register int i, index;
    register real mass;
    register real x;
#if DIM > 1
    register real y;
#if DIM == 3
    register real z;
#endif
#endif

    while ((bodyIndex + offset) < *tree->index) {

        i = bodyIndex + offset;

        if (particles->level[i] == level) {

            //if (particles->level[bodyIndex + offset] == -1 || particles->level[bodyIndex + offset] > 21) {
            //    printf("level[%i] = %i!!!\n", bodyIndex + offset, particles->level[bodyIndex + offset]);
            //    //assert(0);
            //}

            // reset entries
            mass = 0.; //particles->mass[i] = 0.;
            x = 0.; //particles->x[i] = 0.;
#if DIM > 1
            y = 0.; //particles->y[i] = 0.;
#if DIM == 3
            z = 0.; //particles->z[i] = 0.;
#endif
#endif

            // loop over children and add contribution (*=position(child) * mass(child))
            #pragma unroll
            for (int child = 0; child < POW_DIM; ++child) {
                index = POW_DIM * i + child;
                if (tree->child[index] != -1) {
                    x += particles->weightedEntry(tree->child[index], Entry::x);
#if DIM > 1
                    y += particles->weightedEntry(tree->child[index], Entry::y);
#if DIM == 3
                    z += particles->weightedEntry(tree->child[index], Entry::z);
#endif
#endif
                    mass += particles->mass[tree->child[index]];
                }
            }

            // finish center of mass calculation by dividing with mass
            if (mass > 0.) { //particles->mass[i]
                //particles->x[i] /= particles->mass[i];
                x /= mass;
#if DIM > 1
                //particles->y[i] /= particles->mass[i];
                y /= mass;
#if DIM == 3
                //particles->z[i] /= particles->mass[i];
                z /= mass;
#endif
#endif
            }

            particles->mass[i] = mass;
            particles->x[i] = x;
#if DIM > 1
            particles->y[i] = y;
#if DIM == 3
            particles->z[i] = z;
#endif
#endif

        }
        offset += stride;
    }

}

// TODO: not needed anymore and ATTENTION! requires preparation within buildTree (using atomicAdd)
//  therefore moved to calculateCenterOfMass(), since atomicOperations for 64-bit values VERY expensive
__global__ void TreeNS::Kernel::centerOfMass(Tree *tree, Particles *particles, integer n) {

    integer bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
    integer stride = blockDim.x*gridDim.x;
    integer offset = 0;

    //note: most of it already done within buildTreeKernel
    bodyIndex += n;

    while (bodyIndex + offset < *tree->index) {

        if (particles->mass[bodyIndex + offset] != 0) {
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

// TODO: not working properly right now (only working for small number of particles)
__global__ void TreeNS::Kernel::sort(Tree *tree, integer n, integer m) {

    integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    integer stride = blockDim.x * gridDim.x;
    integer offset = 0;

    integer s = 0;
    if (threadIdx.x == 0) {

        for (integer i=0; i<POW_DIM; i++){
            integer node = tree->child[i];
            if (node >= m) { //n // not a leaf node
                tree->start[node] = s;
                s += tree->count[node];
            }
            else if (node >= 0) { // leaf node
                tree->sorted[s] = node;
                s++;
            }
            //if (bodyIndex == 0) {
            //    printf("i = %i, start[%i] = %i (m = %i)\n", i, node, tree->start[node], m);
            //}
        }
    }

    //__threadfence();
    //__syncthreads();

    integer cell = m + bodyIndex;
    integer ind = *tree->index;
    //integer ind = tree->toDeleteNode[1];

    //int counter = 0;
    while ((cell + offset) < ind /*&& counter <= 500*/) {

        //if ((cell + offset) < (m + POW_DIM)) {
        //    for (int i=0; i<POW_DIM; i++) {
        //        printf("cell + offset = %i, start = %i, child = %i, start = %i, stride = %i\n", cell + offset, tree->start[cell + offset],
        //               tree->child[POW_DIM * (cell + offset) + i], tree->start[cell + offset], stride);
        //    }
        //}

        s = tree->start[cell + offset];
        //counter += 1;

        //if (counter >= 500 /*&& s >= 0*/) {
        //    for (int i=0; i<POW_DIM; i++) {
        //        printf("counter: %i, cell + offset = %i, start = %i, child = %i, count = %i\n", counter, cell + offset, s,
        //               tree->child[POW_DIM * (cell + offset) + i], tree->count[tree->child[POW_DIM * (cell + offset) + i]]);
        //    }
        //}

        if (s >= 0) {

            for (integer i_child=0; i_child<POW_DIM; i_child++) {
                integer node = tree->child[POW_DIM*(cell+offset) + i_child];
                if (node >= m) { //m // not a leaf node
                    tree->start[node] = s;
                    s += tree->count[node];
                    //if (tree->count[node] >= 0) {
                    //    s += tree->count[node];
                    //}
                    //else {
                    //    printf("+= %i\n", tree->count[node]);
                    //}
                }
                else if (node >= 0) { // leaf node
                    tree->sorted[s] = node;
                    s++;
                }
            }
            //if (counter >= 0) {
            //    offset -= counter * stride;
            //    counter = 0;
            //}
            //else {
            //    offset += stride;
            //}
            // //counter = 0;
            offset += stride;
        }
        //else {
        //    printf("ATTENTION: s = %i for %i\n", s, cell + offset);
        //    offset += stride;
        //    counter++;
        //    //break;
        //}

    }
}

__global__ void TreeNS::Kernel::getParticleKeys(Tree *tree, Particles *particles, keyType *keys, integer maxLevel,
                                integer n, Curve::Type curveType) {

    integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    integer stride = blockDim.x * gridDim.x;
    integer offset = 0;

    keyType particleKey;

    while (bodyIndex + offset < n) {

        particleKey = tree->getParticleKey(particles, bodyIndex + offset, maxLevel, curveType);
#if DEBUGGING
#if DIM == 3
        if (particleKey == 1UL) {
            printf("particleKey = %lu (%f, %f, %f)\n", particleKey, particles->x[bodyIndex + offset],
                   particles->y[bodyIndex + offset], particles->z[bodyIndex + offset]);
        }
#endif
#endif
        keys[bodyIndex + offset] = particleKey;

        offset += stride;
    }

}

__global__ void TreeNS::Kernel::globalCOM(Tree *tree, Particles *particles, real com[DIM]) {

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

namespace TreeNS {

    namespace Kernel {

        __global__ void set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                                  integer *index, integer *toDeleteLeaf, integer *toDeleteNode, real *minX, real *maxX) {
            tree->set(count, start, child, sorted, index, toDeleteLeaf, toDeleteNode, minX, maxX);
        }

        __global__ void info(Tree *tree, Particles *particles, integer n, integer m) {
            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            int relevantChild = 0;
            int childIndex, temp;

            while ((bodyIndex + offset) < POW_DIM) {
                childIndex = tree->child[bodyIndex + offset];
                temp = childIndex;
                while (childIndex != -1) {
                    childIndex = tree->child[POW_DIM * childIndex + relevantChild];

                    if (childIndex != -1) {
                        if (particles->level[temp] >= particles->level[childIndex]) {
                            cudaAssert("level[%i]: %i vs. level[%i]: %i\n", temp, particles->level[temp], childIndex,
                                       particles->level[childIndex]);
                        }
                    }

                    temp = childIndex;
                }

                offset += stride;
            }

            //while ((bodyIndex + offset) < n) {
            //    if (particles->level[bodyIndex + offset] < 0) {
            //        printf("attention: level[%i] = %i\n", bodyIndex + offset,
            //               particles->level[bodyIndex + offset]);
            //        assert(0);
            //    }
            //    offset += stride;
            //}
            //offset = m;
            //while ((bodyIndex + offset) < *tree->index) {
            //    if (particles->level[bodyIndex + offset] < 0 && particles->mass[bodyIndex + offset] > 1e-4) {
            //#if DIM == 3
            //        printf("attention: level[%i] = %i (%e, %e, %e) %e\n", bodyIndex + offset,
            //               particles->level[bodyIndex + offset],
            //               particles->x[bodyIndex + offset],
            //               particles->y[bodyIndex + offset],
            //               particles->z[bodyIndex + offset],
            //               particles->mass[bodyIndex + offset]);
            //#endif
            //        assert(0);
            //    }
            //    offset += stride;
            //}

            //offset = tree->toDeleteLeaf[0];
            //while ((bodyIndex + offset) < n && (bodyIndex + offset)) {
            //    for (int i=0; i<POW_DIM; i++) {
            //        if (tree->child[POW_DIM * (bodyIndex + offset) + i] != -1) {
            //            printf("tree->child[POW_DIM * %i + %i] = %i\n", bodyIndex + offset, i, tree->child[POW_DIM * (bodyIndex + offset) + i]);
            //            assert(0);
            //        }
            //    }
            //    offset += stride;
            //}
            //offset = tree->toDeleteNode[0];
            //while ((bodyIndex + offset) < m && (bodyIndex + offset)) {
            //    for (int i=0; i<POW_DIM; i++) {
            //        if (tree->child[POW_DIM * (bodyIndex + offset) + i] != -1) {
            //            printf("tree->child[POW_DIM * %i + %i] = %i\n", bodyIndex + offset, i, tree->child[POW_DIM * (bodyIndex + offset) + i]);
            //            assert(0);
            //        }
            //    }
            //    offset += stride;
            //}

            //while (bodyIndex + offset < n) {
            //    if ((bodyIndex + offset) % 10000 == 0) {
            //        printf("tree info\n");
            //    }
            //    offset += stride;
            //}

            //bodyIndex += n;
            //while (bodyIndex + offset < m) {

                //printf("particles->mass[%i] = %f (%f, %f, %f) (%i, %i)\n", bodyIndex + offset,
                //       particles->mass[bodyIndex + offset],
                //       particles->x[bodyIndex + offset],
                //       particles->y[bodyIndex + offset],
                //       particles->z[bodyIndex + offset], n, m);

                //printf("x[%i] = (%f, %f, %f) mass = %f\n", bodyIndex + offset, particles->x[bodyIndex + offset],
                //       particles->y[bodyIndex + offset], particles->z[bodyIndex + offset],
                //       particles->mass[bodyIndex + offset]);
                //#if DIM == 1
                //printf("(%f), \n", particles->x[bodyIndex + offset]);
                //#elif DIM == 2
                //printf("(%f, %f), \n", particles->x[bodyIndex + offset],
                //       particles->y[bodyIndex + offset]);
                //#else
                //printf("(%f, %f, %f), \n", particles->x[bodyIndex + offset],
                //               particles->y[bodyIndex + offset], particles->z[bodyIndex + offset]);
                //#endif
                //offset += stride;
            //}
        }

        __global__ void info(Tree *tree, Particles *particles) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            while (bodyIndex + offset < POW_DIM) {
#if DIM == 3
                printf("child[POW_DIM * 0 + %i] = %i, x = (%f, %f, %f) m = %f\n", bodyIndex + offset,
                       tree->child[bodyIndex + offset], particles->x[tree->child[bodyIndex + offset]],
                       particles->y[tree->child[bodyIndex + offset]], particles->z[tree->child[bodyIndex + offset]],
                       particles->mass[tree->child[bodyIndex + offset]]);

                for (int i=0; i<POW_DIM; i++) {
                    printf("child[POW_DIM * %i + %i] = %i, x = (%f, %f, %f) m = %f\n", tree->child[bodyIndex + offset], i,
                           tree->child[POW_DIM * tree->child[bodyIndex + offset] + i],
                           particles->x[tree->child[POW_DIM * tree->child[bodyIndex + offset] + i]],
                           particles->y[tree->child[POW_DIM * tree->child[bodyIndex + offset] + i]],
                           particles->z[tree->child[POW_DIM * tree->child[bodyIndex + offset] + i]],
                           particles->mass[tree->child[POW_DIM * tree->child[bodyIndex + offset] + i]]);
                }
#endif

                offset += stride;
            }
        }

        __global__ void testTree(Tree *tree, Particles *particles, integer n, integer m) {

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            real mass;
            real masses[POW_DIM];

            while (bodyIndex + offset < POW_DIM) {

                mass = 0;

                for (int i=0; i<POW_DIM; i++) {
                    masses[i] = 0;
                    if (tree->child[POW_DIM * tree->child[bodyIndex + offset] + i] != -1) {
                        masses[i] = particles->mass[tree->child[POW_DIM * tree->child[bodyIndex + offset] + i]];
                        mass += masses[i];
                    }
                }
                if (mass != particles->mass[tree->child[bodyIndex + offset]]) {
                    printf("testTree: index: %i mass %f vs %f (%f, %f, %f, %f, %f, %f, %f, %f)\n", bodyIndex + offset, mass, particles->mass[tree->child[bodyIndex + offset]],
                           masses[0], masses[1], masses[2], masses[3], masses[4], masses[5], masses[6], masses[7]);
                }

                offset += stride;
            }

            //while (bodyIndex + offset < n) {
            //    if (particles->x[bodyIndex + offset] == 0.f &&
            //        particles->y[bodyIndex + offset] == 0.f &&
            //        particles->z[bodyIndex + offset] == 0.f &&
            //        particles->mass[bodyIndex + offset] == 0.f) {
            //        printf("particle ZERO for index = %i: (%f, %f, %f) %f\n", bodyIndex + offset,
            //               particles->x[bodyIndex + offset], particles->y[bodyIndex + offset],
            //               particles->z[bodyIndex + offset], particles->mass[bodyIndex + offset]);
            //    }
            //
            //    offset += stride;
            //}
            //offset = m;
            //while (bodyIndex + offset < *tree->index) {
            //    if (particles->x[bodyIndex + offset] == 0.f &&
            //        particles->y[bodyIndex + offset] == 0.f &&
            //        particles->z[bodyIndex + offset] == 0.f &&
            //        particles->mass[bodyIndex + offset] == 0.f) {
            //        printf("particle ZERO for index = %i: (%f, %f, %f) %f\n", bodyIndex + offset,
            //               particles->x[bodyIndex + offset], particles->y[bodyIndex + offset],
            //               particles->z[bodyIndex + offset], particles->mass[bodyIndex + offset]);
            //    }
            //    offset += stride;
            //}
        }

        void Launch::set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                             integer *index, integer *toDeleteLeaf, integer *toDeleteNode , real *minX, real *maxX) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::TreeNS::Kernel::set, tree, count, start, child, sorted,
                         index, toDeleteLeaf, toDeleteNode, minX, maxX);
        }

        real Launch::info(Tree *tree, Particles *particles, integer n, integer m) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::TreeNS::Kernel::info, tree, particles, n, m);
        }

        real Launch::info(Tree *tree, Particles *particles) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::TreeNS::Kernel::info, tree, particles);
        }

        real Launch::testTree(Tree *tree, Particles *particles, integer n, integer m) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::TreeNS::Kernel::testTree, tree, particles, n, m);
        }

#if DIM > 1

        __global__ void set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                                  integer *index, integer *toDeleteLeaf, integer *toDeleteNode, real *minX, real *maxX,
                                  real *minY, real *maxY) {
            tree->set(count, start, child, sorted, index, toDeleteLeaf, toDeleteNode, minX, maxX, minY, maxY);
        }

        void Launch::set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                             integer *index, integer *toDeleteLeaf, integer *toDeleteNode, real *minX, real *maxX,
                             real *minY, real *maxY) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::TreeNS::Kernel::set, tree, count, start, child, sorted, index,
                         toDeleteLeaf, toDeleteNode, minX, maxX, minY, maxY);
        }

#if DIM == 3

        __global__ void set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                                  integer *index, integer *toDeleteLeaf, integer *toDeleteNode, real *minX, real *maxX,
                                  real *minY, real *maxY, real *minZ, real *maxZ) {
            tree->set(count, start, child, sorted, index, toDeleteLeaf, toDeleteNode, minX, maxX, minY, maxY,
                      minZ, maxZ);
        }

        void Launch::set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                             integer *index, integer *toDeleteLeaf, integer *toDeleteNode, real *minX, real *maxX,
                             real *minY, real *maxY, real *minZ, real *maxZ) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::TreeNS::Kernel::set, tree, count, start, child, sorted, index,
                         toDeleteLeaf, toDeleteNode, minX, maxX, minY, maxY, minZ, maxZ);
        }

#endif
#endif

        namespace Launch {

            real sumParticles(Tree *tree) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::TreeNS::Kernel::sumParticles, tree);
            }

            real buildTree(Tree *tree, Particles *particles, integer n, integer m, bool time) {
                ExecutionPolicy executionPolicy(24, 32);
                return cuda::launch(time, executionPolicy, ::TreeNS::Kernel::buildTree, tree, particles, n, m);
            }

            real prepareSorting(Tree *tree, Particles *particles, integer n, integer m) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(time, executionPolicy, ::TreeNS::Kernel::prepareSorting, tree, particles, n, m);
            }

            real calculateCentersOfMass(Tree *tree, Particles *particles, integer n, integer level, bool time) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(time, executionPolicy, ::TreeNS::Kernel::calculateCentersOfMass, tree, particles, n, level);
            }

            real computeBoundingBox(Tree *tree, Particles *particles, integer *mutex, integer n, integer blockSize,
                                    bool time) {
                size_t sharedMemory = 2 * DIM * sizeof(real) * blockSize;
                ExecutionPolicy executionPolicy(4, 256, sharedMemory);
                return cuda::launch(time, executionPolicy, ::TreeNS::Kernel::computeBoundingBox, tree, particles, mutex,
                                    n, blockSize);
            }

            real centerOfMass(Tree *tree, Particles *particles, integer n, bool time) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(time, executionPolicy, ::TreeNS::Kernel::centerOfMass, tree, particles, n);
            }

            real sort(Tree *tree, integer n, integer m, bool time) {
                ExecutionPolicy executionPolicy(1024, 1024);
                return cuda::launch(time, executionPolicy, ::TreeNS::Kernel::sort, tree, n, m);
            }

            real getParticleKeys(Tree *tree, Particles *particles, keyType *keys, integer maxLevel, integer n,
                                 Curve::Type curveType, bool time) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(time, executionPolicy, ::TreeNS::Kernel::getParticleKeys, tree, particles, keys,
                                    maxLevel, n, curveType);
            }

            real globalCOM(Tree *tree, Particles *particles, real com[DIM]) {
                ExecutionPolicy executionPolicy(1, 1);
                return cuda::launch(true, executionPolicy, ::TreeNS::Kernel::globalCOM, tree, particles, com);
            }
        }
    }
}


#if UNIT_TESTING
namespace UnitTesting {
    namespace Kernel {

        __global__ void test_localTree(Tree *tree, Particles *particles, int n, int m) {

            int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            real childMasses;
            real positionX;
#if DIM > 1
            real positionY;
#if DIM == 3
            real positionZ;
#endif
#endif
            int childIndex;
            // check pseudo-particles
            int offset = n;
            while ((bodyIndex + offset) < m) {
                // check whether pseudo-particles are correctly calculated
                childMasses = 0.;
                positionX = 0.;
#if DIM > 1
                positionY = 0.;
#if DIM == 3
                positionZ = 0.;
#endif
#endif
                for (int child=0; child<POW_DIM; child++) {
                    childIndex = tree->child[POW_DIM * (bodyIndex + offset) + child];

                    if (childIndex != -1) {
                        childMasses += particles->mass[childIndex];
                        positionX += particles->mass[childIndex] * particles->x[childIndex];
#if DIM > 1
                        positionY += particles->mass[childIndex] * particles->y[childIndex];
#if DIM == 3
                        positionZ += particles->mass[childIndex] * particles->z[childIndex];
#endif
#endif
                    }
                }
                if (childMasses > 0.) {
                    positionX /= childMasses;
#if DIM > 1
                    positionY /= childMasses;
#if DIM == 3
                    positionZ /= childMasses;
#endif
#endif
                }
                //if (particles->nodeType[bodyIndex + offset] == 1) { // <
                //    printf("Masses [%i] ?: %e vs %e (type: %i)!\n", bodyIndex + offset, particles->mass[bodyIndex + offset], childMasses, particles->nodeType[bodyIndex + offset]);
                //}
                // now compare
                if (particles->mass[bodyIndex + offset] != childMasses) {
                    if (particles->nodeType[bodyIndex + offset] != 2) { // >=
                        printf("Masses are not correct [%i]: %e vs %e (type: %i)!\n", bodyIndex + offset, particles->mass[bodyIndex + offset], childMasses, particles->nodeType[bodyIndex + offset]);
                        //for (int child=0; child<POW_DIM; child++) {
                        //    printf("[%i] Masses: index: %i\n", bodyIndex + offset, tree->child[POW_DIM * (bodyIndex + offset) + child]);
                        //}
                    }
                }
                if (cuda::math::abs(particles->x[bodyIndex + offset] - positionX) > 1e-3) {
                    if (particles->nodeType[bodyIndex + offset] != 2) {
                        printf("Masses... X position are not correct [%i]: %e vs %e (m = %e vs %e)!\n", bodyIndex + offset,
                               particles->x[bodyIndex + offset], positionX, particles->mass[bodyIndex + offset], childMasses);
                        for (int child=0; child<POW_DIM; child++) {
                            printf("[%i] Masses: index: %i\n", bodyIndex + offset, tree->child[POW_DIM * (bodyIndex + offset) + child]);
                        }
                    }
                }
#if DIM > 1
                if (cuda::math::abs(particles->y[bodyIndex + offset] - positionY) > 1e-3) {
                    if (particles->nodeType[bodyIndex + offset] != 2) {
                        printf("Masses... Y position are not correct: %e vs %e (m = %e vs %e)!\n", particles->y[bodyIndex + offset], positionY, particles->mass[bodyIndex + offset], childMasses);
                    }
                }
#if DIM == 3
                if (cuda::math::abs(particles->z[bodyIndex + offset] - positionZ) > 1e-3) {
                    if (particles->nodeType[bodyIndex + offset] != 2) {
                        printf("Masses... Z position are not correct: %e vs %e (m = %e vs %e)!\n", particles->z[bodyIndex + offset], positionZ, particles->mass[bodyIndex + offset], childMasses);
                    }
                }
#endif
#endif
                offset += stride;
            }
        }

        namespace Launch {
            real test_localTree(Tree *tree, Particles *particles, int n, int m) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::UnitTesting::Kernel::test_localTree, tree, particles, n, m);
            }
        }
    }
}
#endif
