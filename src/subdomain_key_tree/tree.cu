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

    //integer *path = new integer[maxLevel];
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
    printf("ATTENTION: level = -1 (index = %i x = (%f, %f, %f) %f) tree index = %i\n",
           index, particles->x[index], particles->y[index], particles->z[index], particles->mass[index], *this->index);
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

    //delete [] path;

    return -1;
}

//TODO: is this still working, since count only used within buildTree (probably yes)
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

        //if (std::isinf(particles->x[index + offset]) || std::isinf(particles->y[index + offset]) || std::isinf(particles->z[index + offset])) {
        //    printf("x[%i]= (%e, %e, %e)\n", index + offset, particles->x[index + offset], particles->y[index + offset], particles->z[index + offset]);
        //    assert(0);
        //}

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
            //if (std::isinf(x_min_buffer[threadIdx.x]) || std::isinf(x_max_buffer[threadIdx.x]) ||
            //        std::isinf(y_min_buffer[threadIdx.x]) || std::isinf(y_max_buffer[threadIdx.x]) ||
            //        std::isinf(z_min_buffer[threadIdx.x]) || std::isinf(z_max_buffer[threadIdx.x])) {
            //    printf("%i: min_buffer= (%e, %e, %e) max_buffer = (%e, %e, %e)\n", threadIdx.x,
            //           x_min_buffer[threadIdx.x], y_min_buffer[threadIdx.x], z_min_buffer[threadIdx.x],
            //           x_max_buffer[threadIdx.x], y_max_buffer[threadIdx.x], z_max_buffer[threadIdx.x]);
            //    assert(0);
            //}
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

        //if (std::isinf(*tree->minX) || std::isinf(*tree->maxX) ||
        //        std::isinf(*tree->minY) || std::isinf(*tree->maxY) ||
        //        std::isinf(*tree->minZ) ||std::isinf(*tree->maxZ)) {
        //    printf("tree->minX = %e | tree->maxX = %e\n", *tree->minX, *tree->maxX);
        //    printf("tree->minY = %e | tree->maxY = %e\n", *tree->minY, *tree->maxY);
        //    printf("tree->minZ = %e | tree->maxZ = %e\n", *tree->minZ, *tree->maxZ);
        //    assert(0);
        //}
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

                        particles->level[temp] = level;
                        level++;
                        if (level > MAX_LEVEL) {
                            printf("level = %i for index %i (%f, %f, %f)\n", level,
                                   bodyIndex + offset, particles->x[bodyIndex + offset],
                                   particles->y[bodyIndex + offset], particles->z[bodyIndex + offset]);
                            assert(0);
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
                        particles->x[cell] = particles->x[childIndex]; //0.5 * (min_x + max_x);
#if DIM > 1
                        particles->y[cell] = particles->y[childIndex]; //0.5 * (min_y + max_y);
#if DIM == 3
                        particles->z[cell] = particles->z[childIndex]; //0.5 * (min_z + max_z);
#endif
#endif

                        tree->count[cell] += tree->count[childIndex];

                        tree->child[POW_DIM * cell + childPath] = childIndex;
                        particles->level[cell] = level;
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
        __syncthreads(); //TODO: __syncthreads() needed?
    }
}


__global__ void TreeNS::Kernel::buildTreeMiluphcuda(Tree *tree, Particles *particles, integer n, integer m) {

}

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

    integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    integer stride = blockDim.x * gridDim.x;

    integer offset = n;

    integer index;

    while ((bodyIndex + offset) < *tree->index) {

        if (particles->level[bodyIndex + offset] == level) {

            if (particles->level[bodyIndex + offset] == -1 || particles->level[bodyIndex + offset] > 21) {
                printf("level[%i] = %i!!!\n", bodyIndex + offset, particles->level[bodyIndex + offset]);
                //assert(0);
            }

            // reset entries
            particles->mass[bodyIndex + offset] = 0.;
            particles->x[bodyIndex + offset] = 0.;
#if DIM > 1
            particles->y[bodyIndex + offset] = 0.;
#if DIM == 3
            particles->z[bodyIndex + offset] = 0.;
#endif
#endif

            // loop over children and add contribution (*=position(child) * mass(child))
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

            // finish center of mass calculation by dividing with mass
            if (particles->mass[bodyIndex + offset] > 0.) {
                particles->x[bodyIndex + offset] /= particles->mass[bodyIndex + offset];
#if DIM > 1
                particles->y[bodyIndex + offset] /= particles->mass[bodyIndex + offset];
#if DIM == 3
                particles->z[bodyIndex + offset] /= particles->mass[bodyIndex + offset];
#endif
#endif
            }

            //if (std::isnan(particles->x[bodyIndex + offset])) {
            //    printf("NAN! within calculateCenterOfMasses for %i: x = %f, m = %f\n", bodyIndex + offset, particles->x[bodyIndex + offset],
            //           particles->mass[bodyIndex + offset]);
            //    assert(0);
            //}

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

// TODO: currently unused if (computeForces() used this need to be done before)
__global__ void TreeNS::Kernel::sort(Tree *tree, integer n, integer m) {

    integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    integer stride = blockDim.x * gridDim.x;
    integer offset = 0;

    integer s = 0;
    if (threadIdx.x == 0) {

        for (integer i=0; i<POW_DIM; i++){

            integer node = tree->child[i];
            // not a leaf node
            if (node >= m) { //n
                tree->start[node] = s;
                s += tree->count[node];
            }
                // leaf node
            else if (node >= 0) {
                tree->sorted[s] = node;
                s++;
            }
        }
    }
    integer cell = m + bodyIndex;
    //integer ind = *tree->index;
    integer ind = tree->toDeleteNode[1];

    while ((cell + offset) < ind) {

        s = tree->start[cell + offset];

        if (s >= 0) {

            for (integer i=0; i<POW_DIM; i++) {
                integer node = tree->child[POW_DIM*(cell+offset) + i];
                // not a leaf node
                if (node >= m) { //m
                    tree->start[node] = s;
                    s += tree->count[node];
                }
                // leaf node
                else if (node >= 0) {
                    tree->sorted[s] = node;
                    s++;
                }
            }
            offset += stride;
        }
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
#if DIM == 3
        if (particleKey == 1UL) {
            printf("particleKey = %lu (%f, %f, %f)\n", particleKey, particles->x[bodyIndex + offset],
                   particles->y[bodyIndex + offset], particles->z[bodyIndex + offset]);
        }
#endif

        //if ((bodyIndex + offset) % 100 == 0) {
        //    printf("key = %lu\n", particleKey);
        //}

        keys[bodyIndex + offset] = particleKey;

        offset += stride;
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

            offset = tree->toDeleteLeaf[0];
            while ((bodyIndex + offset) < n && (bodyIndex + offset)) {
                for (int i=0; i<POW_DIM; i++) {
                    if (tree->child[POW_DIM * (bodyIndex + offset) + i] != -1) {
                        printf("tree->child[POW_DIM * %i + %i] = %i\n", bodyIndex + offset, i, tree->child[POW_DIM * (bodyIndex + offset) + i]);
                        assert(0);
                    }
                }
                offset += stride;
            }
            offset = tree->toDeleteNode[0];
            while ((bodyIndex + offset) < m && (bodyIndex + offset)) {
                for (int i=0; i<POW_DIM; i++) {
                    if (tree->child[POW_DIM * (bodyIndex + offset) + i] != -1) {
                        printf("tree->child[POW_DIM * %i + %i] = %i\n", bodyIndex + offset, i, tree->child[POW_DIM * (bodyIndex + offset) + i]);
                        assert(0);
                    }
                }
                offset += stride;
            }
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
                ExecutionPolicy executionPolicy;
                return cuda::launch(time, executionPolicy, ::TreeNS::Kernel::buildTree, tree, particles, n, m);
            }

            real buildTreeMiluphcuda(Tree *tree, Particles *particles, integer n, integer m, bool time) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(time, executionPolicy, ::TreeNS::Kernel::buildTreeMiluphcuda, tree, particles, n, m);
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
                ExecutionPolicy executionPolicy;
                return cuda::launch(time, executionPolicy, ::TreeNS::Kernel::sort, tree, n, m);
            }

            real getParticleKeys(Tree *tree, Particles *particles, keyType *keys, integer maxLevel, integer n,
                                 Curve::Type curveType, bool time) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(time, executionPolicy, ::TreeNS::Kernel::getParticleKeys, tree, particles, keys,
                                    maxLevel, n, curveType);
            }
        }
    }
}
