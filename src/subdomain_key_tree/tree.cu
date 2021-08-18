#include "../../include/subdomain_key_tree/tree.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

/*void launchBuildTreeKernel(Foo *foo) {
    ExecutionPolicy executionPolicy(1, 1);
    cudaLaunch(false, executionPolicy, testKernel, foo);
    //testKernel<<<1, 1>>>(foo);
}*/

CUDA_CALLABLE_MEMBER Tree::Tree() {

}

CUDA_CALLABLE_MEMBER Tree::Tree(integer *count, integer *start, integer *child, integer *sorted, integer *index,
                                real *minX, real *maxX) : count(count), start(start), child(child), sorted(sorted),
                                index(index), minX(minX), maxX(maxX) {

}
CUDA_CALLABLE_MEMBER void Tree::set(integer *count, integer *start, integer *child, integer *sorted,
                                        integer *index, real *minX, real *maxX) {
    this->count = count;
    this->start = start;
    this->child = child;
    this->sorted = sorted;
    this->index = index;
    this->minX = minX;
    this->maxX = maxX;
}

#if DIM > 1
CUDA_CALLABLE_MEMBER Tree::Tree(integer *count, integer *start, integer *child, integer *sorted, integer *index,
                                real *minX, real *maxX, real *minY, real *maxY) : count(count), start(start),
                                child(child), sorted(sorted), index(index), minX(minX), maxX(maxX), minY(minY),
                                maxY(maxY) {

}
CUDA_CALLABLE_MEMBER void Tree::set(integer *count, integer *start, integer *child, integer *sorted,
                                        integer *index, real *minX, real *maxX, real *minY, real *maxY) {
    this->count = count;
    this->start = start;
    this->child = child;
    this->sorted = sorted;
    this->index = index;
    this->minX = minX;
    this->maxX = maxX;
    this->minY = minY;
    this->maxY = maxY;
}

#if DIM == 3
CUDA_CALLABLE_MEMBER Tree::Tree(integer *count, integer *start, integer *child, integer *sorted, integer *index,
                                real *minX, real *maxX, real *minY, real *maxY, real *minZ, real *maxZ) : count(count),
                                start(start), child(child), sorted(sorted), index(index), minX(minX), maxX(maxX),
                                minY(minY), maxY(maxY), minZ(minZ), maxZ(maxZ) {

}
CUDA_CALLABLE_MEMBER void Tree::set(integer *count, integer *start, integer *child, integer *sorted,
                                        integer *index, real *minX, real *maxX, real *minY, real *maxY,
                                        real *minZ, real *maxZ) {
    this->count = count;
    this->start = start;
    this->child = child;
    this->sorted = sorted;
    this->index = index;
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
    #pragma unroll 8
    for (integer i=0; i<POW_DIM; i++) {
        child[index * POW_DIM + i] = -1;
    }

    if (index < n) {
        count[index] = 0;
    }
    else {
        count[index] = 0;
    }
    start[index] = 0;
}

CUDA_CALLABLE_MEMBER keyType Tree::getParticleKey(Particles *particles, integer index, integer maxLevel) {

    int level = 0;
    keyType particleKey = (keyType)0;

    int sonBox = 0;
    float min_x = *minX;
    float max_x = *maxX;
    float min_y = *minY;
    float max_y = *maxY;
    float min_z = *minZ;
    float max_z = *maxZ;

    // calculate path to the particle's position assuming an octree with above bounding boxes
    while (level <= maxLevel) {
        sonBox = 0;
        // find insertion point for body
        if (particles->x[index] < 0.5 * (min_x+max_x)) {
            sonBox += 1;
            max_x = 0.5 * (min_x+max_x);
        }
        else { min_x = 0.5 * (min_x+max_x); }
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
        particleKey = particleKey | ((unsigned long)sonBox << (unsigned long)(DIM * (maxLevel-level-1)));
        level ++;
    }
    //TODO: Hilbert change
    return particleKey;
    //return Lebesgue2Hilbert(particleKey, 21);
}

CUDA_CALLABLE_MEMBER Tree::~Tree() {

}

__global__ void TreeNS::Kernel::computeBoundingBox(Tree *tree, Particles *particles, integer *mutex, integer n,
                                                 integer blockSize) {

    integer index = threadIdx.x + blockDim.x * blockIdx.x;
    integer stride = blockDim.x * gridDim.x;

    // initialize local min/max
    //if (particles->x[index] != 0.f || particles->y[index] != 0.f || particles->z[index] != 0.f) {
    //    printf("device: x = (%f, %f, %f)\n", particles->x[index], particles->y[index], particles->z[index]);
    //}

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

    //if (index % 1000 == 0) {
    //    printf("device: x_min[%i] = %f\n", index, x_min);
    //}

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

        x_min = fminf(x_min, particles->x[index + offset]);
        x_max = fmaxf(x_max, particles->x[index + offset]);
#if DIM > 1
        y_min = fminf(y_min, particles->y[index + offset]);
        y_max = fmaxf(y_max, particles->y[index + offset]);
#if DIM == 3
        z_min = fminf(z_min, particles->z[index + offset]);
        z_max = fmaxf(z_max, particles->z[index + offset]);
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
            x_min_buffer[threadIdx.x] = fminf(x_min_buffer[threadIdx.x], x_min_buffer[threadIdx.x + i]);
            x_max_buffer[threadIdx.x] = fmaxf(x_max_buffer[threadIdx.x], x_max_buffer[threadIdx.x + i]);
#if DIM > 1
            y_min_buffer[threadIdx.x] = fminf(y_min_buffer[threadIdx.x], y_min_buffer[threadIdx.x + i]);
            y_max_buffer[threadIdx.x] = fmaxf(y_max_buffer[threadIdx.x], y_max_buffer[threadIdx.x + i]);
#if DIM == 3
            z_min_buffer[threadIdx.x] = fminf(z_min_buffer[threadIdx.x], z_min_buffer[threadIdx.x + i]);
            z_max_buffer[threadIdx.x] = fmaxf(z_max_buffer[threadIdx.x], z_max_buffer[threadIdx.x + i]);
#endif
#endif
        }
        __syncthreads();
        i /= 2;
    }

    // combining the results and generate the root cell
    if (threadIdx.x == 0) {
        while (atomicCAS(mutex, 0 ,1) != 0); // lock

        *tree->minX = fminf(*tree->minX, x_min_buffer[0]);
        *tree->maxX = fmaxf(*tree->maxX, x_max_buffer[0]);
#if DIM > 1
        *tree->minY = fminf(*tree->minY, y_min_buffer[0]);
        *tree->maxY = fmaxf(*tree->maxY, y_max_buffer[0]);
#if DIM == 3
        *tree->minZ = fminf(*tree->minZ, z_min_buffer[0]);
        *tree->maxZ = fmaxf(*tree->maxZ, z_max_buffer[0]);
#endif
#endif

        //if (*tree->minX != 0.f || *tree->minX) {
        //    printf("device: min/max: x = (%f, %f), y = (%f, %f), z = (%f, %f)\n", *tree->minX, *tree->maxX,
        //           *tree->minY, *tree->maxY, *tree->minZ, *tree->maxZ);
        //}

        atomicExch(mutex, 0); // unlock
    }
}

__global__ void TreeNS::Kernel::buildTree(Tree *tree, Particles *particles, integer n, integer m) {

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
    integer tempTemp;

    offset = 0;

    while ((bodyIndex + offset) < n) {

        if (newBody) {

            newBody = false;

            // copy bounding box
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

            tempTemp = temp;
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
                atomicAdd(&particles->x[temp], particles->mass[bodyIndex + offset] * particles->x[bodyIndex + offset]);
#if DIM > 1
                atomicAdd(&particles->y[temp], particles->mass[bodyIndex + offset] * particles->y[bodyIndex + offset]);
#if DIM == 3
                atomicAdd(&particles->z[temp], particles->mass[bodyIndex + offset] * particles->z[bodyIndex + offset]);
#endif
#endif
            }

            atomicAdd(&particles->mass[temp], particles->mass[bodyIndex + offset]);
            atomicAdd(&tree->count[temp], 1);

            childIndex = tree->child[8*temp + childPath];
        }

        // if child is not locked
        if (childIndex != -2) {

            integer locked = temp * 8 + childPath;

            if (atomicCAS(&tree->child[locked], childIndex, -2) == childIndex) {

                // check whether a body is already stored at the location
                if (childIndex == -1) {
                    //insert body and release lock
                    tree->child[locked] = bodyIndex + offset;
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
                            tree->child[8 * temp + childPath] = cell;
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

                        particles->x[cell] += particles->mass[childIndex] * particles->x[childIndex];
#if DIM > 1
                        particles->y[cell] += particles->mass[childIndex] * particles->y[childIndex];
#if DIM == 3
                        particles->z[cell] += particles->mass[childIndex] * particles->z[childIndex];
#endif
#endif

                        particles->mass[cell] += particles->mass[childIndex];
                        tree->count[cell] += tree->count[childIndex];

                        tree->child[POW_DIM * cell + childPath] = childIndex;
                        tree->start[cell] = -1;

                        // insert new particle
                        tempTemp = temp;
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

__global__ void TreeNS::Kernel::centerOfMass(Tree *tree, Particles *particles, integer n) {

    integer bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
    integer stride = blockDim.x*gridDim.x;
    integer offset = 0;

    //note: most of it already done within buildTreeKernel
    bodyIndex += n;

    while (bodyIndex + offset < *tree->index) {

        //TODO: check for mass = 0
        //if (particles->mass[bodyIndex + offset] == 0) {
        //    printf("centreOfMassKernel: mass = 0 (%i)!\n", bodyIndex + offset);
        //}

        if (particles->mass != 0) {
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

__global__ void TreeNS::Kernel::sort(Tree *tree, integer n, integer m) {

    integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    integer stride = blockDim.x * gridDim.x;
    integer offset = 0;

    if (bodyIndex == 0) {
        integer sumParticles = 0;
        for (integer i=0; i<POW_DIM; i++) {
            sumParticles += tree->count[tree->child[i]];
        }
        printf("sumParticles = %i\n", sumParticles);
    }

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
    integer ind = *tree->index;

    //integer counter = 0; // for debugging purposes or rather to achieve the kernel to be finished
    while ((cell + offset) < ind /*&& counter < 100000*/) {
        //counter++;

        //if (counter > 99998) {
        //printf("cell + offset = %i\n", cell+offset);
        //}

        s = tree->start[cell + offset];

        if (s >= 0) {

            for (integer i=0; i<8; i++) {
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

namespace TreeNS {

    namespace Kernel {

        __global__ void set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                                  integer *index, real *minX, real *maxX) {
            tree->set(count, start, child, sorted, index, minX, maxX);
        }

        void Launch::set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                             integer *index, real *minX, real *maxX) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::TreeNS::Kernel::set, tree, count, start, child, sorted,
                         index, minX, maxX);
        }

#if DIM > 1

        __global__ void set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                                  integer *index, real *minX, real *maxX, real *minY, real *maxY) {
            tree->set(count, start, child, sorted, index, minX, maxX, minY, maxY);
        }

        void Launch::set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                             integer *index, real *minX, real *maxX, real *minY, real *maxY) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::TreeNS::Kernel::set, tree, count, start, child, sorted, index,
                         minX, maxX, minY, maxY);
        }

#if DIM == 3

        __global__ void set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                                  integer *index, real *minX, real *maxX, real *minY, real *maxY,
                                  real *minZ, real *maxZ) {
            tree->set(count, start, child, sorted, index, minX, maxX, minY, maxY, minZ, maxZ);
        }

        void Launch::set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                             integer *index, real *minX, real *maxX, real *minY, real *maxY,
                             real *minZ, real *maxZ) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::TreeNS::Kernel::set, tree, count, start, child, sorted, index, minX,
                         maxX, minY, maxY, minZ, maxZ);
        }

#endif
#endif


        __global__ void getParticleKeys(Tree *tree, Particles *particles, keyType *keys, integer maxLevel,
                                              integer n) {

            int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;
            int offset = 0;

            unsigned long particleKey;
            unsigned long hilbertParticleKey;


            while (bodyIndex + offset < n) {

                //particleKey = 0UL;
                particleKey = tree->getParticleKey(particles, bodyIndex + offset, maxLevel);

                //TODO: Hilbert key
                //hilbertParticleKey = Lebesgue2Hilbert(particleKey, 21);
                keys[bodyIndex + offset] = particleKey; //hilbertParticleKey;

                offset += stride;
            }
        }

        namespace Launch {
            real buildTree(Tree *tree, Particles *particles, integer n, integer m, bool time) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(time, executionPolicy, ::TreeNS::Kernel::buildTree, tree, particles, n, m);
            }

            real computeBoundingBox(Tree *tree, Particles *particles, integer *mutex, integer n, integer blockSize,
                                    bool time) {
                size_t sharedMemory = 6 * sizeof(real) * blockSize;
                //ExecutionPolicy executionPolicy(1024, 256, sharedMemory);
                ExecutionPolicy executionPolicy(256, 256, sharedMemory);
                return cuda::launch(time, executionPolicy, ::TreeNS::Kernel::computeBoundingBox, tree, particles, mutex, n, blockSize);
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
                                 bool time) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(time, executionPolicy, ::TreeNS::Kernel::getParticleKeys, tree, particles, keys, maxLevel, n);
            }
        }
    }
}
