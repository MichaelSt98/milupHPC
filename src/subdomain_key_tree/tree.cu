#include "../../include/subdomain_key_tree/tree.cuh"

CUDA_CALLABLE_MEMBER Foo::Foo() {

}

CUDA_CALLABLE_MEMBER Foo::Foo(int *test) : d_test(test) {

}

CUDA_CALLABLE_MEMBER Foo::~Foo() {

}

CUDA_CALLABLE_MEMBER void Foo::aMethod(int *test) {
    d_test = test;
}

__global__ void setKernel(Foo *foo, int *test) {
    foo->d_test = test;
}

__global__ void testKernel(Foo *foo) {

    for (int i=0; i<5; i++) {
        foo->d_test[i] = i;
        printf("<<<testKernel>>> test = %i\n", foo->d_test[i]);
    }

}

void launchTestKernel(Foo *foo) {
    //ExecutionPolicy executionPolicy(1, 1);
    //cudaLaunch(false, executionPolicy, testKernel, foo);
    testKernel<<<1, 1>>>(foo);
}

void launchSetKernel(Foo *foo, int *test) {
    setKernel<<<1, 1>>>(foo, test);
}

/*__global__ void buildTreeKernel(float *x, float *y, float *z, float *mass, int *count, int *start,
                                int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                float *minZ, float *maxZ, int n, int m) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    //note: -1 used as "null pointer"
    //note: -2 used to lock a child (pointer)

    int offset;
    bool newBody = true;

    float min_x;
    float max_x;
    float min_y;
    float max_y;
//#if DIM == 3
    float min_z;
    float max_z;
//#endif

    int childPath;
    int temp;
    int tempTemp;

    offset = 0;

    while ((bodyIndex + offset) < n) {

        if (newBody) {

            newBody = false;

            // copy bounding box
            min_x = *minX;
            max_x = *maxX;
            min_y = *minY;
            max_y = *maxY;
            min_z = *minZ;
            max_z = *maxZ;

            temp = 0;
            childPath = 0;

            // find insertion point for body
            if (x[bodyIndex + offset] < 0.5 * (min_x + max_x)) { // x direction
                childPath += 1;
                max_x = 0.5 * (min_x + max_x);
            }
            else {
                min_x = 0.5 * (min_x + max_x);
            }
            if (y[bodyIndex + offset] < 0.5 * (min_y + max_y)) { // y direction
                childPath += 2;
                max_y = 0.5 * (min_y + max_y);
            }
            else {
                min_y = 0.5 * (min_y + max_y);
            }
            if (z[bodyIndex + offset] < 0.5 * (min_z + max_z)) {  // z direction
                childPath += 4;
                max_z = 0.5 * (min_z + max_z);
            }
            else {
                min_z = 0.5 * (min_z + max_z);
            }
        }

        int childIndex = child[temp*8 + childPath];

        // traverse tree until hitting leaf node
        while (childIndex >= m) { //n

            tempTemp = temp;
            temp = childIndex;

            childPath = 0;

            // find insertion point for body
            if (x[bodyIndex + offset] < 0.5 * (min_x + max_x)) { // x direction
                childPath += 1;
                max_x = 0.5 * (min_x + max_x);
            }
            else {
                min_x = 0.5 * (min_x + max_x);
            }
            if (y[bodyIndex + offset] < 0.5 * (min_y + max_y)) { // y direction
                childPath += 2;
                max_y = 0.5 * (min_y + max_y);
            }
            else {
                min_y = 0.5 * (min_y + max_y);
            }
            if (z[bodyIndex + offset] < 0.5 * (min_z + max_z)) { // z direction
                childPath += 4;
                max_z = 0.5 * (min_z + max_z);
            }
            else {
                min_z = 0.5 * (min_z + max_z);
            }

            if (mass[bodyIndex + offset] != 0) {
                atomicAdd(&x[temp], mass[bodyIndex + offset] * x[bodyIndex + offset]);
                atomicAdd(&y[temp], mass[bodyIndex + offset] * y[bodyIndex + offset]);
                atomicAdd(&z[temp], mass[bodyIndex + offset] * z[bodyIndex + offset]);
            }

            atomicAdd(&mass[temp], mass[bodyIndex + offset]);
            atomicAdd(&count[temp], 1);

            childIndex = child[8*temp + childPath];
        }

        // if child is not locked
        if (childIndex != -2) {

            int locked = temp * 8 + childPath;

            if (atomicCAS(&child[locked], childIndex, -2) == childIndex) {

                // check whether a body is already stored at the location
                if (childIndex == -1) {
                    //insert body and release lock
                    child[locked] = bodyIndex + offset;
                }
                else {
                    if (childIndex >= n) {
                        printf("ATTENTION!\n");
                    }
                    int patch = 8 * m; //8*n
                    while (childIndex >= 0 && childIndex < n) { // was n

                        //create a new cell (by atomically requesting the next unused array index)
                        int cell = atomicAdd(index, 1);
                        patch = min(patch, cell);

                        if (patch != cell) {
                            child[8 * temp + childPath] = cell;
                        }

                        // insert old/original particle
                        childPath = 0;
                        if (x[childIndex] < 0.5 * (min_x + max_x)) { childPath += 1; }
                        if (y[childIndex] < 0.5 * (min_y + max_y)) { childPath += 2; }
                        if (z[childIndex] < 0.5 * (min_z + max_z)) { childPath += 4; }

                        x[cell] += mass[childIndex] * x[childIndex];
                        y[cell] += mass[childIndex] * y[childIndex];
                        z[cell] += mass[childIndex] * z[childIndex];

                        mass[cell] += mass[childIndex];
                        count[cell] += count[childIndex];

                        child[8 * cell + childPath] = childIndex;

                        start[cell] = -1;

                        // insert new particle
                        tempTemp = temp;
                        temp = cell;
                        childPath = 0;

                        // find insertion point for body
                        if (x[bodyIndex + offset] < 0.5 * (min_x + max_x)) {
                            childPath += 1;
                            max_x = 0.5 * (min_x + max_x);
                        } else {
                            min_x = 0.5 * (min_x + max_x);
                        }
                        if (y[bodyIndex + offset] < 0.5 * (min_y + max_y)) {
                            childPath += 2;
                            max_y = 0.5 * (min_y + max_y);
                        } else {
                            min_y = 0.5 * (min_y + max_y);
                        }
                        if (z[bodyIndex + offset] < 0.5 * (min_z + max_z)) {
                            childPath += 4;
                            max_z = 0.5 * (min_z + max_z);
                        } else {
                            min_z = 0.5 * (min_z + max_z);
                        }

                        // COM / preparing for calculation of COM
                        if (mass[bodyIndex + offset] != 0) {
                            x[cell] += mass[bodyIndex + offset] * x[bodyIndex + offset];
                            y[cell] += mass[bodyIndex + offset] * y[bodyIndex + offset];
                            z[cell] += mass[bodyIndex + offset] * z[bodyIndex + offset];
                            mass[cell] += mass[bodyIndex + offset];
                        }
                        count[cell] += count[bodyIndex + offset];
                        childIndex = child[8 * temp + childPath];
                    }

                    child[8 * temp + childPath] = bodyIndex + offset;

                    __threadfence();  // written to global memory arrays (child, x, y, mass) thus need to fence
                    child[locked] = patch;
                }
                offset += stride;
                newBody = true;
            }
        }
        __syncthreads();
    }
}*/
