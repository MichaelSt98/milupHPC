//
// Created by Michael Staneker on 12.08.21.
//

#include "../include/particles.cuh"
#include "../include/cuda_utils/cuda_launcher.cuh"

CUDA_CALLABLE_MEMBER Particles::Particles() {

}

CUDA_CALLABLE_MEMBER Particles::Particles(integer *numParticles, integer *numNodes, real *mass, real *x, real *vx, real *ax) :
                                    numParticles(numParticles), numNodes(numNodes), mass(mass), x(x), vx(vx), ax(ax) {

}

CUDA_CALLABLE_MEMBER void Particles::set(integer *numParticles, integer *numNodes, real *mass, real *x, real *vx, real *ax) {

    this->numParticles = numParticles;
    this->numNodes = numNodes;
    this->mass = mass;
    this->x = x;
    this->vx = vx;
    this->ax = ax;

}

#if DIM > 1
CUDA_CALLABLE_MEMBER Particles::Particles(integer *numParticles, integer *numNodes, real *mass, real *x, real *y,
                                          real *vx, real *vy, real *ax, real *ay) : numParticles(numParticles),
                                          numNodes(numNodes), mass(mass),
                                          x(x), y(y), vx(vx), vy(vy), ax(ax), ay(ay) {

}

CUDA_CALLABLE_MEMBER void Particles::set(integer *numParticles, integer *numNodes, real *mass, real *x, real *y, real *vx, real *vy,
                                      real *ax, real *ay) {

    this->numParticles = numParticles;
    this->numNodes = numNodes;
    this->mass = mass;
    this->x = x;
    this->y = y;
    this->vx = vx;
    this->vy = vy;
    this->ax = ax;
    this->ay = ay;

}
#if DIM == 3
CUDA_CALLABLE_MEMBER Particles::Particles(integer *numParticles, integer *numNodes, real *mass, real *x, real *y, real *z, real *vx, real *vy,
                                          real *vz, real *ax, real *ay, real *az) : numParticles(numParticles),
                                                                                    numNodes(numNodes), mass(mass), x(x), y(y), z(z),
                                                                                    vx(vx), vy(vy), vz(vz), ax(ax), ay(ay), az(az) {

}

CUDA_CALLABLE_MEMBER void Particles::set(integer *numParticles, integer *numNodes, real *mass, real *x, real *y, real *z, real *vx,
                                                 real *vy, real *vz, real *ax, real *ay, real *az) {

    this->numParticles = numParticles;
    this->numNodes = numNodes;
    this->mass = mass;
    this->x = x;
    this->y = y;
    this->z = z;
    this->vx = vx;
    this->vy = vy;
    this->vz = vz;
    this->ax = ax;
    this->ay = ay;
    this->az = az;

}
#endif
#endif

CUDA_CALLABLE_MEMBER void Particles::reset(integer index) {
    x[index] = 0;
#if DIM > 1
    y[index] = 0;
#if DIM == 3
    z[index] = 0;
#endif
#endif
    mass[index] = 0;
}

CUDA_CALLABLE_MEMBER Particles::~Particles() {

}


namespace ParticlesNS {

    namespace Kernel {

        __global__ void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x,
                            real *vx, real *ax) {

            particles->set(numParticles, numNodes, mass, x, vx, ax);

        }

        void Launch::set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x,
                                real *vx, real *ax) {

            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::set, particles, numParticles, numNodes, mass, x, vx, ax);

        }


#if DIM > 1

        __global__ void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x,
                            real *y, real *vx, real *vy, real *ax, real *ay) {

            particles->set(numParticles, numNodes, mass, x, y, vx, vy, ax, ay);

        }

        void Launch::set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x, real *y,
                         real *vx, real *vy, real *ax, real *ay) {

            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::set, particles, numParticles, numNodes,
                         mass, x, y, vx, vy, ax, ay);

        }


#if DIM == 3

        __global__ void set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x,
                                  real *y, real *z, real *vx, real *vy, real *vz, real *ax, real *ay, real *az) {

            particles->set(numParticles, numNodes, mass, x, y, z, vx, vy, vz, ax, ay, az);

        }

        void Launch::set(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x, real *y,
                         real *z, real *vx, real *vy, real *vz, real *ax, real *ay, real *az) {

            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::ParticlesNS::Kernel::set, particles, numParticles, numNodes,
                         mass, x, y, z, vx, vy, vz, ax, ay, az);
            //setKernel<<<1, 1>>>(particles, count, mass, x, y, z, vx, vy, vz, ax, ay, az);

        }

#endif
#endif


        __global__ void test(Particles *particles) {

            int bodyIndex = threadIdx.x + blockDim.x * blockIdx.x;
            int stride = blockDim.x * gridDim.x;
            int offset = 0;

            if (bodyIndex == 0) {
                printf("device: numParticles = %i\n", *particles->numParticles);
            }

            while ((bodyIndex + offset) < *particles->numParticles) {
                if ((bodyIndex + offset) % 10000 == 0) {
                    printf("device: x[%i] = (%f, %f, %f)\n", bodyIndex + offset, particles->x[bodyIndex + offset],
                           particles->y[bodyIndex + offset],
                           particles->z[bodyIndex + offset]);
                }
                offset += stride;
            }


        }

        real Launch::test(Particles *particles, bool time) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(time, executionPolicy, ::ParticlesNS::Kernel::test, particles);
            //testKernel<<<256, 256>>>(particles);
        }
    }

}
