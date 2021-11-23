#include "../../include/processing/kernels.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

namespace Processing {

    namespace Kernel {

        __global__ void particlesWithinRadii(Particles *particles, int *particlesWithin, real deltaRadial, int n) {

            int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;
            int offset = 0;

            real r;
            int index;

            while ((bodyIndex + offset) < n) {

#if DIM == 1
                r = cuda::math::sqrt(particles->x[bodyIndex + offset] * particles->x[bodyIndex + offset]);
#elif DIM == 2
                r = cuda::math::sqrt(particles->x[bodyIndex + offset] * particles->x[bodyIndex + offset] +
                                        particles->y[bodyIndex + offset] * particles->y[bodyIndex + offset]);
#else
                r = cuda::math::sqrt(particles->x[bodyIndex + offset] * particles->x[bodyIndex + offset] +
                                     particles->y[bodyIndex + offset] * particles->y[bodyIndex + offset] +
                                     particles->z[bodyIndex + offset] * particles->z[bodyIndex + offset]);
#endif

                index = (int) (r / deltaRadial);
                atomicAdd(&particlesWithin[index], 1);

                offset += stride;
            }

        }

        template<typename T>
        __global__ void
        cartesianToRadial(Particles *particles, int *particlesWithin, T *input, T *output, real deltaRadial, int n) {

            int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;
            int offset = 0;

            real r;
            int index;

            while ((bodyIndex + offset) < n) {

#if DIM == 1
                r = cuda::math::sqrt(particles->x[bodyIndex + offset] * particles->x[bodyIndex + offset]);
#elif DIM == 2
                r = cuda::math::sqrt(particles->x[bodyIndex + offset] * particles->x[bodyIndex + offset] +
                                        particles->y[bodyIndex + offset] * particles->y[bodyIndex + offset]);
#else
                r = cuda::math::sqrt(particles->x[bodyIndex + offset] * particles->x[bodyIndex + offset] +
                                     particles->y[bodyIndex + offset] * particles->y[bodyIndex + offset] +
                                     particles->z[bodyIndex + offset] * particles->z[bodyIndex + offset]);
#endif

                index = (int) (r / deltaRadial);

                if (particlesWithin[index] > 0) {
                    output[index] += input[bodyIndex + offset] / particlesWithin[index];
                }

                offset += stride;
            }

        }


        namespace Launch {
            void particlesWithinRadii(Particles *particles, int *particlesWithin, real deltaRadial, int n) {
                ExecutionPolicy executionPolicy;
                cuda::launch(false, executionPolicy, ::Processing::Kernel::particlesWithinRadii, particles, particlesWithin, deltaRadial, n);
            }

            template<typename T>
            void cartesianToRadial(Particles *particles, int *particlesWithin, T *input, T *output, real deltaRadial, int n) {
                ExecutionPolicy executionPolicy;
                cuda::launch(false, executionPolicy, ::Processing::Kernel::cartesianToRadial<T>, particles, particlesWithin, input, output, deltaRadial, n);
            }
            template void cartesianToRadial<real>(Particles *particles, int *particlesWithin, real *input, real *output, real deltaRadial, int n);
        }

    }
}
