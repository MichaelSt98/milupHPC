#ifndef MILUPHPC_DEVICE_PREDICTOR_CORRECTOR_EULER_CUH
#define MILUPHPC_DEVICE_PREDICTOR_CORRECTOR_EULER_CUH

#include "../particles.cuh"
#include "../materials/material.cuh"
#include "../simulation_time.cuh"

namespace PredictorCorrectorEulerNS {

    struct Shared {
        real *forces;
        real *courant;
        real *artVisc;
        real *e;
        real *rho;

        CUDA_CALLABLE_MEMBER Shared();
        CUDA_CALLABLE_MEMBER Shared(real *forces, real *courant, real *artVisc);
        CUDA_CALLABLE_MEMBER ~Shared();

        CUDA_CALLABLE_MEMBER void set(real *forces, real *courant, real *artVisc);
        CUDA_CALLABLE_MEMBER void setE(real *e);
        CUDA_CALLABLE_MEMBER void setRho(real *rho);
    };

    namespace SharedNS {
        __global__ void set(Shared *shared, real *forces, real *courant, real *artVisc);
        __global__ void setE(Shared *shared, real *e);
        __global__ void setRho(Shared *shared, real *rho);

        namespace Launch {
            void set(Shared *shared, real *forces, real *courant, real *artVisc);
            void setE(Shared *shared, real *e);
            void setRho(Shared *shared, real *rho);
        }
    }

    struct BlockShared {
        real *forces;
        real *courant;
        real *artVisc;
        real *e;
        real *rho;

        CUDA_CALLABLE_MEMBER BlockShared();
        CUDA_CALLABLE_MEMBER BlockShared(real *forces, real *courant, real *artVisc);
        CUDA_CALLABLE_MEMBER ~BlockShared();

        CUDA_CALLABLE_MEMBER void set(real *forces, real *courant, real *artVisc);
        CUDA_CALLABLE_MEMBER void setE(real *e);
        CUDA_CALLABLE_MEMBER void setRho(real *rho);
    };

    namespace BlockSharedNS {
        __global__ void set(BlockShared *blockShared, real *forces, real *courant, real *artVisc);
        __global__ void setE(BlockShared *blockShared, real *e);
        __global__ void setRho(BlockShared *blockShared, real *e);

        namespace Launch {
            void set(BlockShared *blockShared, real *forces, real *courant, real *artVisc);
            void setE(BlockShared *blockShared, real *e);
            void setRho(BlockShared *blockShared, real *e);
        }
    }

    namespace Kernel {

        __global__ void corrector(Particles *particles, IntegratedParticles *predictor, real dt, int numParticles);
        __global__ void predictor(Particles *particles, IntegratedParticles *predictor, real dt, int numParticles);

        __global__ void setTimeStep(SimulationTime *simulationTime, Material *materials, Particles *particles, BlockShared *blockShared,
                                    int *blockCount, int numParticles);


        __global__ void pressureChangeCheck();

        namespace Launch {

            real corrector(Particles *particles, IntegratedParticles *predictor, real dt, int numParticles);
            real predictor(Particles *particles, IntegratedParticles *predictor, real dt, int numParticles);

            real setTimeStep();

            real pressureChangeCheck();

        }

    }
}

#endif //MILUPHPC_DEVICE_PREDICTOR_CORRECTOR_EULER_CUH
