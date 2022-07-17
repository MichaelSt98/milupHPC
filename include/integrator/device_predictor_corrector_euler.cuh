/**
 * @file device_predictor_corrector_euler.cuh
 * @brief Device functions and kernels for the predictor corrector euler (Heun) integrator.
 *
 * @author Michael Staneker
 * @bug no known bugs
 */
#ifndef MILUPHPC_DEVICE_PREDICTOR_CORRECTOR_EULER_CUH
#define MILUPHPC_DEVICE_PREDICTOR_CORRECTOR_EULER_CUH

#include "../particles.cuh"
#if TARGET_GPU
#include "../materials/material.cuh"
#include "../simulation_time.cuh"

/// predictor corrector euler (Heun) integrator
namespace PredictorCorrectorEulerNS {

    struct Shared {
        real *forces;
        real *courant;
        real *artVisc;
        real *e;
        real *rho;
        real *vmax;

        CUDA_CALLABLE_MEMBER Shared();
        CUDA_CALLABLE_MEMBER Shared(real *forces, real *courant, real *artVisc);
        CUDA_CALLABLE_MEMBER ~Shared();

        CUDA_CALLABLE_MEMBER void set(real *forces, real *courant, real *artVisc);
        CUDA_CALLABLE_MEMBER void setE(real *e);
        CUDA_CALLABLE_MEMBER void setRho(real *rho);
        CUDA_CALLABLE_MEMBER void setVmax(real *vmax);
    };

    namespace SharedNS {
        __global__ void set(Shared *shared, real *forces, real *courant, real *artVisc);
        __global__ void setE(Shared *shared, real *e);
        __global__ void setRho(Shared *shared, real *rho);
        __global__ void setVmax(Shared *shared, real *vmax);

        namespace Launch {
            void set(Shared *shared, real *forces, real *courant, real *artVisc);
            void setE(Shared *shared, real *e);
            void setRho(Shared *shared, real *rho);
            void setVmax(Shared *shared, real *vmax);
        }
    }

    struct BlockShared {
        real *forces;
        real *courant;
        real *artVisc;
        real *e;
        real *rho;
        real *vmax;

        CUDA_CALLABLE_MEMBER BlockShared();
        CUDA_CALLABLE_MEMBER BlockShared(real *forces, real *courant, real *artVisc);
        CUDA_CALLABLE_MEMBER ~BlockShared();

        CUDA_CALLABLE_MEMBER void set(real *forces, real *courant, real *artVisc);
        CUDA_CALLABLE_MEMBER void setE(real *e);
        CUDA_CALLABLE_MEMBER void setRho(real *rho);
        CUDA_CALLABLE_MEMBER void setVmax(real *vmax);
    };

    namespace BlockSharedNS {
        __global__ void set(BlockShared *blockShared, real *forces, real *courant, real *artVisc);
        __global__ void setE(BlockShared *blockShared, real *e);
        __global__ void setRho(BlockShared *blockShared, real *e);
        __global__ void setVmax(BlockShared *blockShared, real *vmax);

        namespace Launch {
            void set(BlockShared *blockShared, real *forces, real *courant, real *artVisc);
            void setE(BlockShared *blockShared, real *e);
            void setRho(BlockShared *blockShared, real *e);
            void setVmax(BlockShared *blockShared, real *vmax);
        }
    }

    /// kernel functions
    namespace Kernel {

        /**
         * @brief Corrector step.
         *
         * > Corresponding wrapper function: ::PredictorCorrectorEulerNS::Kernel::Launch::corrector().
         *
         * @param particles
         * @param predictor
         * @param dt
         * @param numParticles
         */
        __global__ void corrector(Particles *particles, IntegratedParticles *predictor, real dt, int numParticles);

        /**
         * @brief Predictor step.
         *
         * > Corresponding wrapper function: ::PredictorCorrectorEulerNS::Kernel::Launch::predictor().
         *
         * @param particles
         * @param predictor
         * @param dt
         * @param numParticles
         */
        __global__ void predictor(Particles *particles, IntegratedParticles *predictor, real dt, int numParticles);

        /**
         * @brief Setting correct time step.
         *
         * > Corresponding wrapper function: ::PredictorCorrectorEulerNS::Kernel::Launch::setTimeStep().
         *
         * * Conditions to be applied:
         *
         * * sound waves traveling faster than a fraction of the smoothing length
         *     * \f$ \Delta t \leq C \frac{h}{c + 1.2 (\alpha_{\nu} c + \beta_{\nu} \mu_{max})} \f$
         *     * where \f$ c \f$ is the sound speed; \f$ \alpha_{\nu} \f$ and \f$ \beta_{\nu} \f$ are the viscosity
         *     parameters, \f$ \mu_{max} \f$ is the maximal value of \f$ \mu_{ij} \f$ and \f$ C \f$ is the Courant number
         * * time step constrains the distance a particle travels due to acceleration
         *     * \f$ \Delta t \leq \sqrt{\frac{h}{|\vec{a}|}} \f$
         * * all other quantities \f$ f \f$ have to be prevented from growing too fast within one time step
         *     * \f$ \Delta t \leq \begin{cases} a \frac{|f| + f_{min}}{|df|} & |df| > 0 \\ \Delta t_{max} & |df| = 0 \\ \end{cases} \f$ where \f$ a < 1 \f$
         * * additional constraint regarding parallelization approach:
         *     * particles should not move further than h/2
         *     * \f$ \Delta t \cdot v_{max} < \frac{h}{2} \, \Leftrightarrow \Delta t < \frac{h}{2 v_{max}} \f$
         *
         * @param simulationTime SimulationTime class instance
         * @param materials Material class instance
         * @param particles Particle class instance
         * @param blockShared
         * @param blockCount
         * @param numParticles Number of particles
         */
        __global__ void setTimeStep(SimulationTime *simulationTime, Material *materials, Particles *particles, BlockShared *blockShared,
                                    int *blockCount, int numParticles);

        /// wrapped kernel functions
        namespace Launch {

            /**
             * @brief Wrapper for ::PredictorCorrectorEulerNS::Kernel::corrector().
             *
             * @return Wall time of execution
             */
            real corrector(Particles *particles, IntegratedParticles *predictor, real dt, int numParticles);

            /**
             * @brief Wrapper for ::PredictorCorrectorEulerNS::Kernel::predictor().
             *
             * @return Wall time of execution
             */
            real predictor(Particles *particles, IntegratedParticles *predictor, real dt, int numParticles);

            /**
             * @brief Wrapper for ::PredictorCorrectorEulerNS::Kernel::setTimeStep().
             *
             * @return Wall time of execution
             */
            real setTimeStep(int multiProcessorCount, SimulationTime *simulationTime, Material *materials, Particles *particles,
                             BlockShared *blockShared, int *blockCount, real searchRadius, int numParticles);

            real pressureChangeCheck();

        }

    }
}

#endif // TARGET_GPU
#endif //MILUPHPC_DEVICE_PREDICTOR_CORRECTOR_EULER_CUH
