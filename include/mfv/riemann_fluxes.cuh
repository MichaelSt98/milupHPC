/**
 * @file riemann_fluxes.cuh
 * @brief MFV/MFM flux computation for godunov type integrator
 *
 * @author Johannes S. Martin
 * @bug no known bugs
 */

#ifndef MILUPHPC_RIEMANN_FLUXES_CUH
#define MILUPHPC_RIEMANN_FLUXES_CUH

#include "../particles.cuh"
#include "riemann_solver.cuh"
#include "../cuda_utils/linalg.cuh"

namespace MFV {

    /// Struct holding configuration of the slope limiter(s) employed
    struct SlopeLimitingParameters {

        /// critical condition number \f$ N_\text{cond}^\text{crit}\f$
        real critCondNum;
        /// minimum of "trust" parameter for gradients \f$ \beta_\text{min} \f$
        real betaMin;
        /// maximum of "trust" parameter for gradients \f$ \beta_\text{max} \f$
        real betaMax;
#if PAIRWISE_LIMITER
        real psi1, psi2;
#endif
        /**
         * @brief Default constructor
         */
        CUDA_CALLABLE_MEMBER SlopeLimitingParameters();

        /**
         * @brief Constructor
         *
         * @param critCondNum
         * @param betaMin
         * @param betaMax
         */
        CUDA_CALLABLE_MEMBER SlopeLimitingParameters(real critCondNum, real betaMin, real betaMax
#if PAIRWISE_LIMITER
                                                     ,real psi1, real psi2
#endif
                                                     );

    };

    /// Namespace holding device functions
    namespace Compute {

        /**
         * @brief Compute
         *
         * @param[out] x_ij quadrature point vector between x_i and x_j
         * @param[in] i index of particle i
         * @param[in] ip index of interaction partner j
         * @param[in] particles particles class instance
         */
        __device__ void quadraturePoint(real x_ij[DIM], int i, int ip, Particles *particles);

        /**
         * @brief Compute effective face from vector weights.
         *
         * \f[
         *   \begin{equation}
                \vec{A}_{ij} = V_i \vec{\tilde{\psi}}_j(\vec{x}_i) - V_j \vec{\tilde{\psi}}_i(\vec{x}_j)
             \end{equation}
         * \f]
         *
         * @param[out] Aij effective face vector from particle i->j
         * @param[in] i index of particle i
         * @param[in] ip index of interaction partner j
         * @param[in] interaction list of interaction partners (nearest neighbor list)
         * @param[in] particles particles class instance
         */
        __device__ void effectiveFace(real Aij[DIM], int i, int ip, int *interactions, Particles *particles);

        /**
         * @brief Compute gradients from vector weights.
         *
         * Computing second-order accurate gradients and slope limiting
         *
         * \f[
         *   \begin{equation}
                \left( \nabla f \right)_i^\alpha = \sum_j \left( f_j - f_i \right) \tilde{\vec{\psi}}_j^alpha
             \end{equation}
         * \f]
         * @param[out] grad second order accurate gradient of f at particle i
         * @param[in] f array of particle scalar quantity for which the gradient shall be estimated
         * @param[in] i index of particle i
         * @param[in] interactions list of interaction partners (nearest neighbor list)
         * @param[in] noi number of interactions
         * @param[in] particles particles class instance
         * @param[in] slopeLimitingParameters struct containing parameters \f$ N_\text{cond}^\text{crit} \f$,
         *            \f$ \beta_\text{min} \f$ and \f$ \beta_\text{max} \f$ from config file used for the employed
         *            slope limiter
         */
        __device__ void gradient(real grad[DIM], real *f, int i, int *interactions, int noi, Particles *particles,
                                 SlopeLimitingParameters *slopeLimitingParameters);
    }

    namespace Kernel {

        /**
         * @brief Compute the inter-particle fluxes \f$ \vec{\F}_{ij} \f$.
         *
         * > Corresponding wrapper function: ::MFV::Kernel::Launch::riemannFluxes()
         *
         *
         *
         * @param particles Particles class instance
         * @param interactions interaction list/interaction partners
         * @param numParticles amount of particles
         * @param riemannSolver default RiemannSolver function pointer
         * @param[in] slopeLimitingParameters struct containing parameters for slope limiter in the spirit of
         *                                    unstructured mesh codes (Barth & Jespersen, 1989) and pairwise limiter
         *                                    proposed by P.F. Hopkins, 2015
         *
         */
        __global__ void riemannFluxes(Particles *particles, RiemannSolver riemannSolver, int *interactions,
                                      int numParticles, SlopeLimitingParameters *slopeLimitingParameters);

        namespace Launch {
            /**
             * @brief Wrapper for ::MFV::Kernel::riemannFluxes().
             *
             * @param kernel smoothing kernel
             * @param particles Particles class instance
             * @param interactions interaction list/interaction partners
             * @param numParticles amount of particles
             * @param slopeLimitingParameters struct holding parameters for the slope limiting
             */
            real riemannFluxes(Particles *particles, RiemannSolver riemannSolver, int *interactions, int numParticles,
                               SlopeLimitingParameters *slopeLimitingParameters);
        }

    }

}

#endif //MILUPHPC_RIEMANN_FLUXES_CUH