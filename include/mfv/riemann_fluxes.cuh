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

    /// Namespace holding device functions
    namespace Compute {

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
         */
        __device__ void gradient(real grad[DIM], real *f, int i, int *interactions, int noi, Particles *particles);
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
         */
        __global__ void riemannFluxes(Particles *particles, RiemannSolver riemannSolver, int *interactions, int numParticles);

        namespace Launch {
            /**
             * @brief Wrapper for ::MFV::Kernel::riemannFluxes().
             *
             * @param kernel smoothing kernel
             * @param particles Particles class instance
             * @param interactions interaction list/interaction partners
             * @param numParticles amount of particles
             */
            real riemannFluxes(Particles *particles, RiemannSolver riemannSolver, int *interactions, int numParticles);
        }

    }

}

#endif //MILUPHPC_RIEMANN_FLUXES_CUH