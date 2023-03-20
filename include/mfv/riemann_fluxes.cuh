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
#include "../materials/material.cuh"
#include "riemann_solver.cuh"
#include "../cuda_utils/linalg.cuh"

//#define MFM_MASS_FLUX_TOL 1e-5

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
         * @brief Compute quadrature point (first order)
         *
         * @param[out] x_ij quadrature point vector between x_i and x_j
         * @param[in] i index of particle i
         * @param[in] ip index of interaction partner j
         * @param[in] particles particles class instance
         */
        __device__ void quadraturePoint(real x_ij[DIM], int i, int ip, Particles *particles);

        /**
         * @brief Compute frame velocity (firstOrder)
         *
         * @param[out] vFrame velocity vector of the frame of reference between particle i and ip
         * @param[in] i index of particle i
         * @param[in] ip index of interaction partner j
         * @param[in] particles particles class instance
         */
        __device__ void frameVelocity(real vFrame[DIM], int i, int ip, Particles *particles);

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
         * @param[in] ji index of storage of interaction partner j (j+i*MAX_NUM_INTERACTION)
         * @param[in] interaction list of interaction partners (nearest neighbor list)
         * @param[in] particles particles class instance
         */
        __device__ void effectiveFace(real Aij[DIM], int i, int ji, int *interactions, Particles *particles);

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
        __device__ void gradient(real *grad, real *f, int i, int *interactions, int noi, Particles *particles,
                                 SlopeLimitingParameters *slopeLimitingParameters);

#if PAIRWISE_LIMITER
        /**
         * @brief pairwise limiter for interacting particles
         *
         * Using the pairwise limiter for reconstruction of values at the quadrature point proposed by
         * P.F. Hopkins, 2015
         *
         * @param phi0 reconstructed value at xij using slope limited gradients
         * @param phi_i value at particle location x_i
         * @param phi_j value at particle location x_j
         * @param xijxiAbs distance of quadrature point x_ij to particles location x_i
         * @param xjxiAbs distance of interaction partner x_j to x_i
         * @param[in] slopeLimitingParameters struct containing parameters \f$ \psi_1 \f$ and
         *            \f$ \psi_2 \f$ from config file used for the employed slope limiter
         * @return limited reconstructed value
         */
        __device__ double pairwiseLimiter(real &phi0, real &phi_i, real &phi_j, real &xijxiAbs, real &xjxiAbs,
                                          SlopeLimitingParameters *slopeLimitingParameters);
#endif // PAIRWISE_LIMITER
    }

    namespace Kernel {

        /**
         * @brief Compute second-order accurate gradients for primitive variables
         *
         * > Corresponding wrapper function: ::MFV::Kernel::Launch::computeGradients()
         *
         *
         *
         * @param particles Particles class instance
         * @param interactions interaction list/interaction partners
         * @param numParticles amount of particles
         * @param[in] slopeLimitingParameters struct containing parameters for slope limiter in the spirit of
         *                                    unstructured mesh codes (Barth & Jespersen, 1989) and pairwise limiter
         *                                    proposed by P.F. Hopkins, 2015
         *
         */
        __global__ void computeGradients(Particles *particles, int *interactions,
                                         int numParticles, SlopeLimitingParameters *slopeLimitingParameters);


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
         * @param[in] slopeLimitingParameters struct containing parameters for slope limiter in the spirit of
         *                                    unstructured mesh codes (Barth & Jespersen, 1989) and pairwise limiter
         *                                    proposed by P.F. Hopkins, 2015
         * @param dt current timestep
         * @param materials array holding material data
         */
        __global__ void riemannFluxes(Particles *particles, int *interactions,
                                      int numParticles, SlopeLimitingParameters *slopeLimitingParameters,
                                      real *dt, Material *materials);

        namespace Launch {
            /**
             * @brief Wrapper for ::MFV::Kernel::computeGradients().
             *
             * @param particles Particles class instance
             * @param interactions interaction list/interaction partners
             * @param numParticles amount of particles
             * @param slopeLimitingParameters struct holding parameters for the slope limiting
             *
             * @return execution wall time
             */
            real computeGradients(Particles *particles, int *interactions, int numParticles,
                                  SlopeLimitingParameters *slopeLimitingParameters);

            /**
             * @brief Wrapper for ::MFV::Kernel::riemannFluxes().
             *
             * @param particles Particles class instance
             * @param interactions interaction list/interaction partners
             * @param numParticles amount of particles
             * @param slopeLimitingParameters struct holding parameters for the slope limiting
             * @param dt timestep
             * @param materials array holding material data
             *
             * @return execution wall time
             */
            real riemannFluxes(Particles *particles, int *interactions, int numParticles,
                               SlopeLimitingParameters *slopeLimitingParameters, real *dt, Material *materials);
        }

    }

}

#endif //MILUPHPC_RIEMANN_FLUXES_CUH