/**
 * @file internal_forces.cuh
 * @brief SPH internal forces.
 *
 * More detailed description.
 * This file contains ...
 *
 * @author Michael Staneker
 * @bug no known bugs
 */
#ifndef MILUPHPC_INTERNAL_FORCES_CUH
#define MILUPHPC_INTERNAL_FORCES_CUH

#include "../subdomain_key_tree/subdomain.cuh"
#include "../materials/material.cuh"
#include "kernel.cuh"

namespace SPH {
    namespace Kernel {
        /**
         * @brief Internal SPH forces.
         *
         * > Corresponding wrapper function: ::SPH::Kernel::Launch::internalForces()
         *
         * The **artificial viscosity** terms can be taken into account as an additional (artificial) pressure term in the equations for conservation of momentum and energy.
         * The additional term for each interaction pair \f$(a, b)\f$ is
         *
         * \f{align}{
                \Pi^*_{ab} &= \begin{cases}
                \Pi_{ab} & \text{for  } (\boldsymbol{v_a} - \boldsymbol{v_b}) \cdot (\boldsymbol{x_a} - \boldsymbol{x_b}) < 0 \\
                0 & \text{otherwise} \\
                \end{cases} \\
                \text{whereas} \; \,\Pi_{ab} &= \frac{-\alpha_{av} \bar{c}_{s,ab} \nu_{ab} + \beta_{av} \nu_{ab}^2}{\bar{\rho}_{ab}} \\
                \text{with} \; \, \nu_{ab} &= \frac{\bar{h}_{ab}(\boldsymbol{v_a} - \boldsymbol{v_b}) \cdot (\boldsymbol{x_a} - \boldsymbol{x_b})}{(\boldsymbol{x_a} - \boldsymbol{x_b})^2+ \epsilon_{v} \bar{h}_{ab}^2} \; ,\\
                \bar{\rho}_{ab} &= \frac{\rho_a + \rho_b}{2} \; \, \text{and} \\
                \bar{c}_{s,ab} &= \frac{c_{s,a} + c_{s,b}}{2} \; .
         * \f}
         *
         * Here, \f$\alpha_{av}\f$ and \f$\beta_{av}\f$ determine the strength of the viscosity, \f$\nu_{ab}\f$ is an approximation for the divergence, \f$\bar{\rho}_{ab}\f$ is
         * the averaged quantity for density, \f$\bar{c}_{s,ab}\f$ for the speed of sound and \f$\bar{h}_{ab}\f$ for the smoothing length, eventually multiplied by
         * \f$\epsilon_v\f$ for hardly separated particles.
         *
         * @param kernel SPH kernel function
         * @param materials Material parameters
         * @param tree Tree class instance
         * @param particles Particle class instance
         * @param interactions interaction list/interaction partners
         * @param numRealParticles amount of particles
         */
        __global__ void internalForces(::SPH::SPH_kernel kernel, Material *materials, Tree *tree, Particles *particles,
                                       int *interactions, int numRealParticles);

        namespace Launch {
            /**
             * @brief Wrapper for ::SPH::Kernel::internalForces().
             *
             * @param kernel SPH kernel function
             * @param materials Material parameters
             * @param tree Tree class instance
             * @param particles Particle class instance
             * @param interactions interaction list/interaction partners
             * @return Wall time of kernel execution.
             */
            real internalForces(::SPH::SPH_kernel kernel, Material *materials, Tree *tree, Particles *particles,
                                int *interactions, int numRealParticles);
        }
    }
}

#endif //MILUPHPC_INTERNAL_FORCES_CUH
