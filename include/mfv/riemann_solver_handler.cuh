/**
 * @file riemann_solver_handler.cuhh
 * @brief Handling the Riemann solvers.
 *
 * @author Johannes S. Martin
 * @bug no known bugs
 */

#ifndef MILUPHPC_RIEMANN_SOLVER_HANDLER_CUH
#define MILUPHPC_RIEMANN_SOLVER_HANDLER_CUH

#include "riemann_solver.cuh"
#include "../parameter.h"

namespace MFV {

    /**
     * @brief Riemann solver handler for meshless finite methods
     */
    class RiemannSolverHandler {

    public:

        /// Riemann solver function
        RiemannSolver solver;

        /**
         * @brief Default constructor.
         */
        RiemannSolverHandler();

        /**
         * @brief Constructor choosing the default Riemann solver.
         *
         * The default Riemann solver is used as long as a reasonable solution is found.
         * Fallbacks to the exact solver may be possible
         *
         * @param riemannSolver Default Riemann solver
         */
        RiemannSolverHandler(Riemann::Solver riemannSolver);

        /**
         * @brief Destructor.
         */
        ~RiemannSolverHandler();


    };

}

#endif // MILUPHPC_RIEMANN_SOLVER_HANDLER_CUH