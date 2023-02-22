/**
 * @file riemann_solver.cuh
 * @brief One-dimensional unsplit Riemann solvers.
 *
 * @author Johannes S. Martin
 * @bug no known bugs
 */

#ifndef MILUPHPC_RIEMANN_SOLVER_CUH
#define MILUPHPC_RIEMANN_SOLVER_CUH

#include "../parameter.h"

namespace MFV {

    /**
     * @brief Function pointer to generic Riemann solver function
     */
     // TODO: this signature works for an exact solver, solution vector may have to be changed when using HLLC
     typedef void (*RiemannSolver)(real Wsol[DIM+2], real WR[DIM+2], real WL[DIM+2]);

     namespace RiemannSolvers {

         /**
          * @brief Exact Riemann solver (Toro)
          *
          * @param[out] Wsol solution vector of primitive variables
          * @param[in] WR "right" or "i"-side state vector of primitive variables
          * @param[in] WL "left" or "j"-side state vector of primitive variables
          *
          */
          __device__ void exact(real Wsol[DIM+2], real WR[DIM+2], real WL[DIM+2]);

     }
}

#endif // MILUPHPC_RIEMANN_SOLVER_CUH