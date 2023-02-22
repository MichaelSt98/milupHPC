#include "../../include/mfv/riemann_solver_handler.cuh"

extern __device__ MFV::RiemannSolver exact_p;
//extern __device__ MFV::RiemannSolver hllc_p;

MFV::RiemannSolverHandler::RiemannSolverHandler() {

}

MFV::RiemannSolverHandler::RiemannSolverHandler(Riemann::Solver riemannSolver) {

    switch (riemannSolver) {
        case Riemann::exact: {
            cudaMemcpyFromSymbol(&solver, exact_p, sizeof(RiemannSolver));
        } break;
        //case Riemann::hllc: {
        //    cudaMemcpyFromSymbol(&solver, hllc_p, sizeof(RiemannSolver));
        //} break;
        default:
            printf("Chosen default Riemann solver is not available!\n");
    }


}

MFV::RiemannSolverHandler::~RiemannSolverHandler() {

}