#include "../../include/mfv/riemann_solver.cuh"

__device__ MFV::RiemannSolver exact_p = MFV::RiemannSolvers::exact;
//__device__ MFV::RiemannSolver hllc_p = MFV::RiemannSolvers::hllc; //TODO: implement

namespace MFV {
    __device__ void RiemannSolvers::exact(real Wsol[DIM+2], real WR[DIM+2], real WL[DIM+2]){
     //TODO: implement
    }
}

