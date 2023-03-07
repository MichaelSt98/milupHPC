/**
 * @file RiemannSolver.cuh
 *
 * @brief Abstract Riemann solver class and children.
 *
 * Child class ExactRiemannSolver:
 * This Riemann solver is based on the ExactRiemannSolver class in the public
 * simulation code Shadowfax (Vandenbroucke & De Rijcke, 2016), and is almost
 * identical to the Riemann solver in the public simulation code CMacIonize
 * (https://github.com/bwvdnbro/CMacIonize).
 *
 * The original Riemann solver written in pure C++ has been ported to CUDA by Johannes S. Martin
 *
 * To use the Riemann solver, first create a RiemannSolver object. Afterwards call RiemannSolver::init(real gamma)
 * to set the desired adiabatic index
 * Actual Riemann problem solutions are then obtained
 * by calling ExactRiemannSolver::solve()
 *
 * @author Bert Vandenbroucke and Johannes S. Martin
 */

#ifndef MILUPHPC_RIEMANN_SOLVER_CUH
#define MILUPHPC_RIEMANN_SOLVER_CUH

#include "../cuda_utils/cuda_utilities.cuh"
#include "../parameter.h"

namespace MFV {

    class RiemannSolver {
    public:
        /**
          * @brief default constructor
          */
        CUDA_CALLABLE_MEMBER RiemannSolver();

        /**
         * @brief default destructor
         */
        CUDA_CALLABLE_MEMBER ~RiemannSolver();

        CUDA_CALLABLE_MEMBER virtual void init(real gamma) = 0;
        CUDA_CALLABLE_MEMBER virtual int solve(real rhoL, real uL, real PL, real rhoR, real uR,
                                               real PR, real &rhosol, real &usol, real &Psol,
                                               real dxdt) = 0;
    };

    /** @brief Exact Riemann solver
     *
     * Exact Riemann solver.
     * "R" and "L" corresponds to the "right" and "left" state of the problem.
     * Taken from https://github.com/bwvdnbro/python_finite_volume_solver/blob/master/RiemannSolver.hpp
     * written by Bert Vandenbroucke
     *
     */
    class ExactRiemannSolver : public RiemannSolver {
    public:

        /**
        * @brief Initializer to compute helper variables from adiabatic index
        *
        * @param gamma Adiabatic index @f$\gamma{}@f$.
        */
        CUDA_CALLABLE_MEMBER void init(real gamma);

        /**
         * @brief Solve the Riemann problem with the given left and right state.
         *
         * @param rhoL Left state density.
         * @param uL Left state velocity.
         * @param PL Left state pressure.
         * @param rhoR Right state density.
         * @param uR Right state velocity.
         * @param PR Right state pressure.
         * @param rhosol Density solution.
         * @param usol Velocity solution.
         * @param Psol Pressure solution.
         * @param dxdt Point in velocity space where we want to sample the solution.
         * @return Flag signaling whether the left state (-1), the right state (1), or
         *         a vacuum state (0) was sampled.
         */
        CUDA_CALLABLE_MEMBER int solve(real rhoL, real uL, real PL, real rhoR, real uR,
                                       real PR, real &rhosol, real &usol, real &Psol,
                                       real dxdt);

    private:
        /*! @brief Adiabatic index @f$\gamma{}@f$. */
        real _gamma;

        /*! @brief @f$\frac{\gamma+1}{2\gamma}@f$ */
        real _gp1d2g;

        /*! @brief @f$\frac{\gamma-1}{2\gamma}@f$ */
        real _gm1d2g;

        /*! @brief @f$\frac{\gamma-1}{\gamma+1}@f$ */
        real _gm1dgp1;

        /*! @brief @f$\frac{2}{\gamma+1}@f$ */
        real _tdgp1;

        /*! @brief @f$\frac{2}{\gamma-1}@f$ */
        real _tdgm1;

        /*! @brief @f$\frac{\gamma-1}{2}@f$ */
        real _gm1d2;

        /*! @brief @f$\frac{2\gamma}{\gamma-1}@f$ */
        real _tgdgm1;

        /*! @brief @f$\frac{1}{\gamma}@f$ */
        real _ginv;

        /**
         * @brief Get the soundspeed corresponding to the given density and pressure.
         *
         * @param rho Density value.
         * @param P Pressure value.
         * @return Soundspeed.
         */
        CUDA_CALLABLE_MEMBER real get_soundspeed(real rho, real P);

        /**
         * @brief Riemann fL or fR function.
         *
         * @param rho Density of the left or right state.
         * @param P Pressure of the left or right state.
         * @param a Soundspeed of the left or right state.
         * @param Pstar (Temporary) pressure of the middle state.
         * @return Value of the fL or fR function.
         */
        CUDA_CALLABLE_MEMBER real fb(real rho, real P, real a, real Pstar);

        /**
         * @brief Riemann f function.
         *
         * @param rhoL Density of the left state.
         * @param uL Velocity of the left state.
         * @param PL Pressure of the left state.
         * @param aL Soundspeed of the left state.
         * @param rhoR Density of the right state.
         * @param uR Velocity of the right state.
         * @param PR Pressure of the right state.
         * @param aR Soundspeed of the right state.
         * @param Pstar (Temporary) pressure of the middle state.
         * @return Value of the Riemann f function.
         */
        CUDA_CALLABLE_MEMBER real f(real rhoL, real uL, real PL, real aL, real rhoR,
                                    real uR, real PR, real aR, real Pstar);

        /**
         * @brief Derivative of the Riemann fL or fR function.
         *
         * @param rho Density of the left or right state.
         * @param P Pressure of the left or right state.
         * @param a Soundspeed of the left or right state.
         * @param Pstar (Temporary) pressure of the middle state.
         * @return Value of the derivative of the Riemann fL or fR function.
         */
        CUDA_CALLABLE_MEMBER real fprimeb(real rho, real P, real a, real Pstar);

        /**
         * @brief Derivative of the Riemann f function.
         *
         * @param rhoL Density of the left state.
         * @param PL Pressure of the left state.
         * @param aL Soundspeed of the left state.
         * @param rhoR Density of the right state.
         * @param PR Pressure of the right state.
         * @param aR Soundspeed of the right state.
         * @param Pstar (Temporary) pressure of the middle state.
         * @return Value of the derivative of the Riemann f function.
         */
        CUDA_CALLABLE_MEMBER real fprime(real rhoL, real PL, real aL, real rhoR,
                                         real PR, real aR, real Pstar);

        /**
         * @brief Riemann gL or gR function.
         *
         * @param rho Density of the left or right state.
         * @param P Pressure of the left or right state.
         * @param Pstar (Temporary) pressure in the middle state.
         * @return Value of the gL or gR function.
         */
        CUDA_CALLABLE_MEMBER real gb(real rho, real P, real Pstar);

        /**
         * @brief Get an initial guess for the pressure in the middle state.
         *
         * @param rhoL Left state density.
         * @param uL Left state velocity.
         * @param PL Left state pressure.
         * @param aL Left state soundspeed.
         * @param rhoR Right state density.
         * @param uR Right state velocity.
         * @param PR Right state pressure.
         * @param aR Right state soundspeed.
         * @return Initial guess for the pressure in the middle state.
         */
        CUDA_CALLABLE_MEMBER real guess_P(real rhoL, real uL, real PL, real aL,
                                          real rhoR, real uR, real PR, real aR);

        /**
         * @brief Find the pressure of the middle state by using Brent's method.
         *
         * @param rhoL Density of the left state.
         * @param uL Velocity of the left state.
         * @param PL Pressure of the left state.
         * @param aL Soundspeed of the left state.
         * @param rhoR Density of the right state.
         * @param uR Velocity of the right state.
         * @param PR Pressure of the right state.
         * @param aR Soundspeed of the right state.
         * @param Plow Lower bound guess for the pressure of the middle state.
         * @param Phigh Higher bound guess for the pressure of the middle state.
         * @param fPlow Value of the pressure function for the lower bound guess.
         * @param fPhigh Value of the pressure function for the upper bound guess.
         * @return Pressure of the middle state, with a 1.e-8 relative error
         * precision.
         */
        CUDA_CALLABLE_MEMBER real solve_brent(real rhoL, real uL, real PL, real aL,
                                              real rhoR, real uR, real PR, real aR,
                                              real Plow, real Phigh, real fPlow,
                                              real fPhigh);

        /**
         * @brief Sample the Riemann problem solution for a position in the right
         * shock wave regime.
         *
         * @param rhoR Density of the right state.
         * @param uR Velocity of the right state.
         * @param PR Pressure of the right state.
         * @param aR Soundspeed of the right state.
         * @param ustar Velocity of the middle state.
         * @param Pstar Pressure of the middle state.
         * @param rhosol Density solution.
         * @param usol Velocity solution.
         * @param Psol Pressure solution.
         * @param dxdt Point in velocity space where we want to sample the solution.
         */
        CUDA_CALLABLE_MEMBER void sample_right_shock_wave(real rhoR, real uR, real PR,
                                                          real aR, real ustar, real Pstar,
                                                          real &rhosol, real &usol,
                                                          real &Psol, real dxdt);

        /**
         * @brief Sample the Riemann problem solution for a position in the right
         * rarefaction wave regime.
         *
         * @param rhoR Density of the right state.
         * @param uR Velocity of the right state.
         * @param PR Pressure of the right state.
         * @param aR Soundspeed of the right state.
         * @param ustar Velocity of the middle state.
         * @param Pstar Pressure of the middle state.
         * @param rhosol Density solution.
         * @param usol Velocity solution.
         * @param Psol Pressure solution.
         * @param dxdt Point in velocity space where we want to sample the solution.
         */
        CUDA_CALLABLE_MEMBER void sample_right_rarefaction_wave(real rhoR, real uR, real PR,
                                                                real aR, real ustar,
                                                                real Pstar, real &rhosol,
                                                                real &usol, real &Psol,
                                                                real dxdt);

        /**
         * @brief Sample the Riemann problem solution in the right state regime.
         *
         * @param rhoR Density of the right state.
         * @param uR Velocity of the right state.
         * @param PR Pressure of the right state.
         * @param aR Soundspeed of the right state.
         * @param ustar Velocity of the middle state.
         * @param Pstar Pressure of the middle state.
         * @param rhosol Density solution.
         * @param usol Velocity solution.
         * @param Psol Pressure solution.
         * @param dxdt Point in velocity space where we want to sample the solution.
         */
        CUDA_CALLABLE_MEMBER void sample_right_state(real rhoR, real uR, real PR, real aR,
                                       real ustar, real Pstar, real &rhosol,
                                       real &usol, real &Psol,
                                       real dxdt);

        /**
         * @brief Sample the Riemann problem solution for a position in the left shock
         *  wave regime.
         *
         * @param rhoL Density of the left state.
         * @param uL Velocity of the left state.
         * @param PL Pressure of the left state.
         * @param aL Soundspeed of the left state.
         * @param ustar Velocity of the middle state.
         * @param Pstar Pressure of the middle state.
         * @param rhosol Density solution.
         * @param usol Velocity solution.
         * @param Psol Pressure solution.
         * @param dxdt Point in velocity space where we want to sample the solution.
         */
        CUDA_CALLABLE_MEMBER void sample_left_shock_wave(real rhoL, real uL, real PL,
                                                         real aL, real ustar, real Pstar,
                                                         real &rhosol, real &usol, real &Psol,
                                                         real dxdt);

        /**
         * @brief Sample the Riemann problem solution for a position in the left
         * rarefaction wave regime.
         *
         * @param rhoL Density of the left state.
         * @param uL Velocity of the left state.
         * @param PL Pressure of the left state.
         * @param aL Soundspeed of the left state.
         * @param ustar Velocity of the middle state.
         * @param Pstar Pressure of the middle state.
         * @param rhosol Density solution.
         * @param usol Velocity solution.
         * @param Psol Pressure solution.
         * @param dxdt Point in velocity space where we want to sample the solution.
         */
        CUDA_CALLABLE_MEMBER void sample_left_rarefaction_wave(real rhoL, real uL, real PL,
                                                               real aL, real ustar,
                                                               real Pstar, real &rhosol,
                                                               real &usol, real &Psol,
                                                               real dxdt);

        /**
         * @brief Sample the Riemann problem solution in the left state regime.
         *
         * @param rhoL Density of the left state.
         * @param uL Velocity of the left state.
         * @param PL Pressure of the left state.
         * @param aL Soundspeed of the left state.
         * @param ustar Velocity of the middle state.
         * @param Pstar Pressure of the middle state.
         * @param rhosol Density solution.
         * @param usol Velocity solution.
         * @param Psol Pressure solution.
         * @param dxdt Point in velocity space where we want to sample the solution.
         */
        CUDA_CALLABLE_MEMBER void sample_left_state(real rhoL, real uL, real PL, real aL,
                                                    real ustar, real Pstar, real &rhosol,
                                                    real &usol, real &Psol,
                                                    real dxdt);

        /**
         * @brief Sample the vacuum Riemann problem if the right state is a vacuum.
         *
         * @param rhoL Density of the left state.
         * @param uL Velocity of the left state.
         * @param PL Pressure of the left state.
         * @param aL Soundspeed of the left state.
         * @param rhosol Density solution.
         * @param usol Velocity solution.
         * @param Psol Pressure solution.
         * @param dxdt Point in velocity space where we want to sample the solution.
         * @return Flag indicating wether the left state (-1), the right state (1), or
         * a vacuum state (0) was sampled.
         */
        CUDA_CALLABLE_MEMBER int sample_right_vacuum(real rhoL, real uL, real PL, real aL,
                                       real &rhosol, real &usol, real &Psol,
                                       real dxdt);

        /**
         * @brief Sample the vacuum Riemann problem if the left state is a vacuum.
         *
         * @param rhoR Density of the right state.
         * @param uR Velocity of the right state.
         * @param PR Pressure of the right state.
         * @param aR Soundspeed of the right state.
         * @param rhosol Density solution.
         * @param usol Velocity solution.
         * @param Psol Pressure solution.
         * @param dxdt Point in velocity space where we want to sample the solution.
         * @return Flag indicating wether the left state (-1), the right state (1), or
         * a vacuum state (0) was sampled.
         */
        CUDA_CALLABLE_MEMBER int sample_left_vacuum(real rhoR, real uR, real PR, real aR,
                                                    real &rhosol, real &usol, real &Psol,
                                                    real dxdt);

        /**
         * @brief Sample the vacuum Riemann problem in the case vacuum is generated in
         * between the left and right state.
         *
         * @param rhoL Density of the left state.
         * @param uL Velocity of the left state.
         * @param PL Pressure of the left state.
         * @param aL Soundspeed of the left state.
         * @param rhoR Density of the right state.
         * @param uR Velocity of the right state.
         * @param PR Pressure of the right state.
         * @param aR Soundspeed of the right state.
         * @param rhosol Density solution.
         * @param usol Velocity solution.
         * @param Psol Pressure solution.
         * @param dxdt Point in velocity space where we want to sample the solution.
         * @return Flag indicating wether the left state (-1), the right state (1), or
         * a vacuum state (0) was sampled.
         */
        CUDA_CALLABLE_MEMBER int sample_vacuum_generation(real rhoL, real uL, real PL,
                                                          real aL, real rhoR, real uR,
                                                          real PR, real aR, real &rhosol,
                                                          real &usol, real &Psol,
                                                          real dxdt);

        /**
         * @brief Vacuum Riemann solver.
         *
         * This solver is called when one or both states have a zero density, or when
         * the vacuum generation condition is satisfied (meaning vacuum is generated
         * in the middle state, although strictly speaking there is no "middle"
         * state if vacuum is involved).
         *
         * @param rhoL Density of the left state.
         * @param uL Velocity of the left state.
         * @param PL Pressure of the left state.
         * @param aL Soundspeed of the left state.
         * @param rhoR Density of the right state.
         * @param uR Velocity of the right state.
         * @param PR Pressure of the right state.
         * @param aR Soundspeed of the right state.
         * @param rhosol Density solution.
         * @param usol Velocity solution.
         * @param Psol Pressure solution.
         * @param dxdt Point in velocity space where we want to sample the solution.
         * @return Flag indicating wether the left state (-1), the right state (1), or
         * a vacuum state (0) was sampled.
         */
        CUDA_CALLABLE_MEMBER int solve_vacuum(real rhoL, real uL, real PL, real aL,
                                real rhoR, real uR, real PR, real aR,
                                real &rhosol, real &usol, real &Psol,
                                real dxdt);
    };
}

#endif // MILUPHPC_RIEMANN_SOLVER_CUH