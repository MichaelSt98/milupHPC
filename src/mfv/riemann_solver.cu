#include "../../include/mfv/riemann_solver.cuh"

namespace MFV {

    CUDA_CALLABLE_MEMBER RiemannSolver::RiemannSolver() {}

    CUDA_CALLABLE_MEMBER RiemannSolver::~RiemannSolver() {}

    /************************* ExactRiemannSolver private functions *******************************************/
    CUDA_CALLABLE_MEMBER real ExactRiemannSolver::get_soundspeed(real rho, real P){
        return sqrt(_gamma*P/rho);
    }

    CUDA_CALLABLE_MEMBER real ExactRiemannSolver::fb(real rho, real P, real a, real Pstar){
        real fval = 0.;
        if (Pstar > P) {
            real A = _tdgp1 / rho;
            real B = _gm1dgp1 * P;
            fval = (Pstar - P) * sqrt(A / (Pstar + B));
        } else {
            fval = _tdgm1 * a * (pow(Pstar / P, _gm1d2g) - 1.);
        }
        return fval;
    }

    CUDA_CALLABLE_MEMBER real ExactRiemannSolver::f(real rhoL, real uL, real PL, real aL, real rhoR,
                                                    real uR, real PR, real aR, real Pstar){
        return fb(rhoL, PL, aL, Pstar) + fb(rhoR, PR, aR, Pstar) + (uR - uL);
    }

    CUDA_CALLABLE_MEMBER real ExactRiemannSolver::fprimeb(real rho, real P, real a, real Pstar){
        real fval = 0.;
        if (Pstar > P) {
            real A = _tdgp1 / rho;
            real B = _gm1dgp1 * P;
            fval = (1. - 0.5 * (Pstar - P) / (B + Pstar)) * sqrt(A / (Pstar + B));
        } else {
            fval = 1. / (rho * a) * pow(Pstar / P, -_gp1d2g);
        }
        return fval;
    }

    CUDA_CALLABLE_MEMBER real ExactRiemannSolver::fprime(real rhoL, real PL, real aL, real rhoR,
                                                         real PR, real aR, real Pstar){
        return fprimeb(rhoL, PL, aL, Pstar) + fprimeb(rhoR, PR, aR, Pstar);
    }

    CUDA_CALLABLE_MEMBER real ExactRiemannSolver::gb(real rho, real P, real Pstar){
        real A = _tdgp1 / rho;
        real B = _gm1dgp1 * P;
        return sqrt(A / (Pstar + B));
    }

    CUDA_CALLABLE_MEMBER real ExactRiemannSolver::guess_P(real rhoL, real uL, real PL, real aL,
                                                          real rhoR, real uR, real PR, real aR){
        real Pguess;
        real Pmin = min(PL, PR);
        real Pmax = max(PL, PR);
        real qmax = Pmax / Pmin;
        real Ppv = 0.5 * (PL + PR) - 0.125 * (uR - uL) * (PL + PR) * (aL + aR);
        Ppv = max(5.e-9 * (PL + PR), Ppv);
        if (qmax <= 2. && Pmin <= Ppv && Ppv <= Pmax) {
            Pguess = Ppv;
        } else {
            if (Ppv < Pmin) {
                // two rarefactions
                Pguess = pow(
                        (aL + aR - _gm1d2 * (uR - uL)) /
                        (aL / pow(PL, _gm1d2g) + aR / pow(PR, _gm1d2g)),
                        _tgdgm1);
            } else {
                // two shocks
                real gL = gb(rhoL, PL, Ppv);
                real gR = gb(rhoR, PR, Ppv);
                Pguess = (gL * PL + gR * PR - uR + uL) / (gL + gR);
            }
        }
        // Toro: "Not that approximate solutions may predict, incorrectly, a
        // negative value for pressure (...). Thus in order to avoid negative guess
        // values we introduce the small positive constant _tolerance"
        // (tolerance is 1.e-8 in this case)
        Pguess = max(5.e-9 * (PL + PR), Pguess);
        return Pguess;
    }

    CUDA_CALLABLE_MEMBER real ExactRiemannSolver::solve_brent(real rhoL, real uL, real PL, real aL,
                                                              real rhoR, real uR, real PR, real aR,
                                                              real Plow, real Phigh, real fPlow,
                                                              real fPhigh){
        real a = Plow;
        real b = Phigh;
        real c = 0.;
        real d = 1e230;

        real fa = fPlow;
        real fb = fPhigh;
        real fc = 0.;

        real s = 0.;
        real fs = 0.;

        if (fa * fb > 0.) {
            printf("ERROR: Equal sign function values provided to solve_brent (%g %g)!", fa, fb);
        }

        // if |f(a)| < |f(b)| then swap (a,b) end if
        if (abs(fa) < abs(fb)) {
            real tmp = a;
            a = b;
            b = tmp;
            tmp = fa;
            fa = fb;
            fb = tmp;
        }

        c = a;
        fc = fa;
        bool mflag = true;

        while (fb != 0. && (abs(a - b) > 5.e-9 * (a + b))) {
            if ((fa != fc) && (fb != fc)) {
                // Inverse quadratic interpolation
                s = a * fb * fc / (fa - fb) / (fa - fc) +
                    b * fa * fc / (fb - fa) / (fb - fc) +
                    c * fa * fb / (fc - fa) / (fc - fb);
            } else {
                // Secant Rule
                s = b - fb * (b - a) / (fb - fa);
            }

            real tmp2 = 0.25 * (3. * a + b);
            if (!(((s > tmp2) && (s < b)) || ((s < tmp2) && (s > b))) ||
                (mflag && (std::abs(s - b) >= 0.5 * std::abs(b - c))) ||
                (!mflag && (std::abs(s - b) >= 0.5 * std::abs(c - d))) ||
                (mflag && (std::abs(b - c) < 5.e-9 * (b + c))) ||
                (!mflag && (std::abs(c - d) < 5.e-9 * (c + d)))) {
                s = 0.5 * (a + b);
                mflag = true;
            } else {
                mflag = false;
            }
            fs = f(rhoL, uL, PL, aL, rhoR, uR, PR, aR, s);
            d = c;
            c = b;
            fc = fb;
            if (fa * fs < 0.) {
                b = s;
                fb = fs;
            } else {
                a = s;
                fa = fs;
            }

            // if |f(a)| < |f(b)| then swap (a,b) end if
            if (abs(fa) < abs(fb)) {
                real tmp = a;
                a = b;
                b = tmp;
                tmp = fa;
                fa = fb;
                fb = tmp;
            }
        }
        return b;
    }

    CUDA_CALLABLE_MEMBER void ExactRiemannSolver::sample_right_shock_wave(real rhoR, real uR, real PR,
                                                                          real aR, real ustar, real Pstar,
                                                                          real &rhosol, real &usol,
                                                                          real &Psol, real dxdt){
        // variable used twice below
        real PdPR = Pstar / PR;
        // get the shock speed
        real SR = uR + aR * sqrt(_gp1d2g * PdPR + _gm1d2g);
        if (SR > dxdt) {
            /// middle state (shock) regime
            rhosol = rhoR * (PdPR + _gm1dgp1) / (_gm1dgp1 * PdPR + 1.);
            usol = ustar;
            Psol = Pstar;
        } else {
            /// right state regime
            rhosol = rhoR;
            usol = uR;
            Psol = PR;
        }
    }

    CUDA_CALLABLE_MEMBER void ExactRiemannSolver::sample_right_rarefaction_wave(real rhoR, real uR, real PR,
                                                                                real aR, real ustar,
                                                                                real Pstar, real &rhosol,
                                                                                real &usol, real &Psol,
                                                                                real dxdt){
        // get the velocity of the head of the rarefaction wave
        real SHR = uR + aR;
        if (SHR > dxdt) {
            /// rarefaction wave regime
            // variable used twice below
            real PdPR = Pstar / PR;
            // get the velocity of the tail of the rarefaction wave
            real STR = ustar + aR * pow(PdPR, _gm1d2g);
            if (STR > dxdt) {
                /// middle state regime
                rhosol = rhoR * pow(PdPR, _ginv);
                usol = ustar;
                Psol = Pstar;
            } else {
                /// rarefaction fan regime
                // variable used twice below
                real base = _tdgp1 - _gm1dgp1 * (uR - dxdt) / aR;
                rhosol = rhoR * pow(base, _tdgm1);
                usol = _tdgp1 * (-aR + _gm1d2 * uR + dxdt);
                Psol = PR * pow(base, _tgdgm1);
            }
        } else {
            /// right state regime
            rhosol = rhoR;
            usol = uR;
            Psol = PR;
        }
    }

    CUDA_CALLABLE_MEMBER void ExactRiemannSolver::sample_right_state(real rhoR, real uR, real PR, real aR,
                                                                     real ustar, real Pstar, real &rhosol,
                                                                     real &usol, real &Psol,
                                                                     real dxdt){
        if (Pstar > PR) {
            /// shock wave
            sample_right_shock_wave(rhoR, uR, PR, aR, ustar, Pstar, rhosol, usol,
                                    Psol, dxdt);
        } else {
            /// rarefaction wave
            sample_right_rarefaction_wave(rhoR, uR, PR, aR, ustar, Pstar, rhosol,
                                          usol, Psol, dxdt);
        }
    }

    CUDA_CALLABLE_MEMBER void ExactRiemannSolver::sample_left_shock_wave(real rhoL, real uL, real PL,
                                                     real aL, real ustar, real Pstar,
                                                     real &rhosol, real &usol, real &Psol,
                                                     real dxdt){
        // variable used twice below
        real PdPL = Pstar / PL;
        // get the shock speed
        real SL = uL - aL * sqrt(_gp1d2g * PdPL + _gm1d2g);
        if (SL < dxdt) {
            /// middle state (shock) regime
            rhosol = rhoL * (PdPL + _gm1dgp1) / (_gm1dgp1 * PdPL + 1.);
            usol = ustar;
            Psol = Pstar;
        } else {
            /// left state regime
            rhosol = rhoL;
            usol = uL;
            Psol = PL;
        }
    }

    CUDA_CALLABLE_MEMBER void ExactRiemannSolver::sample_left_rarefaction_wave(real rhoL, real uL, real PL,
                                                                               real aL, real ustar,
                                                                               real Pstar, real &rhosol,
                                                                               real &usol, real &Psol,
                                                                               real dxdt){
        // get the velocity of the head of the rarefaction wave
        real SHL = uL - aL;
        if (SHL < dxdt) {
            /// rarefaction wave regime
            // variable used twice below
            real PdPL = Pstar / PL;
            // get the velocity of the tail of the rarefaction wave
            real STL = ustar - aL * pow(PdPL, _gm1d2g);
            if (STL > dxdt) {
                /// rarefaction fan regime
                // variable used twice below
                real base = _tdgp1 + _gm1dgp1 * (uL - dxdt) / aL;
                rhosol = rhoL * pow(base, _tdgm1);
                usol = _tdgp1 * (aL + _gm1d2 * uL + dxdt);
                Psol = PL * pow(base, _tgdgm1);
            } else {
                /// middle state regime
                rhosol = rhoL * pow(PdPL, _ginv);
                usol = ustar;
                Psol = Pstar;
            }
        } else {
            /// left state regime
            rhosol = rhoL;
            usol = uL;
            Psol = PL;
        }
    }

    CUDA_CALLABLE_MEMBER void ExactRiemannSolver::sample_left_state(real rhoL, real uL, real PL, real aL,
                                                                    real ustar, real Pstar, real &rhosol,
                                                                    real &usol, real &Psol,
                                                                    real dxdt){
        if (Pstar > PL) {
            /// shock wave
            sample_left_shock_wave(rhoL, uL, PL, aL, ustar, Pstar, rhosol, usol, Psol,
                                   dxdt);
        } else {
            /// rarefaction wave
            sample_left_rarefaction_wave(rhoL, uL, PL, aL, ustar, Pstar, rhosol, usol,
                                         Psol, dxdt);
        }
    }

    CUDA_CALLABLE_MEMBER int ExactRiemannSolver::sample_right_vacuum(real rhoL, real uL, real PL, real aL,
                                                                     real &rhosol, real &usol, real &Psol,
                                                                     real dxdt){
        if (uL - aL < dxdt) {
            /// vacuum regime
            // get the vacuum rarefaction wave speed
            real SL = uL + _tdgm1 * aL;
            if (SL > dxdt) {
                /// rarefaction wave regime
                // variable used twice below
                real base = _tdgp1 + _gm1dgp1 * (uL - dxdt) / aL;
                rhosol = rhoL * pow(base, _tdgm1);
                usol = _tdgp1 * (aL + _gm1d2 * uL + dxdt);
                Psol = PL * pow(base, _tgdgm1);
                return -1;
            } else {
                /// vacuum
                rhosol = 0.;
                usol = 0.;
                Psol = 0.;
                return 0;
            }
        } else {
            /// left state regime
            rhosol = rhoL;
            usol = uL;
            Psol = PL;
            return -1;
        }
    }

    CUDA_CALLABLE_MEMBER int ExactRiemannSolver::sample_left_vacuum(real rhoR, real uR, real PR, real aR,
                                                                    real &rhosol, real &usol, real &Psol,
                                                                    real dxdt){
        if (dxdt < uR + aR) {
            /// vacuum regime
            // get the vacuum rarefaction wave speed
            real SR = uR - _tdgm1 * aR;
            if (SR < dxdt) {
                /// rarefaction wave regime
                // variable used twice below
                real base = _tdgp1 - _gm1dgp1 * (uR - dxdt) / aR;
                rhosol = rhoR * pow(base, _tdgm1);
                usol = _tdgp1 * (-aR + _tdgm1 * uR + dxdt);
                Psol = PR * pow(base, _tgdgm1);
                return 1;
            } else {
                /// vacuum
                rhosol = 0.;
                usol = 0.;
                Psol = 0.;
                return 0;
            }
        } else {
            /// right state regime
            rhosol = rhoR;
            usol = uR;
            Psol = PR;
            return 1;
        }
    }

    CUDA_CALLABLE_MEMBER int ExactRiemannSolver::sample_vacuum_generation(real rhoL, real uL, real PL,
                                                                          real aL, real rhoR, real uR,
                                                                          real PR, real aR, real &rhosol,
                                                                          real &usol, real &Psol,
                                                                          real dxdt){
        // get the speeds of the left and right rarefaction waves
        real SR = uR - _tdgm1 * aR;
        real SL = uL + _tdgm1 * aL;
        if (SR > dxdt && SL < dxdt) {
            /// vacuum
            rhosol = 0.;
            usol = 0.;
            Psol = 0.;
            return 0;
        } else {
            if (SL < dxdt) {
                /// right state
                if (dxdt < uR + aR) {
                    /// right rarefaction wave regime
                    // variable used twice below
                    real base = _tdgp1 - _gm1dgp1 * (uR - dxdt) / aR;
                    rhosol = rhoR * pow(base, _tdgm1);
                    usol = _tdgp1 * (-aR + _tdgm1 * uR + dxdt);
                    Psol = PR * pow(base, _tgdgm1);
                } else {
                    /// right state regime
                    rhosol = rhoR;
                    usol = uR;
                    Psol = PR;
                }
                return 1;
            } else {
                /// left state
                if (dxdt > uL - aL) {
                    /// left rarefaction wave regime
                    // variable used twice below
                    real base = _tdgp1 + _gm1dgp1 * (uL - dxdt) / aL;
                    rhosol = rhoL * pow(base, _tdgm1);
                    usol = _tdgp1 * (aL + _tdgm1 * uL + dxdt);
                    Psol = PL * pow(base, _tgdgm1);
                } else {
                    /// left state regime
                    rhosol = rhoL;
                    usol = uL;
                    Psol = PL;
                }
                return -1;
            }
        }
    }

    CUDA_CALLABLE_MEMBER int ExactRiemannSolver::solve_vacuum(real rhoL, real uL, real PL, real aL,
                                                              real rhoR, real uR, real PR, real aR,
                                                              real &rhosol, real &usol, real &Psol,
                                                              real dxdt){
        // if both states are vacuum, the solution is also vacuum
        if (rhoL == 0. && rhoR == 0.) {
            rhosol = 0.;
            usol = 0.;
            Psol = 0.;
            return 0;
        }

        if (rhoR == 0.) {
            /// vacuum right state
            return sample_right_vacuum(rhoL, uL, PL, aL, rhosol, usol, Psol, dxdt);
        } else if (rhoL == 0.) {
            /// vacuum left state
            return sample_left_vacuum(rhoR, uR, PR, aR, rhosol, usol, Psol, dxdt);
        } else {
            /// vacuum "middle" state
            return sample_vacuum_generation(rhoL, uL, PL, aL, rhoR, uR, PR, aR,
                                            rhosol, usol, Psol, dxdt);
        }

    }

    /************************* ExactRiemannSolver public functions *******************************************/
    CUDA_CALLABLE_MEMBER void ExactRiemannSolver::init(real gamma){
        _gamma = gamma;
        // related quantities:
        _gp1d2g = 0.5 * (_gamma + 1.) / _gamma; // gamma plus 1 divided by 2 gamma
        _gm1d2g = 0.5 * (_gamma - 1.) / _gamma; // gamma minus 1 divided by 2 gamma
        _gm1dgp1 = (_gamma - 1.) / (_gamma + 1.); // gamma minus 1 divided by gamma plus 1
        _tdgp1 = 2. / (_gamma + 1.);       // two divided by gamma plus 1
        _tdgm1 = 2. / (_gamma - 1.);       // two divided by gamma minus 1
        _gm1d2 = 0.5 * (_gamma - 1.);      // gamma minus 1 divided by 2
        _tgdgm1 = 2. * _gamma / (_gamma - 1.); // two times gamma divided by gamma minus 1
        _ginv = 1. / _gamma;             // gamma inverse
    }

    CUDA_CALLABLE_MEMBER int ExactRiemannSolver::solve(real rhoL, real uL, real PL, real rhoR, real uR,
                                                        real PR, real &rhosol, real &usol, real &Psol,
                                                        real dxdt){

        // get the soundspeeds
        real aL = get_soundspeed(rhoL, PL);
        real aR = get_soundspeed(rhoR, PR);

        // handle vacuum
        if (rhoL == 0. || rhoR == 0.) {
            return solve_vacuum(rhoL, uL, PL, aL, rhoR, uR, PR, aR, rhosol, usol,
                                Psol, dxdt);
        }

        // handle vacuum generation
        if (2. * aL / (_gamma - 1.) + 2. * aR / (_gamma - 1.) <= uR - uL) {
            return solve_vacuum(rhoL, uL, PL, aL, rhoR, uR, PR, aR, rhosol, usol,
                                Psol, dxdt);
        }

        // find the pressure and velocity in the middle state
        // since this is an exact Riemann solver, this is an iterative process,
        // whereby we basically find the root of a function (the Riemann f function
        // defined above)
        // we start by using a Newton-Raphson method, since we do not have an
        // interval in which the function changes sign
        // however, as soon as we have such an interval, we switch to a much more
        // robust root finding method (Brent's method). We do this because the
        // Newton-Raphson method in some cases can overshoot and return a negative
        // pressure, for which the Riemann f function is not defined. Brent's method
        // will never stroll outside of the initial interval in which the function
        // changes sign.
        real Pstar = 0.;
        real Pguess = guess_P(rhoL, uL, PL, aL, rhoR, uR, PR, aR);
        // we only store this variable to store the sign of the function for
        // pressure zero
        // we need to find a larger pressure for which this sign changes to have an
        // interval where we can use Brent's method
        real fPstar = f(rhoL, uL, PL, aL, rhoR, uR, PR, aR, Pstar);
        real fPguess = f(rhoL, uL, PL, aL, rhoR, uR, PR, aR, Pguess);
        if (fPstar * fPguess >= 0.) {
            // Newton-Raphson until convergence or until usable interval is
            // found to use Brent's method
            while (std::abs(Pstar - Pguess) > 5.e-9 * (Pstar + Pguess) &&
                   fPguess < 0.) {
                Pstar = Pguess;
                fPstar = fPguess;
                Pguess -= fPguess / fprime(rhoL, PL, aL, rhoR, PR, aR, Pguess);
                fPguess = f(rhoL, uL, PL, aL, rhoR, uR, PR, aR, Pguess);
            }
        }

        // As soon as there is a suitable interval: use Brent's method
        if (std::abs(Pstar - Pguess) > 5.e-9 * (Pstar + Pguess) && fPguess > 0.) {
            Pstar = solve_brent(rhoL, uL, PL, aL, rhoR, uR, PR, aR, Pstar, Pguess,
                                fPstar, fPguess);
        } else {
            Pstar = Pguess;
        }

        // the middle state velocity is fixed once the middle state pressure is
        // known
        real ustar = 0.5 * (uL + uR) +
                       0.5 * (fb(rhoR, PR, aR, Pstar) - fb(rhoL, PL, aL, Pstar));

#if MESHLESS_FINITE_METHOD == 2
        // return middle state pressure and velocity as the rest is not needed
        // compare https://github.com/SWIFTSIM/SWIFT/blob/master/src/riemann/riemann_exact.h
        // l. 561ff.
        rhosol = 0.; // DUMMY
        usol = ustar;
        Psol = Pstar;
        return -1; // left state is sampled by default
#else

        // we now have solved the Riemann problem: we have the left, middle and
        // right state, and this completely fixes the solution
        // we just need to sample the solution for x/t = 0.
        if (ustar < dxdt) {
            // right state
            sample_right_state(rhoR, uR, PR, aR, ustar, Pstar, rhosol, usol, Psol,
                               dxdt);
            return 1;
        } else {
            // left state
            sample_left_state(rhoL, uL, PL, aL, ustar, Pstar, rhosol, usol, Psol,
                              dxdt);
            return -1;
        }
#endif // MESHLESS_FINITE_METHOD == 2
    }


}

