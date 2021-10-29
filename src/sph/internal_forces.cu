#include "../../include/sph/internal_forces.cuh"
#include "../include/cuda_utils/cuda_launcher.cuh"

//TODO: correct #ifs and correct ???
__global__ void SPH::Kernel::internalForces(::SPH::SPH_kernel kernel, Material *materials, Tree *tree, Particles *particles,
                                            int *interactions, int numRealParticles) {

    int i, k, inc, j, numInteractions;
    int f, kk;

    real W;
    real tmp;
    real x, vx, vxj, ax, dvx;
    ax = 0;
#if DIM > 1
    real y, vy, vyj, ay, dvy;
    ay = 0;
#if DIM == 3
    real z, vz, vzj, az, dvz;
    az = 0;
#endif // DIM > 1
#endif // DIM == 3

    real sml;

    int matId;
    int matIdj;

    real sml1;
    real Sj[DIM*DIM];

    real vr; // vr = v_ij * r_ij
    real rr;
    real rhobar; // rhobar = 0.5*(rho_i + rho_j)
    real mu;
    real muijmax;
    real smooth;
    real csbar;
    real alpha, beta;

#if ARTIFICIAL_STRESS
    real artf = 0;
#endif // ARTIFICIAL_STRESS

    int d;
    int dd;
    int e;

    real dr[DIM];
    real dv[DIM];

    real drhodt;

#if INTEGRATE_ENERGY
    real dedt;
#endif // INTEGRATE_ENERGY

#if NAVIER_STOKES
    double eta;
    double zetaij;
#endif // NAVIER_STOKES

    real vvnablaW;
    real dWdr;
    real dWdrj;
    real dWdx[DIM];
    real Wj;
    real dWdxj[DIM];
    real pij = 0;
    real r;
    real accels[DIM];
    real accelsj[DIM];
    real accelshearj[DIM];

    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numRealParticles; i += inc) {

        matId = particles->materialId[i];
        //if (EOS_TYPE_IGNORE == matEOS[p_rhs.materialId[i]] || matId == EOS_TYPE_IGNORE) {
        //    continue;
        //}

        numInteractions = particles->noi[i];

        ax = 0;
#if DIM > 1
        ay = 0;
#if DIM == 3
        az = 0;
#endif // DIM > 1
#endif // DIM == 3

        alpha = materials[matId].artificialViscosity.alpha; //alpha = matAlpha[matId];
        beta = materials[matId].artificialViscosity.beta; //beta = matBeta[matId];

        muijmax = 0;

        sml1 = particles->sml[i];

        drhodt = 0;

#if INTEGRATE_ENERGY
        dedt = 0;
#endif // INTEGRATE_ENERGY

#if INTEGRATE_SML
        particles->dsmldt[i] = 0.0;
#endif // INTEGRATE_SML

        for (d = 0; d < DIM; d++) {
            accels[d] = 0.0;
            accelsj[d] = 0.0;
            accelshearj[d] = 0.0;
        } // for (d = 0; d < DIM; d++)
        sml = particles->sml[i];

        x = particles->x[i];
        vx = particles->vx[i];
#if DIM > 1
        y = particles->y[i];
        vy = particles->vy[i];
#if DIM == 3
        z = particles->z[i];
        vz = particles->vz[i];
#endif // DIM > 1
#endif // DIM == 3

        // TODO: do I really want to zero the acceleration?
        particles->ax[i] = 0;
        // TODO: there is no particle entry dxdt within particles (but IntegratedParticles?!)
        //particles->dxdt[i] = 0;
#if DIM > 1
        particles->ay[i] = 0;
        //particles->dydt[i] = 0;
#if DIM == 3
        particles->az[i] = 0;
        //particles->dzdt[i] = 0;
#endif // DIM > 1
#endif // DIM == 3


        particles->drhodt[i] = 0.0;
#if INTEGRATE_ENERGY
        particles->dedt[i] = 0.0;
#endif // INTEGRATE_ENERGY
#if INTEGRATE_SML
        particles->dsmldt[i] = 0.0;
#endif // INTEGRATE_SML

        // if particle has no interactions continue and set all derivs to zero
        // but not the accels (these are handled in the tree for gravity)
        if (numInteractions == 0) {
            // finally continue
            continue;
        } // if (numInteractions == 0)

        // THE MAIN SPH LOOP FOR ALL INTERNAL FORCES
        // loop over interaction partners for SPH sums
        for (k = 0; k < numInteractions; k++) {

            //matIdj = EOS_TYPE_IGNORE;
            // the interaction partner
            j = interactions[i * MAX_NUM_INTERACTIONS + k];

            matIdj = particles->materialId[j];
            //if (EOS_TYPE_IGNORE == matEOS[p_rhs.materialId[j]] || matIdj == EOS_TYPE_IGNORE) {
            //    continue;
            //}

            for (d = 0; d < DIM; d++) {
                accelsj[d] = 0.0;
            } // for (d = 0; d < DIM; d++)


#if (VARIABLE_SML || INTEGRATE_SML)// || DEAL_WITH_TOO_MANY_INTERACTIONS)
            sml = 0.5*(particles->sml[i] + particles->sml[j]);
#endif // (VARIABLE_SML || INTEGRATE_SML)// || DEAL_WITH_TOO_MANY_INTERACTIONS)

            vxj = particles->vx[j];
#if DIM > 1
            vyj = particles->vy[j];
#if DIM == 3
            vzj = particles->vz[j];
#endif // DIM > 1
#endif // DIM == 3

            // relative vector
            dr[0] = x - particles->x[j];
#if DIM > 1
            dr[1] = y - particles->y[j];
#if DIM == 3
            dr[2] = z - particles->z[j];
#endif // DIM > 1
#endif // DIM == 3
            r = 0;
            for (e = 0; e < DIM; e++) {
                r += dr[e] * dr[e];
                dWdx[e] = 0.0;
            } // for (e = 0; e < DIM; e++)

            W = 0.0;
            dWdr = 0.0;

            r = sqrt(r);

            // get kernel values for this interaction
            kernel(&W, dWdx, &dWdr, dr, sml);

            dv[0] = dvx = vx - vxj;
#if DIM > 1
            dv[1] = dvy = vy - vyj;
#if DIM == 3
            dv[2] = dvz = vz - vzj;
#endif // DIM > 1
#endif // DIM == 3

            vvnablaW = dvx * dWdx[0];
#if DIM > 1
            vvnablaW += dvy * dWdx[1];
#if DIM == 3
            vvnablaW += dvz * dWdx[2];
#endif // DIM > 1
#endif // DIM == 3

            rr = 0.0;
            vr = 0.0;
            for (e = 0; e < DIM; e++) {
                rr += dr[e] * dr[e];
                vr += dv[e] * dr[e];
            } // for (e = 0; e < DIM; e++)

            // artificial viscosity force only if v_ij * r_ij < 0
            if (vr < 0) {
                csbar = 0.5 * (particles->cs[i] + particles->cs[j]);
                smooth = 0.5 * (sml1 + particles->sml[j]);

                const double eps_artvisc = 1e-2;
                mu = smooth * vr/(rr + smooth * smooth * eps_artvisc);

                if (mu > muijmax) {
                    muijmax = mu;
                } // if (mu > muijmax)
                rhobar = 0.5 * (particles->rho[i] + particles->rho[j]);

                pij = (beta * mu - alpha * csbar) * mu/rhobar;

            } // if (vr < 0)


#if NAVIER_STOKES
            eta = (particles->eta[i] + particles->eta[j]) * 0.5 ;
            for (d = 0; d < DIM; d++) {
                accelshearj[d] = 0;
                for (dd = 0; dd < DIM; dd++) {
#if (SPH_EQU_VERSION == 1)
//#if SML_CORRECTION
//                    accelshearj[d] += eta * p.m[j] * (p.Tshear[stressIndex(j,d,dd)]/(p.sml_omega[j]*p.rho[j]*p.rho[j])+ p.Tshear[stressIndex(i,d,dd)]/(p.sml_omega[i]*p.rho[i]*p.rho[i])) *dWdx[dd];
//#else // !SML_CORRECTION
                    accelshearj[d] += eta * particles->mass[j] *
                                (particles->Tshear[CudaUtils::stressIndex(j,d,dd)]/(particles->rho[j]*particles->rho[j]) +
                                    particles->Tshear[CudaUtils::stressIndex(i,d,dd)]/(particles->rho[i]*particles->rho[i])) *
                                    dWdx[dd];
//#endif // SML_CORRECTION
#elif (SPH_EQU_VERSION == 2)
//#if SML_CORRECTION
//                    accelshearj[d] += eta * p.m[j] * (p.Tshear[stressIndex(j,d,dd)]+p.Tshear[stressIndex(i,d,dd)])/(p.sml_omega[i]*p.rho[i]*p.sml_omega[j]*p.rho[j]) *dWdx[dd];
//#else // !SML_CORRECTION
                    accelshearj[d] += eta * particles->mass[j] *
                            (particles->Tshear[CudaUtils::stressIndex(j,d,dd)]+particles->Tshear[CudaUtils::stressIndex(i,d,dd)])/
                                (particles->rho[i]*particles->rho[j]) * dWdx[dd];
//#endif // SML_CORRECTION
#endif // SPH_EQU_VERSION
                }
            }

#endif // NAVIER_STOKES

#if SPH_EQU_VERSION == 1
//#if SML_CORRECTION
//            for (d = 0; d < DIM; d++) {
//                accelsj[d] =  -p.m[j] * (p.p[i]/(p.sml_omega[i]*p.rho[i]*p.rho[i]) + p.p[j]/(p.sml_omega[j]*p.rho[j]*p.rho[j])) * dWdx[d];
//                accels[d] += accelsj[d];
//            }
//#else // !if SML_CORRECTION
            for (d = 0; d < DIM; d++) {
                accelsj[d] =  -particles->mass[j] * (particles->p[i]/(particles->rho[i]*particles->rho[i]) +
                                                     particles->p[j]/(particles->rho[j]*particles->rho[j])) * dWdx[d];
                accels[d] += accelsj[d];
            } // for (d = 0; d < DIM; d++)
//#endif // SML_CORRECTION
#elif SPH_EQU_VERSION == 2
//#if SML_CORRECTION
//            for (d = 0; d < DIM; d++) {
//                accelsj[d] =  -p.m[j] * ((p.p[i]+p.p[j])/(p.sml_omega[i]*p.rho[i]*p.sml_omega[j]*p.rho[j])) * dWdx[d];
//                accels[d] += accelsj[d];
//            }
//#else // !SML_CORRECTION
            for (d = 0; d < DIM; d++) {
                accelsj[d] =  -particles->mass[j] * ((particles->p[i] + particles->p[j])/ (particles->rho[i]*particles->rho[j])) * dWdx[d];
                accels[d] += accelsj[d];
            } //  for (d = 0; d < DIM; d++)
//#  endif // SML_CORRECTION
#else // SPH_EQU_VERSION
            printf("SPH equation representation not available!\n");
#endif // SPH_EQU_VERSION

#if NAVIER_STOKES
            // add viscous accel to total accel
            for (d = 0; d < DIM; d++) {
                accels[d] += accelshearj[d];
            }
#endif // NAVIER_STOKES

            accels[0] += particles->mass[j]*(-pij)*dWdx[0];
#if DIM > 1
            accels[1] += particles->mass[j]*(-pij)*dWdx[1];
#if DIM == 3
            accels[2] += particles->mass[j]*(-pij)*dWdx[2];
#endif // DIM > 1
#endif // DIM == 3

//# if SML_CORRECTION
//            drhodt += p.m[j]*vvnablaW;
//# else // !SML_CORRECTION
            drhodt += particles->rho[i]/particles->rho[j] * particles->mass[j] * vvnablaW;
//# endif // SML_CORRECTION

#if INTEGRATE_SML
    // minus since vvnablaW is v_i - v_j \nabla W_ij
//#  if !SML_CORRECTION
            particles->dsmldt[i] -= 1./DIM * particles->sml[i] * particles->mass[j]/particles->rho[j] * vvnablaW;
//#  endif // !SML_CORRECTION
#endif // INTEGRATE_SML

#if INTEGRATE_ENERGY
            if (true) { // !isRelaxationRun) { //TODO: isRelaxationRun
//#  if SML_CORRECTION
//                dedt += p.m[j] * vvnablaW;
//#  else // !SML_CORRECTION
                dedt += 0.5 * particles->mass[j] * pij * vvnablaW;
//#  endif // SML_CORRECTION
            } // if (true) { // !isRelaxationRun)

            // remember, accelsj  are accelerations by particle j, and dv = v_i - v_j
            dedt += 0.5 * accelsj[0] * -dvx;
#if DIM > 1
            dedt += 0.5 * accelsj[1] * -dvy;
#if DIM == 3
            dedt += 0.5 * accelsj[2] * -dvz;
#endif // DIM > 1
#endif // DIM == 3

#endif // INTEGRATE ENERGY

        } // for (k = 0; k < numInteractions; k++) // neighbors loop end

        ax = accels[0];
#if DIM > 1
        ay = accels[1];
#if DIM == 3
        az = accels[2];
#endif // DIM > 1
#endif // DIM == 3
        particles->ax[i] = ax;
#if DIM > 1
        particles->ay[i] = ay;
#if DIM == 3
        particles->az[i] = az;
#endif // DIM > 1
#endif // DIM == 3

//#if SML_CORRECTION
//        p.drhodt[i] = 1 / p.sml_omega[i] * drhodt;
//        p.dhdt[i] = - p.h[i] / (DIM * p.rho[i]) * p.drhodt[i];
//#else // !SML_CORRECTION
        particles->drhodt[i] = drhodt;
//#endif // SML_CORRECTION

#if INTEGRATE_ENERGY

//#if SML_CORRECTION
//        p.dedt[i] = p.p[i]/(p.rho[i]*p.rho[i] * p.sml_omega[i]) * dedt;
//#else
        particles->dedt[i] = dedt;
//#endif // SML_CORRECTION
#endif // INTEGRATE_ENERGY

    } // particle loop end // for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numRealParticles; i += inc)
}


