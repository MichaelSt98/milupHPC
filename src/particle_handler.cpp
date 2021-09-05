#include "../include/particle_handler.h"

ParticleHandler::ParticleHandler(integer numParticles, integer numNodes) : numParticles(numParticles),
                                                                            numNodes(numNodes) {

    Logger(INFO) << "numParticles: " << numParticles << "   numNodes: " << numNodes;

    h_mass = new real[numNodes];
    h_x = new real[numNodes];
    h_vx = new real[numNodes];
    h_ax = new real[numNodes];
#if DIM > 1
    h_y = new real[numNodes];
    h_vy = new real[numNodes];
    h_ay = new real[numNodes];
#if DIM == 3
    h_z = new real[numNodes];
    h_vz = new real[numNodes];
    h_az = new real[numNodes];
#endif
#endif
    h_uid = new idInteger[numParticles];
    h_materialId = new integer[numParticles];
    h_sml = new real[numParticles];
    h_nnl = new integer[numParticles * MAX_NUM_INTERACTIONS];
    h_noi = new integer [numParticles];
    h_e = new real[numParticles];
    h_dedt = new real[numParticles];
    h_cs = new real[numParticles];
    h_rho = new real[numParticles];
    h_p = new real[numParticles];

#if INTEGRATE_DENSITY
    h_drhodt = new real[numParticles];
#endif
#if VARIABLE_SML
    h_dsmldt = new real[numParticles];
#endif
#if SOLID
    h_S = new real[DIM * DIM * numParticles];
    h_dSdt = new real[DIM * DIM * numParticles];
    h_localStrain = new real[numParticles];
#endif
#if SOLID || NAVIER_STOKES
    h_sigma = new real[DIM * DIM * numParticles];
#endif
#if ARTIFICIAL_STRESS
    h_R = new real[DIM * DIM * numParticles];;
#endif
#if POROSITY
    h_pold = new real[numParticles];
    h_alpha_jutzi = new real[numParticles];
    h_alpha_jutzi_old = new real[numParticles];
    h_dalphadt = new real[numParticles];
    h_dalphadp = new real[numParticles];
    h_dp = new real[numParticles];
    h_dalphadrho = new real[numParticles];
    h_f = new real[numParticles];
    h_delpdelrho = new real[numParticles];
    h_delpdele = new real[numParticles];
    h_cs_old = new real[numParticles];
    h_alpha_epspor = new real[numParticles];
    h_dalpha_epspordt = new real[numParticles];
    h_epsilon_v = new real[numParticles];
    h_depsilon_vdt = new real[numParticles];
#endif
#if ZERO_CONSISTENCY
    h_shepardCorrection = new real[numParticles];
#endif
#if LINEAR_CONSISTENCY
    h_tensorialCorrectionMatrix = new real[DIM * DIM * numParticles];
#endif
#if FRAGMENTATION
    h_d = new real[numParticles];
    h_damage_total = new real[numParticles];
    h_dddt = new real[numParticles];
    h_numFlaws = new integer[numParticles];
    h_maxNumFlaws = new integer[numParticles];
    h_numActiveFlaws = new integer[numParticles];
    h_flaws = new real[numParticles];
#if PALPHA_POROSITY
    h_damage_porjutzi = new real[numParticles];
    h_ddamage_porjutzidt = new real[numParticles];
#endif
#endif

    h_particles = new Particles();

    gpuErrorcheck(cudaMalloc((void**)&d_numParticles, sizeof(integer)));
    gpuErrorcheck(cudaMalloc((void**)&d_numNodes, sizeof(integer)));

    gpuErrorcheck(cudaMalloc((void**)&d_mass, numNodes * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_x, numNodes * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_vx, numNodes * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_ax, numNodes * sizeof(real)));
#if DIM > 1
    gpuErrorcheck(cudaMalloc((void**)&d_y, numNodes * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_vy, numNodes * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_ay, numNodes * sizeof(real)));
#if DIM == 3
    gpuErrorcheck(cudaMalloc((void**)&d_z, numNodes * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_vz, numNodes * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_az, numNodes * sizeof(real)));
#endif
#endif
    gpuErrorcheck(cudaMalloc((void**)&d_uid, numParticles * sizeof(idInteger)));
    gpuErrorcheck(cudaMalloc((void**)&d_materialId, numParticles * sizeof(integer)));
    gpuErrorcheck(cudaMalloc((void**)&d_sml, numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_nnl, numParticles * MAX_NUM_INTERACTIONS * sizeof(integer)));
    gpuErrorcheck(cudaMalloc((void**)&d_noi, numParticles * sizeof(integer)));
    gpuErrorcheck(cudaMalloc((void**)&d_e, numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_dedt, numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_cs, numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_rho, numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_p, numParticles * sizeof(real)));

#if INTEGRATE_DENSITY
    gpuErrorcheck(cudaMalloc((void**)&d_drhodt, numParticles * sizeof(real)));
#endif
#if VARIABLE_SML
    gpuErrorcheck(cudaMalloc((void**)&d_dsmldt, numParticles * sizeof(real)));
#endif
#if SOLID
    gpuErrorcheck(cudaMalloc((void**)&d_S, DIM * DIM *numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_dSdt, DIM * numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_localStrain, numParticles * sizeof(real)));
#endif
#if SOLID || NAVIER_STOKES
    gpuErrorcheck(cudaMalloc((void**)&d_sigma, DIM * DIM * numParticles * sizeof(real)));
#endif
#if ARTIFICIAL_STRESS
    gpuErrorcheck(cudaMalloc((void**)&d_R, DIM * DIM * numParticles * sizeof(real)));
#endif
#if POROSITY
    gpuErrorcheck(cudaMalloc((void**)&d_pold, numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_alpha_jutzi, numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_alpha_jutzi_old, numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_dalphadt, numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_dalphadp, numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_dp, numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_dalphadrho, numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_f, numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_delpdelrho, numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_delpdele, numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_cs_old, numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_alpha_epspor, numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_dalpha_epspordt, numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_epsilon_v, numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_depsilon_vdt, numParticles * sizeof(real)));
#endif
#if ZERO_CONSISTENCY
    gpuErrorcheck(cudaMalloc((void**)&d_shepardCorrection, numParticles * sizeof(real)));
#endif
#if LINEAR_CONSISTENCY
    gpuErrorcheck(cudaMalloc((void**)&d_tensorialCorrectionMatrix, DIM * DIM * numParticles * sizeof(real)));
#endif
#if FRAGMENTATION
    gpuErrorcheck(cudaMalloc((void**)&d_d, numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_damage_total, numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_dddt, numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_numFlaws, numParticles * sizeof(integer)));
    gpuErrorcheck(cudaMalloc((void**)&d_maxNumFlaws, numParticles * sizeof(integer)));
    gpuErrorcheck(cudaMalloc((void**)&d_numActiveFlaws, numParticles * sizeof(integer)));
    gpuErrorcheck(cudaMalloc((void**)&d_flaws, numParticles * sizeof(real)));
#if PALPHA_POROSITY
    gpuErrorcheck(cudaMalloc((void**)&d_damage_porjutzi, numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_ddamage_porjutzidt, numParticles * sizeof(real)));
#endif
#endif

    gpuErrorcheck(cudaMalloc((void**)&d_particles, sizeof(Particles)));

#if DIM == 1
    h_particles->set(numParticles, numNodes, h_mass, h_x, h_vx, h_ax, h_uid, h_materialId, h_sml, h_nnl, h_noi, h_e,
                     h_dedt, h_cs, h_rho, h_p);
    ParticlesNS::Kernel::Launch::set(d_particles, d_numParticles, d_numNodes, d_mass, d_x, d_vx, d_ax, d_uid,
                                     d_materialId, d_sml, d_nnl, d_noi, d_e, d_dedt, d_cs, d_rho, d_p);
#elif DIM == 2
    h_particles->set(numParticles, numNodes, h_mass, h_x, h_y, h_vx, h_vy, h_ax, h_ay, h_uid, h_materialId, h_sml,
                     h_nnl, h_noi, h_e, h_dedt, h_cs, h_rho, h_p);
    ParticlesNS::Kernel::Launch::set(d_particles, d_numParticles, d_numNodes, d_mass, d_x, d_y, d_vx, d_vy, d_ax, d_ay,
                                     d_uid, d_materialId, d_sml, d_nnl, d_noi, d_e, d_dedt, d_cs, d_rho, d_p);
#else
    h_particles->set(&numParticles, &numNodes, h_mass, h_x, h_y, h_z, h_vx, h_vy, h_vz, h_ax, h_ay, h_az, h_uid,
                     h_materialId, h_sml, h_nnl, h_noi, h_e, h_dedt, h_cs, h_rho, h_p);
    ParticlesNS::Kernel::Launch::set(d_particles, d_numParticles, d_numNodes, d_mass, d_x, d_y, d_z, d_vx, d_vy, d_vz,
                                     d_ax, d_ay, d_az, d_uid, d_materialId, d_sml, d_nnl, d_noi, d_e, d_dedt, d_cs,
                                     d_rho, d_p);
#endif

#if INTEGRATE_DENSITY
    h_particles->setIntegrateDensity(h_drhodt);
    ParticlesNS::Kernel::Launch::setIntegrateDensity(d_particles, d_drhodt);
#endif
#if VARIABLE_SML
    h_particles->setVariableSML(h_dsmldt);
    ParticlesNS::Kernel::Launch::setVariableSML(d_particles, d_dsmldt);
#endif
#if SOLID
    h_particles->setSolid(h_S, h_dSdt, h_localStrain);
    ParticlesNS::Kernel::Launch::setSolid(d_particles, d_S, d_dSdt, d_localStrain);
#endif
#if SOLID || NAVIER_STOKES
    h_particles->setNavierStokes(h_sigma);
    ParticlesNS::Kernel::Launch::setNavierStokes(d_particles, d_sigma);
#endif
#if ARTIFICIAL_STRESS
    h_particles->setArtificialStress(h_R);
    ParticlesNS::Kernel::Launch::setArtificialStress(d_particles, d_R);
#endif
#if POROSITY
    h_particles->setPorosity(h_pold, h_alpha_jutzi, h_alpha_jutzi_old, h_dalphadt, h_dalphadp, h_dp, h_dalphadrho, h_f,
                             h_delpdelrho, h_delpdele, h_cs_old, h_alpha_epspor, h_dalpha_epspordt, h_epsilon_v,
                             h_depsilon_vdt);
    ParticlesNS::Kernel::Launch::setPorosity(d_particles, d_pold, d_alpha_jutzi, d_alpha_jutzi_old, d_dalphadt,
                                             d_dalphadp, d_dp, d_dalphadrho, d_f, d_delpdelrho, d_delpdele, d_cs_old,
                                             d_alpha_epspor, d_dalpha_epspordt, d_epsilon_v, d_depsilon_vdt);
#endif
#if ZERO_CONSISTENCY
    h_particles->setZeroConsistency(h_shepardCorrection);
    ParticlesNS::Kernel::Launch::setZeroConsistency(d_particles, d_shepardCorrection);
#endif
#if LINEAR_CONSISTENCY
    h_particles->setLinearConsistency(h_tensorialCorrectionMatrix);
    ParticlesNS::Kernel::Launch::setLinearConsistency(d_particles, d_tensorialCorrectionMatrix);
#endif
#if FRAGMENTATION
    h_particles->setFragmentation(h_d, h_damage_total, h_dddt, h_numFlaws, h_maxNumFlaws,
                                  h_numActiveFlaws, h_flaws);
    ParticlesNS::Kernel::Launch::setFragmentation(d_particles, d_d, d_damage_total, d_dddt, d_numFlaws, d_maxNumFlaws,
                                                  d_numActiveFlaws, d_flaws);
#if PALPHA_POROSITY
    h_particles->setPalphaPorosity(h_damage_porjutzi, h_ddamage_porjutzidt);
    ParticlesNS::Kernel::Launch::setPalphaPorosity(d_particles, d_damage_porjutzi, d_ddamage_porjutzidt);
#endif
#endif

    //gpuErrorcheck(cudaMemset(d_numParticles, numParticles, sizeof(integer)));
    //gpuErrorcheck(cudaMemset(d_numNodes, numNodes, sizeof(integer)));
    gpuErrorcheck(cudaMemcpy(d_numParticles, &numParticles, sizeof(integer), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_numNodes, &numNodes, sizeof(integer), cudaMemcpyHostToDevice));

}

ParticleHandler::~ParticleHandler() {

    delete [] h_mass;
    delete [] h_x;
    delete [] h_vx;
    delete [] h_ax;
#if DIM > 1
    delete [] h_y;
    delete [] h_vy;
    delete [] h_ay;
#if DIM == 3
    delete [] h_z;
    delete [] h_vz;
    delete [] h_az;
#endif
#endif
    delete [] h_uid;
    delete [] h_materialId;
    delete [] h_sml;
    delete [] h_nnl;
    delete [] h_noi;
    delete [] h_e;
    delete [] h_dedt;
    delete [] h_cs;
    delete [] h_rho;
    delete [] h_p;

    // device particle entries
    gpuErrorcheck(cudaFree(d_numParticles));
    gpuErrorcheck(cudaFree(d_numNodes));

    gpuErrorcheck(cudaFree(d_mass));
    gpuErrorcheck(cudaFree(d_x));
    gpuErrorcheck(cudaFree(d_vx));
    gpuErrorcheck(cudaFree(d_ax));
#if DIM > 1
    gpuErrorcheck(cudaFree(d_y));
    gpuErrorcheck(cudaFree(d_vy));
    gpuErrorcheck(cudaFree(d_ay));
#if DIM == 3
    gpuErrorcheck(cudaFree(d_z));
    gpuErrorcheck(cudaFree(d_vz));
    gpuErrorcheck(cudaFree(d_az));
#endif
#endif
    gpuErrorcheck(cudaFree(d_uid));
    gpuErrorcheck(cudaFree(d_materialId));
    gpuErrorcheck(cudaFree(d_sml));
    gpuErrorcheck(cudaFree(d_nnl));
    gpuErrorcheck(cudaFree(d_noi));
    gpuErrorcheck(cudaFree(d_e));
    gpuErrorcheck(cudaFree(d_dedt));
    gpuErrorcheck(cudaFree(d_cs));
    gpuErrorcheck(cudaFree(d_rho));
    gpuErrorcheck(cudaFree(d_p));

#if INTEGRATE_DENSITY
    delete [] h_drhodt;
    gpuErrorcheck(cudaFree(d_drhodt));
#endif
#if VARIABLE_SML
    delete [] h_dsmldt;
    gpuErrorcheck(cudaFree(d_dsmldt));
#endif
#if SOLID
    delete [] h_S;
    gpuErrorcheck(cudaFree(d_S));
    delete [] h_dSdt;
    gpuErrorcheck(cudaFree(d_dSdt));
    delete [] h_localStrain;
    gpuErrorcheck(cudaFree(d_localStrain));
#endif
#if SOLID || NAVIER_STOKES
    delete [] h_sigma;
    gpuErrorcheck(cudaFree(d_sigma));
#endif
#if ARTIFICIAL_STRESS
    delete [] h_R;
    gpuErrorcheck(cudaFree(d_R));
#endif
#if POROSITY
    delete [] h_pold;
    gpuErrorcheck(cudaFree(d_pold));
    delete [] h_alpha_jutzi;
    gpuErrorcheck(cudaFree(d_alpha_jutzi));
    delete [] h_alpha_jutzi_old;
    gpuErrorcheck(cudaFree(d_alpha_jutzi_old));
    delete [] h_dalphadt;
    gpuErrorcheck(cudaFree(d_dalphadt));
    delete [] h_dalphadp;
    gpuErrorcheck(cudaFree(d_dalphadp));
    delete [] h_dp;
    gpuErrorcheck(cudaFree(d_dp));
    delete [] h_dalphadrho;
    gpuErrorcheck(cudaFree(d_dalphadrho));
    delete [] h_f;
    gpuErrorcheck(cudaFree(d_f));
    delete [] h_delpdelrho;
    gpuErrorcheck(cudaFree(d_delpdelrho));
    delete [] h_delpdele;
    gpuErrorcheck(cudaFree(d_delpdele));
    delete [] h_cs_old;
    gpuErrorcheck(cudaFree(d_cs_old));
    delete [] h_alpha_epspor;
    gpuErrorcheck(cudaFree(d_alpha_epspor));
    delete [] h_dalpha_epspordt;
    gpuErrorcheck(cudaFree(d_dalpha_epspordt));
    delete [] h_epsilon_v;
    gpuErrorcheck(cudaFree(d_epsilon_v));
    delete [] h_depsilon_vdt;
    gpuErrorcheck(cudaFree(d_depsilon_vdt));
#endif
#if ZERO_CONSISTENCY
    delete [] h_shepardCorrection;
    gpuErrorcheck(cudaFree(d_shepardCorrection));
#endif
#if LINEAR_CONSISTENCY
    delete [] h_tensorialCorrectionMatrix;
    gpuErrorcheck(cudaFree(d_tensorialCorrectionMatrix));
#endif
#if FRAGMENTATION
    delete [] h_d;
    gpuErrorcheck(cudaFree(d_d));
    delete [] h_damage_total;
    gpuErrorcheck(cudaFree(d_damage_total));
    delete [] h_dddt;
    gpuErrorcheck(cudaFree(d_dddt));
    delete [] h_numFlaws;
    gpuErrorcheck(cudaFree(d_numFlaws));
    delete [] h_maxNumFlaws;
    gpuErrorcheck(cudaFree(d_maxNumFlaws));
    delete [] h_numActiveFlaws;
    gpuErrorcheck(cudaFree(d_numActiveFlaws));
    delete [] h_flaws;
    gpuErrorcheck(cudaFree(d_flaws));
#if PALPHA_POROSITY
    delete [] h_damage_porjutzi;
    gpuErrorcheck(cudaFree(d_damage_porjutzi));
    delete [] h_ddamage_porjutzidt;
    gpuErrorcheck(cudaFree(d_ddamage_porjutzidt));
#endif
#endif

    delete h_particles;
    gpuErrorcheck(cudaFree(d_particles));

}

void ParticleHandler::positionToDevice() {
    gpuErrorcheck(cudaMemcpy(d_x,  h_x,  numParticles*sizeof(real), cudaMemcpyHostToDevice));
#if DIM > 1
    gpuErrorcheck(cudaMemcpy(d_y,  h_y,  numParticles*sizeof(real), cudaMemcpyHostToDevice));
#if DIM == 3
    gpuErrorcheck(cudaMemcpy(d_z,  h_z,  numParticles*sizeof(real), cudaMemcpyHostToDevice));
#endif
#endif
}
void ParticleHandler::velocityToDevice() {
    gpuErrorcheck(cudaMemcpy(d_vx, h_vx, numParticles*sizeof(real), cudaMemcpyHostToDevice));
#if DIM > 1
    gpuErrorcheck(cudaMemcpy(d_vy, h_vy, numParticles*sizeof(real), cudaMemcpyHostToDevice));
#if DIM == 3
    gpuErrorcheck(cudaMemcpy(d_vz, h_vz, numParticles*sizeof(real), cudaMemcpyHostToDevice));
#endif
#endif
}
void ParticleHandler::accelerationToDevice() {
    gpuErrorcheck(cudaMemcpy(d_ax, h_ax, numParticles*sizeof(real), cudaMemcpyHostToDevice));
#if DIM > 1
    gpuErrorcheck(cudaMemcpy(d_ay, h_ay, numParticles*sizeof(real), cudaMemcpyHostToDevice));
#if DIM == 3
    gpuErrorcheck(cudaMemcpy(d_az, h_az, numParticles*sizeof(real), cudaMemcpyHostToDevice));
#endif
#endif
}

void ParticleHandler::distributionToDevice(bool velocity, bool acceleration) {

    gpuErrorcheck(cudaMemcpy(d_mass,  h_mass,  numParticles*sizeof(real), cudaMemcpyHostToDevice));
    positionToDevice();
    if (velocity) {
        velocityToDevice();
    }
    if (acceleration) {
        accelerationToDevice();
    }

}

void ParticleHandler::positionToHost() {
    gpuErrorcheck(cudaMemcpy(h_x, d_x, numParticles*sizeof(real), cudaMemcpyDeviceToHost));
#if DIM > 1
    gpuErrorcheck(cudaMemcpy(h_y, d_y, numParticles*sizeof(real), cudaMemcpyDeviceToHost));
#if DIM == 3
    gpuErrorcheck(cudaMemcpy(h_z, d_z, numParticles*sizeof(real), cudaMemcpyDeviceToHost));
#endif
#endif
}
void ParticleHandler::velocityToHost() {
    gpuErrorcheck(cudaMemcpy(h_vx, d_vx, numParticles*sizeof(real), cudaMemcpyDeviceToHost));
#if DIM > 1
    gpuErrorcheck(cudaMemcpy(h_vy, d_vy, numParticles*sizeof(real), cudaMemcpyDeviceToHost));
#if DIM == 3
    gpuErrorcheck(cudaMemcpy(h_vz, d_vz, numParticles*sizeof(real), cudaMemcpyDeviceToHost));
#endif
#endif
}
void ParticleHandler::accelerationToHost() {
    gpuErrorcheck(cudaMemcpy(h_x, d_x, numParticles*sizeof(real), cudaMemcpyDeviceToHost));
#if DIM > 1
    gpuErrorcheck(cudaMemcpy(h_y, d_y, numParticles*sizeof(real), cudaMemcpyDeviceToHost));
#if DIM == 3
    gpuErrorcheck(cudaMemcpy(h_z, d_z, numParticles*sizeof(real), cudaMemcpyDeviceToHost));
#endif
#endif
}

void ParticleHandler::distributionToHost(bool velocity, bool acceleration) {

    gpuErrorcheck(cudaMemcpy(h_mass, d_mass, numParticles*sizeof(real), cudaMemcpyDeviceToHost));
    positionToHost();
    if (velocity) {
        velocityToHost();
    }
    if (acceleration) {
        accelerationToDevice();
    }

}



IntegratedParticleHandler::IntegratedParticleHandler(integer numParticles, integer numNodes) :
                                                        numParticles(numParticles), numNodes(numNodes) {

    gpuErrorcheck(cudaMalloc((void**)&d_uid, numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_drhodt, numParticles * sizeof(real)));

    gpuErrorcheck(cudaMalloc((void**)&d_dxdt, numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_dvxdt, numParticles * sizeof(real)));
#if DIM > 1
    gpuErrorcheck(cudaMalloc((void**)&d_dydt, numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_dvydt, numParticles * sizeof(real)));
#if DIM == 3
    gpuErrorcheck(cudaMalloc((void**)&d_dzdt, numParticles * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_dvzdt, numParticles * sizeof(real)));
#endif
#endif

    gpuErrorcheck(cudaMalloc((void**)&d_integratedParticles, sizeof(IntegratedParticles)));

#if DIM == 1
    IntegratedParticlesNS::Kernel::Launch::set(d_integratedParticles, d_uid, d_drhodt, d_dxdt, d_dvxdt);
#elif DIM == 2
    IntegratedParticlesNS::Kernel::Launch::set(d_integratedParticles, d_uid, d_drhodt, d_dxdt, d_dydt, d_dvxdt,
                                               d_dvydt);
#else
    IntegratedParticlesNS::Kernel::Launch::set(d_integratedParticles, d_uid, d_drhodt, d_dxdt, d_dydt, d_dzdt,
                                               d_dvxdt, d_dvydt, d_dvzdt);
#endif

}

IntegratedParticleHandler::~IntegratedParticleHandler() {

    gpuErrorcheck(cudaFree(d_uid));
    gpuErrorcheck(cudaFree(d_drhodt));

    gpuErrorcheck(cudaFree(d_dxdt));
    gpuErrorcheck(cudaFree(d_dvxdt));
#if DIM > 1
    gpuErrorcheck(cudaFree(d_dydt));
    gpuErrorcheck(cudaFree(d_dvydt));
#if DIM == 3
    gpuErrorcheck(cudaFree(d_dzdt));
    gpuErrorcheck(cudaFree(d_dvzdt));
#endif
#endif
    gpuErrorcheck(cudaFree(d_integratedParticles));

}


