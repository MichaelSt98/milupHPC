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

    cuda::malloc(d_numParticles, 1);
    cuda::malloc(d_numNodes, sizeof(integer));

    cuda::malloc(d_mass, numNodes);
    cuda::malloc(d_x, numNodes);
    cuda::malloc(d_vx, numNodes);
    cuda::malloc(d_ax, numNodes);
#if DIM > 1
    cuda::malloc(d_y, numNodes);
    cuda::malloc(d_vy, numNodes);
    cuda::malloc(d_ay, numNodes);
#if DIM == 3
    cuda::malloc(d_z, numNodes);
    cuda::malloc(d_vz, numNodes);
    cuda::malloc(d_az, numNodes);
#endif
#endif
    cuda::malloc(d_uid, numParticles);
    cuda::malloc(d_materialId, numParticles);
    cuda::malloc(d_sml, numParticles);
    cuda::malloc(d_nnl, numParticles * MAX_NUM_INTERACTIONS);
    cuda::malloc(d_noi, numParticles);
    cuda::malloc(d_e, numParticles);
    cuda::malloc(d_dedt, numParticles);
    cuda::malloc(d_cs, numParticles);
    cuda::malloc(d_rho, numParticles);
    cuda::malloc(d_p, numParticles);

#if INTEGRATE_DENSITY
    cuda::malloc(d_drhodt, numParticles);
#endif
#if VARIABLE_SML
    cuda::malloc(d_dsmldt, numParticles);
#endif
#if SOLID
    cuda::malloc(d_S, DIM * DIM *numParticles);
    cuda::malloc(d_dSdt, DIM * numParticles);
    cuda::malloc(d_localStrain, numParticles);
#endif
#if SOLID || NAVIER_STOKES
    cuda::malloc(d_sigma, DIM * DIM * numParticles);
#endif
#if ARTIFICIAL_STRESS
    cuda::malloc(d_R, DIM * DIM * numParticles);
#endif
#if POROSITY
    cuda::malloc(d_pold, numParticles);
    cuda::malloc(d_alpha_jutzi, numParticles);
    cuda::malloc(d_alpha_jutzi_old, numParticles);
    cuda::malloc(d_dalphadt, numParticles);
    cuda::malloc(d_dalphadp, numParticles);
    cuda::malloc(d_dp, numParticles);
    cuda::malloc(d_dalphadrho, numParticles);
    cuda::malloc(d_f, numParticles);
    cuda::malloc(d_delpdelrho, numParticles);
    cuda::malloc(d_delpdele, numParticles);
    cuda::malloc(d_cs_old, numParticles);
    cuda::malloc(d_alpha_epspor, numParticles);
    cuda::malloc(d_dalpha_epspordt, numParticles);
    cuda::malloc(d_epsilon_v, numParticles);
    cuda::malloc(d_depsilon_vdt, numParticles);
#endif
#if ZERO_CONSISTENCY
    cuda::malloc(d_shepardCorrection, numParticles);
#endif
#if LINEAR_CONSISTENCY
    cuda::malloc(d_tensorialCorrectionMatrix, DIM * DIM * numParticles);
#endif
#if FRAGMENTATION
    cuda::malloc(d_d, numParticles);
    cuda::malloc(d_damage_total, numParticles);
    cuda::malloc(d_dddt, numParticles);
    cuda::malloc(d_numFlaws, numParticles);
    cuda::malloc(d_maxNumFlaws, numParticles);
    cuda::malloc(d_numActiveFlaws, numParticles);
    cuda::malloc(d_flaws, numParticles);
#if PALPHA_POROSITY
    cuda::malloc(d_damage_porjutzi, numParticles);
    cuda::malloc(d_ddamage_porjutzidt, numParticles);
#endif
#endif

    cuda::malloc(d_particles, 1);

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

    cuda::copy(&numParticles, d_numParticles, 1, To::device);
    cuda::copy(&numNodes, d_numNodes, 1, To::device);

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
    cuda::free(d_numParticles);
    cuda::free(d_numNodes);

    cuda::free(d_mass);
    cuda::free(d_x);
    cuda::free(d_vx);
    cuda::free(d_ax);
#if DIM > 1
    cuda::free(d_y);
    cuda::free(d_vy);
    cuda::free(d_ay);
#if DIM == 3
    cuda::free(d_z);
    cuda::free(d_vz);
    cuda::free(d_az);
#endif
#endif
    cuda::free(d_uid);
    cuda::free(d_materialId);
    cuda::free(d_sml);
    cuda::free(d_nnl);
    cuda::free(d_noi);
    cuda::free(d_e);
    cuda::free(d_dedt);
    cuda::free(d_cs);
    cuda::free(d_rho);
    cuda::free(d_p);

#if INTEGRATE_DENSITY
    delete [] h_drhodt;
    cuda::free(d_drhodt);
#endif
#if VARIABLE_SML
    delete [] h_dsmldt;
    cuda::free(d_dsmldt);
#endif
#if SOLID
    delete [] h_S;
    cuda::free(d_S);
    delete [] h_dSdt;
    cuda::free(d_dSdt);
    delete [] h_localStrain;
    cuda::free(d_localStrain);
#endif
#if SOLID || NAVIER_STOKES
    delete [] h_sigma;
    cuda::free(d_sigma);
#endif
#if ARTIFICIAL_STRESS
    delete [] h_R;
    cuda::free(d_R);
#endif
#if POROSITY
    delete [] h_pold;
    cuda::free(d_pold);
    delete [] h_alpha_jutzi;
    cuda::free(d_alpha_jutzi);
    delete [] h_alpha_jutzi_old;
    cuda::free(d_alpha_jutzi_old);
    delete [] h_dalphadt;
    cuda::free(d_dalphadt);
    delete [] h_dalphadp;
    cuda::free(d_dalphadp);
    delete [] h_dp;
    cuda::free(d_dp);
    delete [] h_dalphadrho;
    cuda::free(d_dalphadrho);
    delete [] h_f;
    cuda::free(d_f);
    delete [] h_delpdelrho;
    cuda::free(d_delpdelrho);
    delete [] h_delpdele;
    cuda::free(d_delpdele);
    delete [] h_cs_old;
    cuda::free(d_cs_old);
    delete [] h_alpha_epspor;
    cuda::free(d_alpha_epspor);
    delete [] h_dalpha_epspordt;
    cuda::free(d_dalpha_epspordt);
    delete [] h_epsilon_v;
    cuda::free(d_epsilon_v);
    delete [] h_depsilon_vdt;
    cuda::free(d_depsilon_vdt);
#endif
#if ZERO_CONSISTENCY
    delete [] h_shepardCorrection;
    cuda::free(d_shepardCorrection);
#endif
#if LINEAR_CONSISTENCY
    delete [] h_tensorialCorrectionMatrix;
    cuda::free(d_tensorialCorrectionMatrix);
#endif
#if FRAGMENTATION
    delete [] h_d;
    cuda::free(d_d);
    delete [] h_damage_total;
    cuda::free(d_damage_total);
    delete [] h_dddt;
    cuda::free(d_dddt);
    delete [] h_numFlaws;
    cuda::free(d_numFlaws);
    delete [] h_maxNumFlaws;
    cuda::free(d_maxNumFlaws);
    delete [] h_numActiveFlaws;
    cuda::free(d_numActiveFlaws);
    delete [] h_flaws;
    cuda::free(d_flaws);
#if PALPHA_POROSITY
    delete [] h_damage_porjutzi;
    cuda::free(d_damage_porjutzi);
    delete [] h_ddamage_porjutzidt;
    cuda::free(d_ddamage_porjutzidt);
#endif
#endif

    delete h_particles;
    cuda::free(d_particles);

}

template <typename T>
T*& ParticleHandler::getEntry(Entry::Name entry, Execution::Location location) {
    switch (location) {
        case Execution::device: {
            switch (entry) {
                case Entry::x: {
                    return d_x;
                }
#if DIM > 1
                case Entry::y: {
                    return d_y;
                } break;
#if DIM == 3
                case Entry::z: {
                    return d_z;
                } break;
#endif
#endif
                case Entry::mass: {
                    return d_mass;
                } break;
                default: {
                    printf("Entry is not available!\n");
                    return NULL;
                }
            }
        } break;
        case Execution::host: {
            switch (entry) {
                case Entry::x: {
                    return h_x;
                }
#if DIM > 1
                case Entry::y: {
                    return h_y;
                } break;
#if DIM == 3
                case Entry::z: {
                    return h_z;
                } break;
#endif
#endif
                case Entry::mass: {
                    return h_mass;
                } break;
                default: {
                    printf("Entry is not available!\n");
                    return NULL;
                }
            }
        } break;
        default: {
            printf("Location is not available!\n");
            return NULL;
        }
    }
}

void ParticleHandler::copyMass(To::Target target) {
    cuda::copy(h_mass, d_mass, numParticles, target);
}

void ParticleHandler::copyPosition(To::Target target) {
    cuda::copy(h_x, d_x, numParticles, target);
#if DIM > 1
    cuda::copy(h_y, d_y, numParticles, target);
#if DIM == 3
    cuda::copy(h_z, d_z, numParticles, target);
#endif
#endif
}

void ParticleHandler::copyVelocity(To::Target target) {
    cuda::copy(h_vx, d_vx, numParticles, target);
#if DIM > 1
    cuda::copy(h_vy, d_vy, numParticles, target);
#if DIM == 3
    cuda::copy(h_vz, d_vz, numParticles, target);
#endif
#endif
}

void ParticleHandler::copyAcceleration(To::Target target) {
    cuda::copy(h_ax, d_ax, numParticles, target);
#if DIM > 1
    cuda::copy(h_ay, d_ay, numParticles, target);
#if DIM == 3
    cuda::copy(h_az, d_az, numParticles, target);
#endif
#endif
}

void ParticleHandler::copyDistribution(To::Target target, bool velocity, bool acceleration) {
    copyMass(target);
    copyPosition(target);
    if (velocity) {
        copyVelocity(target);
    }
    if (acceleration) {
        copyAcceleration(target);
    }
}

IntegratedParticleHandler::IntegratedParticleHandler(integer numParticles, integer numNodes) :
                                                        numParticles(numParticles), numNodes(numNodes) {

    cuda::malloc(d_uid, numParticles);
    cuda::malloc(d_drhodt, numParticles);

    cuda::malloc(d_dxdt, numParticles);
    cuda::malloc(d_dvxdt, numParticles);
#if DIM > 1
    cuda::malloc(d_dydt, numParticles);
    cuda::malloc(d_dvydt, numParticles);
#if DIM == 3
    cuda::malloc(d_dzdt, numParticles);
    cuda::malloc(d_dvzdt, numParticles);
#endif
#endif

    cuda::malloc(d_integratedParticles, 1);

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

    cuda::free(d_uid);
    cuda::free(d_drhodt);

    cuda::free(d_dxdt);
    cuda::free(d_dvxdt);
#if DIM > 1
    cuda::free(d_dydt);
    cuda::free(d_dvydt);
#if DIM == 3
    cuda::free(d_dzdt);
    cuda::free(d_dvzdt);
#endif
#endif
    cuda::free(d_integratedParticles);

}


