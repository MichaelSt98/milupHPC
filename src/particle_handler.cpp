#include "../include/particle_handler.h"

ParticleHandler::ParticleHandler(integer numParticles, integer numNodes) : numParticles(numParticles),
                                                                            numNodes(numNodes) {

    Logger(INFO) << "numParticles: " << numParticles << "   numNodes: " << numNodes;

    h_mass = new real[numNodes];
    _h_x = new real[numNodes];
    h_x = _h_x;
    _h_vx = new real[numParticles]; // numNodes
    h_vx = _h_vx;
    _h_ax = new real[numParticles]; // numNodes
    h_ax = _h_ax;
    h_g_ax = new real[numParticles]; // numNodes
#if DIM > 1
    _h_y = new real[numNodes];
    h_y = _h_y;
    _h_vy = new real[numParticles]; // numNodes
    h_vy = _h_vy;
    _h_ay = new real[numParticles]; // numNodes
    h_ay = _h_ay;
    h_g_ay = new real[numParticles]; // numNodes
#if DIM == 3
    _h_z = new real[numNodes];
    h_z = _h_z;
    _h_vz = new real[numParticles]; // numNodes
    h_vz = _h_vz;
    _h_az = new real[numParticles]; // numNodes
    h_az = _h_az;
    h_g_az = new real[numParticles]; // numNodes
#endif
#endif

    h_nodeType = new integer[numNodes];
    h_level = new integer[numNodes];
    _h_uid = new idInteger[numParticles];
    h_uid = _h_uid;
    h_materialId = new integer[numParticles];

#if SPH_SIM
    _h_sml = new real[numParticles];
    h_sml = _h_sml;
    h_nnl = new integer[numParticles * MAX_NUM_INTERACTIONS];
    h_noi = new integer [numParticles];

    _h_rho = new real[numParticles];
    h_rho = _h_rho;
    _h_p = new real[numParticles];
    h_p = _h_p;
    _h_e = new real[numParticles];
    h_e = _h_e;
    _h_cs = new real[numParticles];
    h_cs = _h_cs;
    h_u = new real[numParticles];

#if MESHLESS_FINITE_METHOD
    h_omega = new real[numParticles];
    h_massFlux =  new real[numParticles];
    h_vxFlux = new real[numParticles];
    h_psix = new real[numParticles*MAX_NUM_INTERACTIONS];
#if DIM > 2
    h_psiy = new real[numParticles*MAX_NUM_INTERACTIONS];
    h_vyFlux = new real[numParticles];
#if DIM == 3
    h_psiz = new real[numParticles*MAX_NUM_INTERACTIONS];
    h_vzFlux = new real[numParticles];
#endif
#endif
    h_energyFlux = new real[numParticles];
    h_Ncond = new real[numParticles];

#else // avoid allocation of memory for everything that is not needed for the meshless finite methods
    _h_dedt = new real[numParticles];
    h_dedt = _h_dedt;
    h_muijmax = new real[numParticles];

//#if INTEGRATE_DENSITY
    _h_drhodt = new real[numParticles];
    h_drhodt = _h_drhodt;
//#endif
#if VARIABLE_SML || INTEGRATE_SML
    _h_dsmldt = new real[numParticles];
    h_dsmldt = _h_dsmldt;
#endif
#if SML_CORRECTION
    h_sml_omega = new real[numParticles];
#endif
#if NAVIER_STOKES
    h_Tshear = new real[DIM * DIM * numParticles];
    h_eta = new real[DIM * DIM * numParticles];
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

#endif // !MESHLESS_FINITE_METHOD
#endif // SPH_SIM

    h_particles = new Particles();

    cuda::malloc(d_numParticles, 1);
    cuda::malloc(d_numNodes, 1);

    //real *d_positions;
    //cuda::malloc(d_positions, DIM * numNodes);
    cuda::malloc(d_mass, numNodes);
    cuda::malloc(_d_x, numNodes);
    //_d_x = &d_positions[0];
    d_x = _d_x;
    cuda::malloc(_d_vx, numParticles); // numNodes
    d_vx = _d_vx;
    cuda::malloc(_d_ax, numParticles); // numNodes
    d_ax = _d_ax;
    cuda::malloc(d_g_ax, numParticles); // numNodes
#if DIM > 1
    cuda::malloc(_d_y, numNodes);
    //_d_y = &d_positions[numNodes];
    d_y = _d_y;
    cuda::malloc(_d_vy, numParticles); // numNodes
    d_vy = _d_vy;
    cuda::malloc(_d_ay, numParticles); // numNodes
    d_ay = _d_ay;
    cuda::malloc(d_g_ay, numParticles); // numNodes
#if DIM == 3
    cuda::malloc(_d_z, numNodes);
    //_d_z = &d_positions[2 * numNodes];
    d_z = _d_z;
    cuda::malloc(_d_vz, numParticles); // numNodes
    d_vz = _d_vz;
    cuda::malloc(_d_az, numParticles); // numNodes
    d_az = _d_az;
    cuda::malloc(d_g_az, numParticles); // numNodes
#endif
#endif
    cuda::malloc(d_nodeType, numNodes);
    cuda::malloc(d_level, numNodes);
    cuda::malloc(_d_uid, numParticles);
    d_uid = _d_uid;
    cuda::malloc(d_materialId, numParticles);

#if SPH_SIM

    cuda::malloc(_d_sml, numParticles);
    d_sml = _d_sml;
    cuda::malloc(d_nnl, numParticles * MAX_NUM_INTERACTIONS);
    cuda::malloc(d_noi, numParticles);
    cuda::malloc(_d_e, numParticles);
    d_e = _d_e;
    cuda::malloc(_d_cs, numParticles);
    d_cs = _d_cs;
    cuda::malloc(d_u, numParticles);
    cuda::malloc(_d_rho, numParticles);
    d_rho = _d_rho;
    cuda::malloc(_d_p, numParticles);
    d_p = _d_p;

#if MESHLESS_FINITE_METHOD
    cuda::malloc(d_omega, numParticles);
    cuda::malloc(d_psix, numParticles*MAX_NUM_INTERACTIONS);
    cuda::malloc(d_massFlux, numParticles);
    cuda::malloc(d_vxFlux, numParticles);
#if DIM > 1
    cuda::malloc(d_psiy, numParticles*MAX_NUM_INTERACTIONS);
    cuda::malloc(d_vyFlux, numParticles);
#if DIM == 3
    cuda::malloc(d_psiz, numParticles*MAX_NUM_INTERACTIONS);
    cuda::malloc(d_vzFlux, numParticles);
#endif
#endif
    cuda::malloc(d_energyFlux, numParticles);
    cuda::malloc(d_Ncond, numParticles);

#else // avoid allocation of memory for everything that is not needed for the meshless finite methods
    cuda::malloc(_d_dedt, numParticles);
    d_dedt = _d_dedt;
    cuda::malloc(d_muijmax, numParticles);

//#if INTEGRATE_DENSITY
    cuda::malloc(_d_drhodt, numParticles);
    d_drhodt =_d_drhodt;
//#endif
#if VARIABLE_SML || INTEGRATE_SML
    cuda::malloc(_d_dsmldt, numParticles);
    d_dsmldt = _d_dsmldt;
#endif
#if SML_CORRECTION
    cuda::malloc(d_sml_omega, numParticles);
#endif
#if NAVIER_STOKES
    cuda::malloc(d_Tshear, DIM * DIM * numParticles);
    cuda::malloc(d_eta, DIM * DIM * numParticles);
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

#endif // !MESHLESS_FINITE_METHOD
#endif // SPH_SIM

#if BALSARA_SWITCH
    h_divv = new real[numParticles];
    h_curlv = new real[numParticles * DIM];
    cuda::malloc(d_divv, numParticles);
    cuda::malloc(d_curlv, numParticles * DIM);
#endif

    cuda::malloc(d_particles, 1);


#if DIM == 1
    h_particles->set(&numParticles, &numNodes, h_mass, h_x, h_vx, h_ax, h_level, h_uid, h_materialId, h_sml, h_nnl,
                     h_noi, h_e, h_dedt, h_cs, h_rho, h_p);
    ParticlesNS::Kernel::Launch::set(d_particles, d_numParticles, d_numNodes, d_mass, d_x, d_vx, d_ax, d_level, d_uid,
                                     d_materialId, d_sml, d_nnl, d_noi, d_e, d_dedt, d_cs, d_rho, d_p);
#elif DIM == 2
    h_particles->set(&numParticles, &numNodes, h_mass, h_x, h_y, h_vx, h_vy, h_ax, h_ay, h_level, h_uid, h_materialId,
                     h_sml, h_nnl, h_noi, h_e, h_dedt, h_cs, h_rho, h_p);
    ParticlesNS::Kernel::Launch::set(d_particles, d_numParticles, d_numNodes, d_mass, d_x, d_y, d_vx, d_vy, d_ax, d_ay,
                                     d_level, d_uid, d_materialId, d_sml, d_nnl, d_noi, d_e, d_dedt, d_cs, d_rho, d_p);
#else
    h_particles->set(&numParticles, &numNodes, h_mass, h_x, h_y, h_z, h_vx, h_vy, h_vz, h_ax, h_ay, h_az,
                     h_level, h_uid, h_materialId, h_sml, h_nnl, h_noi, h_e, h_dedt, h_cs, h_rho, h_p);
    ParticlesNS::Kernel::Launch::set(d_particles, d_numParticles, d_numNodes, d_mass, d_x, d_y, d_z, d_vx, d_vy, d_vz,
                                     d_ax, d_ay, d_az, d_level, d_uid, d_materialId, d_sml,
                                     d_nnl, d_noi, d_e, d_dedt, d_cs, d_rho, d_p);
#endif

#if DIM == 1
    h_particles->setGravity(h_g_ax);
    ParticlesNS::Kernel::Launch::setGravity(d_particles, d_g_ax);
#elif DIM == 2
    h_particles->setGravity(h_g_ax, h_g_ay);
    ParticlesNS::Kernel::Launch::setGravity(d_particles, d_g_ax, d_g_ay);
#else
    h_particles->setGravity(h_g_ax, h_g_ay, h_g_az);
    ParticlesNS::Kernel::Launch::setGravity(d_particles, d_g_ax, d_g_ay, d_g_az);
#endif

    h_particles->setNodeType(h_nodeType);
    ParticlesNS::Kernel::Launch::setNodeType(d_particles, d_nodeType);
    h_particles->setU(h_u);
    ParticlesNS::Kernel::Launch::setU(d_particles, d_u);

#if MESHLESS_FINITE_METHOD
    //TODO: handle 1D and 2D cases
    h_particles->setMeshlessFinite(h_omega, h_psix, h_psiy, h_psiz,
                                   h_massFlux, h_vxFlux, h_vyFlux, h_vzFlux, h_energyFlux, h_Ncond);
    ParticlesNS::Kernel::Launch::setMeshlessFinite(d_particles, d_omega, d_psix, d_psiy, d_psiz,
                                                   d_massFlux, d_vxFlux, d_vyFlux, d_vzFlux,
                                                   d_energyFlux, d_Ncond);
#endif

#if SPH_SIM
    h_particles->setArtificialViscosity(h_muijmax);
    ParticlesNS::Kernel::Launch::setArtificialViscosity(d_particles, d_muijmax);

//#if INTEGRATE_DENSITY
    h_particles->setIntegrateDensity(h_drhodt);
    ParticlesNS::Kernel::Launch::setIntegrateDensity(d_particles, d_drhodt);
//#endif
#if VARIABLE_SML || INTEGRATE_SML
    h_particles->setVariableSML(h_dsmldt);
    ParticlesNS::Kernel::Launch::setVariableSML(d_particles, d_dsmldt);
#endif
#if SML_CORRECTION
    h_particles->setSMLCorrection(h_sml_omega);
    ParticlesNS::Kernel::Launch::setSMLCorrection(d_particles, d_sml_omega);
#endif
#if NAVIER_STOKES
    h_particles->setNavierStokes(h_Tshear, h_eta);
    ParticlesNS::Kernel::Launch::setNavierStokes(d_particles, d_Tshear, d_eta);
#endif
#if SOLID
    h_particles->setSolid(h_S, h_dSdt, h_localStrain);
    ParticlesNS::Kernel::Launch::setSolid(d_particles, d_S, d_dSdt, d_localStrain);
#endif
#if SOLID || NAVIER_STOKES
    h_particles->setSolidNavierStokes(h_sigma);
    ParticlesNS::Kernel::Launch::setSolidNavierStokes(d_particles, d_sigma);
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
#endif // SPH_SIM

#if BALSARA_SWITCH
    h_particles->setDivCurl(h_divv, h_curlv);
    ParticlesNS::Kernel::Launch::setDivCurl(d_particles, d_divv, d_curlv);
#endif

    cuda::copy(&numParticles, d_numParticles, 1, To::device);
    cuda::copy(&numNodes, d_numNodes, 1, To::device);

}

ParticleHandler::~ParticleHandler() {

    delete [] h_mass;
    delete [] _h_x;
    delete [] _h_vx;
    delete [] _h_ax;
    delete [] h_g_ax;
#if DIM > 1
    delete [] _h_y;
    delete [] _h_vy;
    delete [] _h_ay;
    delete [] h_g_ay;
#if DIM == 3
    delete [] _h_z;
    delete [] _h_vz;
    delete [] _h_az;
    delete [] h_g_az;
#endif
#endif
    delete [] h_nodeType;
    delete [] h_uid;
    delete [] h_materialId;
#if SPH_SIM
    delete [] _h_sml;
    delete [] h_nnl;
    delete [] h_noi;
    delete [] _h_e;
    delete [] _h_dedt;
    delete [] _h_cs;
    delete [] _h_rho;
    delete [] _h_p;
    delete [] h_muijmax;
#endif // SPH_SIM

    // device particle entries
    cuda::free(d_numParticles);
    cuda::free(d_numNodes);

    cuda::free(d_mass);
    cuda::free(_d_x);
    cuda::free(_d_vx);
    cuda::free(_d_ax);
    cuda::free(d_g_ax);
#if DIM > 1
    cuda::free(_d_y);
    cuda::free(_d_vy);
    cuda::free(_d_ay);
    cuda::free(d_g_ay);
#if DIM == 3
    cuda::free(_d_z);
    cuda::free(_d_vz);
    cuda::free(_d_az);
    cuda::free(d_g_az);
#endif
#endif
    cuda::free(d_nodeType);
    cuda::free(d_uid);
    cuda::free(d_materialId);
#if SPH_SIM
    cuda::free(_d_sml);
    cuda::free(d_nnl);
    cuda::free(d_noi);
    cuda::free(_d_e);
    cuda::free(_d_dedt);
    cuda::free(_d_cs);
    cuda::free(_d_rho);
    cuda::free(_d_p);
    cuda::free(d_muijmax);
#endif // SPH_SIM

#if SPH_SIM
    delete [] h_muijmax;
    cuda::free(d_muijmax);

//#if INTEGRATE_DENSITY
    delete [] _h_drhodt;
    cuda::free(_d_drhodt);
//#endif
#if VARIABLE_SML || INTEGRATE_SML
    delete [] _h_dsmldt;
    cuda::free(_d_dsmldt);
#endif
#if SML_CORRECTION
    delete [] h_sml_omega;
    cuda::free(d_sml_omega);
#endif
#if NAVIER_STOKES
    delete [] h_Tshear;
    cuda::free(d_Tshear);
    delete [] h_eta;
    cuda::free(d_eta);
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
#endif // SPH_SIM

#if BALSARA_SWITCH
    delete [] h_divv;
    delete [] h_curlv;
    cuda::free(d_divv);
    cuda::free(d_curlv);
#endif

    delete h_particles;
    cuda::free(d_particles);

}

void ParticleHandler::initLeapfrog() {

    leapfrog = true;

    // TODO: should be numParticles instead of numNodes
    h_ax_old = new real[numNodes];
    h_g_ax_old = new real[numNodes];
    cuda::malloc(d_ax_old, numNodes);
    cuda::set(d_ax_old, (real)0, numNodes);
    cuda::malloc(d_g_ax_old, numNodes);
    cuda::set(d_g_ax_old, (real)0, numNodes);
#if DIM > 1
    h_ay_old = new real[numNodes];
    h_g_ay_old = new real[numNodes];
    cuda::malloc(d_ay_old, numNodes);
    cuda::set(d_ay_old, (real)0, numNodes);
    cuda::malloc(d_g_ay_old, numNodes);
    cuda::set(d_g_ay_old, (real)0, numNodes);
#if DIM == 3
    h_az_old = new real[numNodes];
    h_g_az_old = new real[numNodes];
    cuda::malloc(d_az_old, numNodes);
    cuda::set(d_az_old, (real)0, numNodes);
    cuda::malloc(d_g_az_old, numNodes);
    cuda::set(d_g_az_old, (real)0, numNodes);
#endif
#endif

#if DIM == 1

#elif DIM == 2

#else
    h_particles->setLeapfrog(h_ax_old, h_ay_old, h_az_old, h_g_ax_old, h_g_ay_old, h_g_az_old);
    ParticlesNS::Kernel::Launch::setLeapfrog(d_particles, d_ax_old, d_ay_old, d_az_old, d_g_ax_old, d_g_ay_old, d_g_az_old);
#endif

}

void ParticleHandler::freeLeapfrog() {

    delete h_g_ax_old;
    cuda::free(d_g_ax_old);
#if DIM > 1
    delete h_g_ay_old;
    cuda::free(d_g_ay_old);
#if DIM == 3
    delete h_g_az_old;
    cuda::free(d_g_az_old);
#endif
#endif

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
                }
#if DIM == 3
                case Entry::z: {
                    return d_z;
                }
#endif
#endif
                case Entry::mass: {
                    return d_mass;
                }
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
                }
#if DIM == 3
                case Entry::z: {
                    return h_z;
                }
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

void ParticleHandler::setPointer(IntegratedParticleHandler *integratedParticleHandler) {

    d_x = integratedParticleHandler->d_x;
    d_vx = integratedParticleHandler->d_vx;
    d_ax = integratedParticleHandler->d_ax;
#if DIM > 1
    d_y = integratedParticleHandler->d_y;
    d_vy = integratedParticleHandler->d_vy;
    d_ay = integratedParticleHandler->d_ay;
#if DIM == 3
    d_z = integratedParticleHandler->d_z;
    d_vz = integratedParticleHandler->d_vz;
    d_az = integratedParticleHandler->d_az;
#endif
#endif
    d_uid = integratedParticleHandler->d_uid;

#if SPH_SIM
    d_rho = integratedParticleHandler->d_rho;
    d_e = integratedParticleHandler->d_e;
    d_dedt = integratedParticleHandler->d_dedt;
    d_p = integratedParticleHandler->d_p;
    d_cs = integratedParticleHandler->d_cs;
    d_sml = integratedParticleHandler->d_sml;

//#if INTEGRATE_DENSITY
    d_drhodt = integratedParticleHandler->d_drhodt;
//#endif

#if VARIABLE_SML || INTEGRATE_SML
    d_dsmldt = integratedParticleHandler->d_dsmldt;
#endif
#endif // SPH_SIM

// Already redirected pointers, thus just call setter like in constructor
#if DIM == 1
    h_particles->set(&numParticles, &numNodes, h_mass, h_x, h_vx, h_ax, h_level, h_uid, h_materialId, h_sml, h_nnl,
                     h_noi, h_e, h_dedt, h_cs, h_rho, h_p);
    ParticlesNS::Kernel::Launch::set(d_particles, d_numParticles, d_numNodes, d_mass, d_x, d_vx, d_ax, d_level, d_uid,
                                     d_materialId, d_sml, d_nnl, d_noi, d_e, d_dedt, d_cs, d_rho, d_p);
#elif DIM == 2
    h_particles->set(&numParticles, &numNodes, h_mass, h_x, h_y, h_vx, h_vy, h_ax, h_ay, h_level, h_uid, h_materialId,
                     h_sml, h_nnl, h_noi, h_e, h_dedt, h_cs, h_rho, h_p);
    ParticlesNS::Kernel::Launch::set(d_particles, d_numParticles, d_numNodes, d_mass, d_x, d_y, d_vx, d_vy, d_ax, d_ay,
                                     d_level, d_uid, d_materialId, d_sml, d_nnl, d_noi, d_e, d_dedt, d_cs, d_rho, d_p);
#else
    h_particles->set(&numParticles, &numNodes, h_mass, h_x, h_y, h_z, h_vx, h_vy, h_vz, h_ax, h_ay, h_az,
                     h_level, h_uid, h_materialId, h_sml, h_nnl, h_noi, h_e, h_dedt, h_cs, h_rho, h_p);
    h_particles->setIntegrateDensity(h_drhodt);

#if SPH_SIM
#if VARIABLE_SML || INTEGRATE_SML
    h_particles->setVariableSML(h_dsmldt);
#endif

    ParticlesNS::Kernel::Launch::set(d_particles, d_numParticles, d_numNodes, d_mass, d_x, d_y, d_z, d_vx, d_vy, d_vz,
                                     d_ax, d_ay, d_az, d_level, d_uid, d_materialId, d_sml,
                                     d_nnl, d_noi, d_e, d_dedt, d_cs, d_rho, d_p);
    ParticlesNS::Kernel::Launch::setIntegrateDensity(d_particles, d_drhodt);
#if VARIABLE_SML || INTEGRATE_SML
    ParticlesNS::Kernel::Launch::setVariableSML(d_particles, d_dsmldt);
#endif
#endif
#endif // SPH_SIM

}

void ParticleHandler::resetPointer() {

    d_x = _d_x;
    d_vx = _d_vx;
    d_ax = _d_ax;
#if DIM > 1
    d_y = _d_y;
    d_vy = _d_vy;
    d_ay = _d_ay;
#if DIM == 3
    d_z = _d_z;
    d_vz = _d_vz;
    d_az = _d_az;
#endif
#endif

    d_uid = _d_uid;

#if SPH_SIM
    d_rho = _d_rho;
    d_e = _d_e;
    d_dedt = _d_dedt;
    d_p = _d_p;
    d_cs = _d_cs;
    d_sml = _d_sml;

//#if INTEGRATE_DENSITY
    d_drhodt = _d_drhodt;
//#endif

#if VARIABLE_SML || INTEGRATE_SML
    d_dsmldt = _d_dsmldt;
#endif
#endif // SPH_SIM

#if DIM == 1
    h_particles->set(&numParticles, &numNodes, h_mass, h_x, h_vx, h_ax, h_level, h_uid, h_materialId, h_sml, h_nnl,
                     h_noi, h_e, h_dedt, h_cs, h_rho, h_p);
    ParticlesNS::Kernel::Launch::set(d_particles, d_numParticles, d_numNodes, d_mass, d_x, d_vx, d_ax, d_level, d_uid,
                                     d_materialId, d_sml, d_nnl, d_noi, d_e, d_dedt, d_cs, d_rho, d_p);
#elif DIM == 2
    h_particles->set(&numParticles, &numNodes, h_mass, h_x, h_y, h_vx, h_vy, h_ax, h_ay, h_level, h_uid, h_materialId,
                     h_sml, h_nnl, h_noi, h_e, h_dedt, h_cs, h_rho, h_p);
    ParticlesNS::Kernel::Launch::set(d_particles, d_numParticles, d_numNodes, d_mass, d_x, d_y, d_vx, d_vy, d_ax, d_ay,
                                     d_level, d_uid, d_materialId, d_sml, d_nnl, d_noi, d_e, d_dedt, d_cs, d_rho, d_p);
#else
    h_particles->set(&numParticles, &numNodes, h_mass, h_x, h_y, h_z, h_vx, h_vy, h_vz, h_ax, h_ay, h_az,
                     h_level, h_uid, h_materialId, h_sml, h_nnl, h_noi, h_e, h_dedt, h_cs, h_rho, h_p);
    h_particles->setIntegrateDensity(h_drhodt);

#if SPH_SIM
#if VARIABLE_SML || INTEGRATE_SML
    h_particles->setVariableSML(h_dsmldt);
#endif

    ParticlesNS::Kernel::Launch::set(d_particles, d_numParticles, d_numNodes, d_mass, d_x, d_y, d_z, d_vx, d_vy, d_vz,
                                     d_ax, d_ay, d_az, d_level, d_uid, d_materialId, d_sml,
                                     d_nnl, d_noi, d_e, d_dedt, d_cs, d_rho, d_p);
    ParticlesNS::Kernel::Launch::setIntegrateDensity(d_particles, d_drhodt);
#if VARIABLE_SML || INTEGRATE_SML
    ParticlesNS::Kernel::Launch::setVariableSML(d_particles, d_dsmldt);
#endif
#endif
#endif // SPH_SIM

}


void ParticleHandler::copyMass(To::Target target, bool includePseudoParticles) {
    int length;
    if (includePseudoParticles) {
        length = numNodes;
    }
    else {
        length = numParticles;
    }
    cuda::copy(h_mass, d_mass, length, target);
}

void ParticleHandler::copyUid(To::Target target) {
    int length = numParticles;
    cuda::copy(h_uid, d_uid, length, target);
}

void ParticleHandler::copyMatId(To::Target target) {
    int length = numParticles;
    cuda::copy(h_materialId, d_materialId, length, target);
}

void ParticleHandler::copySML(To::Target target) {
    int length = numParticles;
    cuda::copy(h_sml, d_sml, length, target);
}

void ParticleHandler::copyPosition(To::Target target, bool includePseudoParticles) {
    int length;
    if (includePseudoParticles) {
        length = numNodes;
    }
    else {
        length = numParticles;
    }
    cuda::copy(h_x, d_x, length, target);
#if DIM > 1
    cuda::copy(h_y, d_y, length, target);
#if DIM == 3
    cuda::copy(h_z, d_z, length, target);
#endif
#endif
}

void ParticleHandler::copyVelocity(To::Target target, bool includePseudoParticles) {
    int length = numParticles;
    //if (includePseudoParticles) {
    //    length = numNodes;
    //}
    //else {
    //    length = numParticles;
    //}
    cuda::copy(h_vx, d_vx, length, target);
#if DIM > 1
    cuda::copy(h_vy, d_vy, length, target);
#if DIM == 3
    cuda::copy(h_vz, d_vz, length, target);
#endif
#endif
}

void ParticleHandler::copyAcceleration(To::Target target, bool includePseudoParticles) {
    int length = numParticles;
    //if (includePseudoParticles) {
    //    length = numNodes;
    //}
    //else {
    //    length = numParticles;
    //}
    cuda::copy(h_ax, d_ax, length, target);
#if DIM > 1
    cuda::copy(h_ay, d_ay, length, target);
#if DIM == 3
    cuda::copy(h_az, d_az, length, target);
#endif
#endif
}

void ParticleHandler::copyDistribution(To::Target target, bool velocity, bool acceleration, bool includePseudoParticles) {
    copyUid(target);
    copyMass(target, includePseudoParticles);
    copyPosition(target, includePseudoParticles);
    copyMatId(target);
#if SPH_SIM
    copySML(target);
#endif
    if (velocity) {
        copyVelocity(target, includePseudoParticles);
    }
    if (acceleration) {
        copyAcceleration(target, includePseudoParticles);
    }
}

void ParticleHandler::copySPH(To::Target target) {
    int length = numParticles;
    cuda::copy(h_rho, d_rho, length, target);
    cuda::copy(h_p, d_p, length, target);
    cuda::copy(h_e, d_e, length, target);
    cuda::copy(h_sml, d_sml, length, target);
    cuda::copy(h_noi, d_noi, length, target);
    cuda::copy(h_cs, d_cs, length, target);
}

IntegratedParticleHandler::IntegratedParticleHandler(integer numParticles, integer numNodes) :
                                                        numParticles(numParticles), numNodes(numNodes) {

    cuda::malloc(d_uid, numParticles);

    cuda::malloc(d_x, numNodes);
    cuda::malloc(d_vx, numParticles); // numNodes
    cuda::malloc(d_ax, numParticles); // numNodes
#if DIM > 1
    cuda::malloc(d_y, numNodes);
    cuda::malloc(d_vy, numParticles); // numNodes
    cuda::malloc(d_ay, numParticles); // numNodes
#if DIM == 3
    cuda::malloc(d_z, numNodes);
    cuda::malloc(d_vz, numParticles); // numNodes
    cuda::malloc(d_az, numParticles); // numNodes
#endif
#endif

    cuda::malloc(d_rho, numParticles);
    cuda::malloc(d_e, numParticles);
    cuda::malloc(d_dedt, numParticles);
    cuda::malloc(d_p, numParticles);
    cuda::malloc(d_cs, numParticles);

    cuda::malloc(d_sml, numParticles);

//#if INTEGRATE_DENSITY
    cuda::malloc(d_drhodt, numParticles);
//#endif

#if VARIABLE_SML || INTEGRATE_SML
    cuda::malloc(d_dsmldt, numParticles);
#endif

    cuda::malloc(d_integratedParticles, 1);

#if DIM == 1
    IntegratedParticlesNS::Kernel::Launch::set(d_integratedParticles, d_uid, d_rho, d_e, d_dedt, d_p, d_cs, d_x,
                                               d_vx, d_ax);
#elif DIM == 2
    IntegratedParticlesNS::Kernel::Launch::set(d_integratedParticles, d_uid, d_rho, d_e, d_dedt, d_p, d_cs, d_x,
                                               d_y, d_vx, d_vy, d_ax, d_ay);
#else
    IntegratedParticlesNS::Kernel::Launch::set(d_integratedParticles, d_uid, d_rho, d_e, d_dedt, d_p, d_cs, d_x,
                                               d_y, d_z, d_vx, d_vy, d_vz, d_ax, d_ay, d_az);
#endif

    IntegratedParticlesNS::Kernel::Launch::setSML(d_integratedParticles, d_sml);

//#if INTEGRATE_DENSITY
    IntegratedParticlesNS::Kernel::Launch::setIntegrateDensity(d_integratedParticles, d_drhodt);
//#endif

#if VARIABLE_SML || INTEGRATE_SML
    IntegratedParticlesNS::Kernel::Launch::setIntegrateSML(d_integratedParticles, d_dsmldt);
#endif

}

IntegratedParticleHandler::~IntegratedParticleHandler() {

    cuda::free(d_uid);

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

    cuda::free(d_rho);
    cuda::free(d_e);
    cuda::free(d_dedt);
    cuda::free(d_p);
    cuda::free(d_cs);

    cuda::free(d_sml);

//#if INTEGRATE_DENSITY
    cuda::free(d_drhodt);
//#endif

#if VARIABLE_SML || INTEGRATE_SML
    cuda::free(d_dsmldt);
#endif

    cuda::free(d_integratedParticles);

}


