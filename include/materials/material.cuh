#ifndef MILUPHPC_MATERIAL_CUH
#define MILUPHPC_MATERIAL_CUH

#include "../cuda_utils/cuda_utilities.cuh"
#include "../parameter.h"

#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <boost/mpi.hpp>

//TODO: implement missing parameters/variables

/**
 * Artificial viscosity parameters
 */
struct ArtificialViscosity {

    // enable communication via MPI (send instance of struct directly)
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & alpha;
        ar & beta;
    }

    real alpha;
    real beta;

    CUDA_CALLABLE_MEMBER ArtificialViscosity();
    CUDA_CALLABLE_MEMBER ArtificialViscosity(real alpha, real beta);
};

struct EqOfSt {

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & type;
        ar & polytropic_K;
        ar & polytropic_gamma;
    }

    int type;
    real polytropic_K;
    real polytropic_gamma;

    CUDA_CALLABLE_MEMBER EqOfSt();
    CUDA_CALLABLE_MEMBER EqOfSt(int type, real polytropic_K, real polytropic_gamma);

};

/**
 * Material parameters
 */
class Material {

public:

    // enable communication via MPI (send instance of class directly)
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & ID;
        ar & interactions;
        ar & artificialViscosity;
        ar & eos;
    }

    integer ID;
    integer interactions;

    ArtificialViscosity artificialViscosity;
    EqOfSt eos;

    CUDA_CALLABLE_MEMBER Material();
    CUDA_CALLABLE_MEMBER ~Material();

    CUDA_CALLABLE_MEMBER void info();

    /*real density_floor;

    struct eos {
        integer type;
        real shear_modulus;
        real bulk_modulus;
        real yield_stress;

        real cs_limit;

        // ideal gas
        real polytropic_gamma;
        real ideal_gas_rho_0;
        real ideal_gas_p_0;
        real ideal_gas_conv_e_to_T;

        // Tillotson
        real till_rho_0;
        real till_A;
        real till_B;
        real till_E_0;
        real till_E_iv;
        real _E_cv;
        real till_a;
        real till_b;
        real till_alpha;
        real till_beta;
        real rho_limit;

        // ANEOS
        char *table_path;
        integer n_rho;
        integer n_e;
        real aneos_rho_0;
        real aneos_bulk_cs;
        real aneos_e_norm;

        // plasticity
        real cohesion;
        real friction_angle;
        real cohesion_damaged;
        real friction_angle_damaged;
        real melt_energy;


        // fragmentation
        real weibull_k;
        real weibull_m;

    };

    struct porosity {
        real porjutzi_p_elastic;
        real porjutzi_p_transition;
        real porjutzi_p_compacted;
        real porjutzi_alpha_0;
        real porjutzi_alpha_e;
        real porjutzi_alpha_t;
        real porjutzi_n1;
        real porjutzi_n2;
        real cs_porous;
        integer crushcurve_style;
    };

    struct plasticity {
        real yield_stress;
        real cohesion;
        real friction_angle;
        real friction_angle_damaged;
        // ...
    };*/
};


namespace MaterialNS {

    namespace Kernel {

        __global__ void info(Material *material);

        namespace Launch {
            void info(Material *material);
        }
    }

}

#endif //MILUPHPC_MATERIAL_CUH
