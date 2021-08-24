#ifndef MILUPHPC_MATERIAL_HANDLER_H
#define MILUPHPC_MATERIAL_HANDLER_H

#include "material.cuh"
#include "../parameter.h"
//#include "../cuda_utils/cuda_utlities.cuh"
//#include "../cuda_utils/cuda_launcher.cuh"

//TODO: initialize classes using config file(s)
// use libconfig? (a C++ API is available!)
class MaterialHandler {

public:
    integer numMaterials;

    Material *h_materials;
    Material *d_materials;

    MaterialHandler(integer numMaterials);
    MaterialHandler(integer numMaterials, integer ID, integer interactions, real alpha, real beta);
    ~MaterialHandler();

    void toDevice(integer index = -1);
    void toHost(integer index = -1);

    // COMMUNICATING MATERIAL INSTANCES BETWEEN MPI PROCESSES
    // ATTENTION: it is not possible to send it from device to device
    //  since serialize functionality not usable on device
    void communicate(int from, int to, bool fromDevice = false, bool toDevice = true);
    void broadcast(int root = 0, bool fromDevice = false, bool toDevice = true);

};

#endif //MILUPHPC_MATERIAL_HANDLER_H
