#ifndef MILUPHPC_MATERIAL_HANDLER_H
#define MILUPHPC_MATERIAL_HANDLER_H

#include "material.cuh"
#include "../cuda_utils/cuda_runtime.h"
#include "../parameter.h"

#include <fstream>
#include <libconfig.h>

class LibConfigReader {
public:
    config_t config;
    config_setting_t *materials;
    int loadConfigFromFile(const char *configFile);
};

class MaterialHandler {

public:
    integer numMaterials;

    Material *h_materials;
    Material *d_materials;

    MaterialHandler(integer numMaterials);
    MaterialHandler(const char *material_cfg);
    MaterialHandler(integer numMaterials, integer ID, integer interactions, real alpha, real beta);
    ~MaterialHandler();

    void copy(To::Target target, integer index = -1);

    // COMMUNICATING MATERIAL INSTANCES BETWEEN MPI PROCESSES
    // ATTENTION: it is not possible to send it from device to device
    //  since serialize functionality not usable on device
    void communicate(int from, int to, bool fromDevice = false, bool toDevice = true);
    void broadcast(int root = 0, bool fromDevice = false, bool toDevice = true);

};

#endif //MILUPHPC_MATERIAL_HANDLER_H
