/**
 * @file material_handler.h
 * @brief Handler for material parameters and settings.
 *
 * Handler for material parameters/attributes/properties and settings like:
 *
 * * Equation of state
 * * Artificial viscosity (parameters)
 * * smoothing length
 * * interactions
 *
 * @author Michael Staneker
 * @bug no known bugs
 * @todo implement missing parameters/variables
 */
#ifndef MILUPHPC_MATERIAL_HANDLER_H
#define MILUPHPC_MATERIAL_HANDLER_H

#include "material.cuh"
#include "../cuda_utils/cuda_runtime.h"
#include "../parameter.h"
#include "../utils/logger.h"

#include <fstream>
#include <libconfig.h>

/**
 * @brief Read material config files.
 */
class LibConfigReader {
public:
    config_t config;
    config_setting_t *materials;
    /**
     * Load/read config file.
     *
     * @param configFile provided config file/path
     * @return number of materials within provided config file
     */
    int loadConfigFromFile(const char *configFile);
};

/**
 * @brief Material class handler.
 *
 * * handling host and device instances
 * * initializing values using `LibConfigReader`
 * * copying instances/values between MPI processes and/or device and host
 */
class MaterialHandler {

public:
    /// number of materials or rather material instances
    integer numMaterials;
    /// host instance of material class
    Material *h_materials;
    /// device instance of material class
    Material *d_materials;

    /**
     * @brief Constructor.
     *
     * @param numMaterials
     */
    MaterialHandler(integer numMaterials);

    /**
     * @brief Constructor from config file.
     *
     * @param material_cfg Config file name/path
     */
    MaterialHandler(const char *material_cfg);

    /**
     * @brief Constructor.
     *
     * @param numMaterials
     * @param ID
     * @param interactions
     * @param alpha
     * @param beta
     */
    MaterialHandler(integer numMaterials, integer ID, integer interactions, real alpha, real beta);

    /**
     * @brief Destructor.
     */
    ~MaterialHandler();

    /**
     * Copy material instance(s) from host to device or vice-versa.
     *
     * @param target target: host or device
     * @param index material instance index to be copied, if `-1` copy all instances
     */
    void copy(To::Target target, integer index = -1);

    /**
     * Communicate material instances between MPI processes and in addition
     * from and/or to the device(s).
     *
     * @warning it is not possible to send it from device to device via CUDA-aware MPI,
     * since serialize functionality not usable on device
     *
     * @param from MPI process source
     * @param to MPI process target
     * @param fromDevice flag whether start from device
     * @param toDevice flag whether start from device
     */
    void communicate(int from, int to, bool fromDevice = false, bool toDevice = true);

    /**
     * Broadcast material instances to all MPI processes from a root
     *
     * @param root root to broadcast from (default: MPI process 0)
     * @param fromDevice flag whether start from device
     * @param toDevice flag whether start from device
     */
    void broadcast(int root = 0, bool fromDevice = false, bool toDevice = true);

};

#endif //MILUPHPC_MATERIAL_HANDLER_H
