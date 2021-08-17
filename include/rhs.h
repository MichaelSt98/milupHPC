#ifndef MILUPHPC_RHS_H
#define MILUPHPC_RHS_H

#include "../include/constants.h"
#include "../include/utils/logger.h"
#include "../include/utils/timer.h"
#include "../include/utils/cxxopts.h"
#include "../include/utils/config_parser.h"
#include "../include/cuda_utils/cuda_utilities.cuh"
//#include "../include/cuda_utils/cuda_launcher.cuh"

#include "../include/subdomain_key_tree/tree.cuh"
#include "../include/subdomain_key_tree/sample.cuh"
#include "../include/particles.cuh"
#include "../include/particle_handler.h"
#include "../include/memory_handling.h"
#include "../include/device_rhs.cuh"
#include "../include/subdomain_key_tree/subdomain_handler.h"

void rhs();

#endif //MILUPHPC_RHS_H
