//
// Created by Michael Staneker on 14.08.21.
//

#ifndef MILUPHPC_TREEHANDLER_H
#define MILUPHPC_TREEHANDLER_H

#include "tree.cuh"
#include "../parameter.h"

class TreeHandler {

public:
    integer *d_count;
    integer *d_start;
    integer *d_sorted;
    integer *d_child;
    integer *d_index;
    integer *d_mutex;

    real *d_minX, *d_maxX;
    real *h_minX, *h_maxX;
#if DIM > 1
    real *d_minY, *d_maxY;
    real *h_minY, *h_maxY;
#if DIM == 3
    real *d_minZ, *d_maxZ;
    real *h_minZ, *h_maxZ;
#endif
#endif

    Tree *d_tree;

    TreeHandler(integer numParticles, integer numNodes);
    ~TreeHandler();

};


#endif //MILUPHPC_TREEHANDLER_H
