#ifndef MILUPHPC_TREEHANDLER_H
#define MILUPHPC_TREEHANDLER_H

#include "tree.cuh"
#include "../parameter.h"
#include "../utils/logger.h"
#include "../cuda_utils/cuda_runtime.h"
#include <boost/mpi.hpp>

/**
 * Class to handle device (and potentially host) instance of Tree class.
 */
class TreeHandler {

public:

    /// number of particles
    integer numParticles;
    /// number of nodes
    integer numNodes;

    /// device (pointer to) count (array)
    integer *d_count;
    /// device (pointer to) start (array)
    integer *d_start;
    /// device (pointer to) sorted (array)
    integer *d_sorted;
    /// device (pointer to) child/children (array)
    integer *d_child;
    /// device (pointer to) index
    integer *d_index;
    /// host (pointer to) index
    integer *h_index;
    /// device (pointer to) mutex/lock
    integer *d_mutex;

    /// host (pointer to) array remembering leaf indices for rebuilding after temporarily inserting particles
    integer *h_toDeleteLeaf;
    /// host (pointer to) array remembering leaf indices for rebuilding after temporarily inserting particles
    integer *h_toDeleteNode;
    /// device (pointer to) array remembering leaf indices for rebuilding after temporarily inserting particles
    integer *d_toDeleteLeaf;
    /// device (pointer to) array remembering leaf indices for rebuilding after temporarily inserting particles
    integer *d_toDeleteNode;

    /// device (pointer to) bounding box minimal x
    real *d_minX;
    /// device (pointer to) bounding box maximal x
    real *d_maxX;
    /// host (pointer to) bounding box minimal x
    real *h_minX;
    /// host (pointer to) bounding box maximal x
    real *h_maxX;
#if DIM > 1
    /// device (pointer to) bounding box minimal y
    real *d_minY;
    /// device (pointer to) bounding box maximal y
    real *d_maxY;
    /// host (pointer to) bounding box minimal y
    real *h_minY;
    /// host (pointer to) bounding box maximal y
    real *h_maxY;
#if DIM == 3
    /// device (pointer to) bounding box minimal z
    real *d_minZ;
    /// device (pointer to) bounding box maximal z
    real *d_maxZ;
    /// host (pointer to) bounding box minimal z
    real *h_minZ;
    /// host (pointer to) bounding box maximal x
    real *h_maxZ;
#endif
#endif

    /// device instance of Class `Tree`
    Tree *d_tree;

    /**
     * Constructor
     *
     * @param numParticles number of particles
     * @param numNodes number of nodes
     */
    TreeHandler(integer numParticles, integer numNodes);

    /**
     * Destructor
     */
    ~TreeHandler();

    /**
     * all reduce bounding box(es)/borders (among MPI processes)
     *
     * @param exLoc execute on device or host
     */
    void globalizeBoundingBox(Execution::Location exLoc=Execution::device);

    /**
     * Copy (parts of the) tree instance(s) between host and device
     *
     * @param target copy to target
     * @param borders flag whether borders should be copied
     * @param index flag whether `index` should be copied
     * @param toDelete flag whether `toDeleteLeaf` and `toDeleteNode`
     */
    void copy(To::Target target = To::device, bool borders = true, bool index = true, bool toDelete = true);

};


#endif //MILUPHPC_TREEHANDLER_H
