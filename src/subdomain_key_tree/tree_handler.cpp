#include "../../include/subdomain_key_tree/tree_handler.h"

TreeHandler::TreeHandler(integer numParticles, integer numNodes) : numParticles(numParticles),
                                                        numNodes(numNodes) {

    cuda::malloc(d_count, numNodes);
    cuda::malloc(d_start, numNodes);
    cuda::malloc(d_sorted, numNodes);
    cuda::malloc(d_child, POW_DIM * numNodes);
    cuda::malloc(d_index, 1);
    h_index = new integer;
    cuda::malloc(d_mutex, 1);

    cuda::malloc(d_minX, 1);
    cuda::malloc(d_maxX, 1);
    h_minX = new real;
    h_maxX = new real;
#if DIM > 1
    cuda::malloc(d_minY, 1);
    cuda::malloc(d_maxY, 1);
    h_minY = new real;
    h_maxY = new real;
#if DIM == 3
    cuda::malloc(d_minZ, 1);
    cuda::malloc(d_maxZ, 1);
    h_minZ = new real;
    h_maxZ = new real;
#endif
#endif

    h_toDeleteLeaf = new integer[2];
    h_toDeleteNode = new integer[2];
    cuda::malloc(d_toDeleteLeaf, 2);
    cuda::malloc(d_toDeleteNode, 2);

    cuda::malloc(d_tree, 1);

#if DIM == 1
    TreeNS::Kernel::Launch::set(d_tree, d_count, d_start, d_child, d_sorted, d_index, d_toDeleteLeaf, d_toDeleteNode,
                                d_minX, d_maxX);
#elif DIM == 2
    TreeNS::Kernel::Launch::set(d_tree, d_count, d_start, d_child, d_sorted, d_index, d_toDeleteLeaf, d_toDeleteNode,
                               d_minX, d_maxX, d_minY, d_maxY);
#else
    TreeNS::Kernel::Launch::set(d_tree, d_count, d_start, d_child, d_sorted, d_index, d_toDeleteLeaf, d_toDeleteNode,
                                d_minX, d_maxX, d_minY, d_maxY, d_minZ, d_maxZ);
#endif

}

TreeHandler::~TreeHandler() {

    cuda::free(d_count);
    cuda::free(d_start);
    cuda::free(d_sorted);
    cuda::free(d_child);
    cuda::free(d_index);
    cuda::free(d_mutex);

    cuda::free(d_minX);
    cuda::free(d_maxX);
    delete h_minX;
    delete h_maxX;
#if DIM > 1
    cuda::free(d_minY);
    cuda::free(d_maxY);
    delete h_minY;
    delete h_maxY;
#if DIM == 3
    cuda::free(d_minZ);
    cuda::free(d_maxZ);
    delete h_minZ;
    delete h_maxZ;
#endif
#endif

    delete [] h_toDeleteLeaf;
    delete [] h_toDeleteNode;
    cuda::free(d_toDeleteLeaf);
    cuda::free(d_toDeleteNode);

    cuda::free(d_tree);

}

void TreeHandler::copy(To::Target target, bool borders, bool index, bool toDelete) {

    if (borders) {
        cuda::copy(h_minX, d_minX, 1, target);
        cuda::copy(h_maxX, d_maxX, 1, target);
#if DIM > 1
        cuda::copy(h_minY, d_minY, 1, target);
        cuda::copy(h_maxY, d_maxY, 1, target);
#if DIM == 3
        cuda::copy(h_minZ, d_minZ, 1, target);
        cuda::copy(h_maxZ, d_maxZ, 1, target);
#endif
#endif
    }
    if (index) {
        cuda::copy(h_index, d_index, 1, target);
    }
    if (toDelete) {
        cuda::copy(h_toDeleteLeaf, d_toDeleteLeaf, 2, target);
        cuda::copy(h_toDeleteNode, d_toDeleteNode, 2, target);
    }

}

void TreeHandler::globalizeBoundingBox(Execution::Location exLoc) {

    boost::mpi::communicator comm;

    switch (exLoc) {
        case Execution::device:
            all_reduce(comm, boost::mpi::inplace_t<real*>(d_minX), 1, boost::mpi::minimum<real>());
            all_reduce(comm, boost::mpi::inplace_t<real*>(d_maxX), 1, boost::mpi::maximum<real>());
#if DIM > 1
            all_reduce(comm, boost::mpi::inplace_t<real*>(d_minY), 1, boost::mpi::minimum<real>());
            all_reduce(comm, boost::mpi::inplace_t<real*>(d_maxY), 1, boost::mpi::maximum<real>());
#if DIM == 3
            all_reduce(comm, boost::mpi::inplace_t<real*>(d_minZ), 1, boost::mpi::minimum<real>());
            all_reduce(comm, boost::mpi::inplace_t<real*>(d_maxZ), 1, boost::mpi::maximum<real>());
#endif
#endif
            break;
        case Execution::host:
            all_reduce(comm, boost::mpi::inplace_t<real*>(h_minX), 1, boost::mpi::minimum<real>());
            all_reduce(comm, boost::mpi::inplace_t<real*>(h_maxX), 1, boost::mpi::maximum<real>());
#if DIM > 1
            all_reduce(comm, boost::mpi::inplace_t<real*>(h_minY), 1, boost::mpi::minimum<real>());
            all_reduce(comm, boost::mpi::inplace_t<real*>(h_maxY), 1, boost::mpi::maximum<real>());
#if DIM == 3
            all_reduce(comm, boost::mpi::inplace_t<real*>(h_minZ), 1, boost::mpi::minimum<real>());
            all_reduce(comm, boost::mpi::inplace_t<real*>(h_maxZ), 1, boost::mpi::maximum<real>());
#endif
#endif
            break;
        default:
            Logger(ERROR) << "not available!";
    }

}

