# Miluphcuda NNS

* `rhs.cu::rightHandSide()`
	* `tree.cu::buildTree()` (on GPU)
	* `tree.cu::getTreeDepth()` (on GPU)
	* `tree.cu::setEmptyMassForInnerNodes()` (on GPU)
	* `density.cu::calculateDensity()` (on GPU)
	* `soundspeed.cu::calculateSoundSpeed()` (on GPU)
	* `pressure.cu::calculatePressure()` (on GPU)
	* ...

	
## tree.cu

### tree.h

* `__global__ void buildTree()`
* `__global__ void getTreeDepth(int *treeDepthPerBlock)`
* `__global__ void measureTreeChange(int *movingparticlesPerBlock)`
* `__global__ void calculateCentersOfMass()`
* `__global__ void setEmptyMassForInnerNodes(void)`
* `__global__ void nearNeighbourSearch(int *interactions)`
* `__global__ void nearNeighbourSearch_modify_sml(int *interactions)`
* `__global__ void knnNeighbourSearch(int *interactions)`
* `__global__ void symmetrizeInteractions(int *interactions)`
* `__global__ void check_sml_boundary(void)`
* `__device__ void redo_NeighbourSearch(int particle_id, int *interactions)`
* `__global__ void computationalDomain(double *minxPerBlock, double *maxxPerBlock, double *minzPerBlock, double *maxzPerBlock)`

