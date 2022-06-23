# Notes

## N-Body & SPH

* N-Body: 
	* (parallel) tree construction
		* bounding box
		* domain list
		* tree
		* common coarse tree
	* (parallel) pseudo-particles
	* (parallel) gravitational force
* SPH **with** N-Body:
	* (parallel) tree construction
		* bounding box
		* domain list
		* tree
		* common coarse tree
	* (parallel) pseudo-particles
	* (parallel) gravitational force
	* (parallel) SPH forces
* SPH **without** N-Body:
	* (parallel) tree construction
		* bounding box
		* domain list
		* tree
		* common coarse tree
	* (parallel) SPH forces

**Thus** 4 *modules*:

* (parallel) tree construction
* (parallel) pseudo-particles
* (parallel) gravitational force
* (parallel) SPH forces


## TODO

* compile time
* compiler directive constants as varialbles!?
* overloaded `atomicAdd()` for double input
	* [atomicAdd with doubles](https://forums.developer.nvidia.com/t/why-does-atomicadd-not-work-with-doubles-as-input/56429) 	
* remove pseudo-particle calculations within `buildTree()` for SPH **without** gravity
	* compiler directive `#GRAVITY`!? 
* remove `Helper` instance from `compTheta()`
* *device\_rhs.cuh/cu* needed? move to *helper.cuh/cu*

### Ideas 

* ...

### Problems, Challenges, Improvements, ...

* smoothing in `gravity::computeForces()`
	* how to avoid $r = 0$
	* how to avoid $r \approx 0$ 
* `symbolicForce`: insert cell itself or children
* 1D Space filling curve
* Performance:
	* kernel call `buildDomainTree()` with more than one thread
	* kernel call `createDomainList()` with more than one thread
* Structure
	* how to structure `main` function/functionality?
	* generalize functionality
	* naming (conventions)
	* where to put which function


## Generalizing functions/functionality

* particle exchange process for N-Body and SPH
* ...

## Buffers

| **id** |**description** |**functions** |**type** |**size** |**location** |**intersection** |
|--- | --- | --- | --- | --- | --- | --- | 
|1 | mutex/lock | `computeBoundingBox` | int | 1 | device |  | 
 |  |  |  |  |  |  | 
2 | arranging particle (entries) particles process | `arrangeParticleEntries` | int  | numParticlesLocal | device |  | 
3 | arranging particle (entries) particles process sorted | `arrangeParticleEntries` | int  | numParticlesLocal | device |  | 
4 | arranging particle (entries) sorted entries | `arrangeParticleEntries` | float | numParticlesLocal | device |  | 
5 | arranging particle (entries) sorted entries | `arrangeParticleEntries` | int | numParticlesLocal | device |  | 
 |  |  |  |  |  |  | 
6 | sorting domain list (entries) unsorted entries | `sortArray()` within `parallel_pseudoParticles()` | float | domainListSize | device |  | 
7 | sorting domain list (entries) sorted entries | `sortArray()` within `parallel_pseudoParticles()` | float | domainListSize | device |  | 
 |  |  |  |  |  |  | 
8 | symbolic force: mark send indices | *Gravity::* `intermediateSymbolicForce()`, `symbolicForce()`, `collectSendIndices()` | int | numNodes | device |  | 
9 | symbolic force: collected collected particle (entries) | `collectValues()`, `sendParticles()` | float | up to 50% of numNodes | device |  | 
10 | **???** symbolic force: collected particle (entries) | `collectValues()`, `sendParticles()` | int | up to 50% of numNodes | device |  | 
11 | symbolic force: send indices (particles) | `collectSendIndices()`, `sendParticles()` | int | up to 50% of numParticles | device |  | 
12 | symbolic force: send indices (pseudo-particles) | `collectSendIndices()`, `sendParticles()` | int | up to 50% of numNodes | device |  | 
13 | symbolic force: send levels (pseudo-particles) | `collectSendIndices()`, `sendParticles()` | int | up to 50% of numNodes | device |  | 
14 | symbolic force: receive levels (pseudo-particles) | `collectSendIndices()`, `sendParticles()` | int | up to 50% of numNodes | device |  | 
15 | symbolic force: send count (particles) | `collectSendIndices()` | int | numProcesses | device |  | 
16 | symbolic force: send count (pseudo-particles) | `collectSendIndices()` | int | numProcesses | device |  | 
 |  |  |  |  |  |  | 
17 | sph symbolic force: mark send indices | *SPH::* `intermediateSymbolicForce()`, `symbolicForce()`, `collectSendIndices()` | int | numParticles | device |  | 
18 | sph symbolic force: send indices | `collectSendIndices()`, `sendParticles()` | int | up to 50% of numParticles | device |  | 
19 | sph symbolic force: collected particle (entries) | `collectValues()`, `sendParticles()` | float | up to 50% of numParticles | device |  | 
20 | sph symbolic force: collected particle (entries) | `collectValues()`, `sendParticles()` | int | up to 50% of numParticles | device |  | 
21 | sph symbolic force: send count (particles) | `collectSendIndices()` | int | numProcesses | device |  | 
22 | sph symbolic force: send count (pseudo-particles) | `collectSendIndices()` | int | numProcesses | device |  | 
 |  |  |  |  |  |  | 
23 | file I/O | `particles2file()` | keyType/unsigned long | numParticlesLocal | device |  | 
      


                       
 