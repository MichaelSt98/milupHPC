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

* `barnesHut()`, `sph()` -> `rhs()`
	* `integrator/*.cuh/cu` -> `integrator/*.h/cpp` ? 
* remove pseudo-particle calculations within `buildTree()` for SPH **without** gravity
	* compiler directive `#GRAVITY`!? 
* remove `Helper` instance from `compTheta()`
* *device\_rhs.cuh/cu* needed? move to *helper.cuh/cu*

### Ideas 

* `Miluphpc` as `BaseIntegrator`

```cpp

namespace miluph {
	// miluphpc.h/.cpp
	class Miluphpc {
	public:
		ParticleHandler *particleHandler;
    		SubDomainKeyTreeHandler *subDomainKeyTreeHandler;
    		TreeHandler *treeHandler;
    		DomainListHandler *domainListHandler;
    		DomainListHandler *lowestDomainListHandler;
    		// ...
		IntegratedParticles *integratedParticles();
		Miluphpc() {
			...
		}
		virtual void integrate() {};
		void rhs() {
			//(parallel) tree construction
			//(parallel) pseudo-particles
			//(parallel) gravitational force
			//(parallel) SPH forces
			...
		}
	} 
	
	// euler.h/.cpp
	class Euler : public Miluphpc {
	public:
		Euler() {
			integratedParticles = new IntegratedParticles[1];
		}
		void integrate() {
			rhs();
			// save information in integratedParticles
			rhs();
			...
		}
	}
	
	// ...
}
// main.cpp
miluph::Miluphpc *miluphpc;

switch(integratorSelection) {
    case IntegratorSelection::euler: {
        miluphpc = new miluph::Euler();
    } break;
    case IntegratorSelection::predictor_corrector: {
        miluphpc = new miluph::PredictorCorrector();
    } break;
    default: { }
}

while (condition) {
	miluphpc.integrate();
}

```

### Problems, Challenges, Improvements, ...

* smoothing in `gravity::computeForces()`
	* how to avoid $r = 0$
	* how to avoid $r \approx 0$ 
* `symbolicForce`: insert cell itself or children
* **?bug?** removing duplicates for `symbolicForce()`
	* not sufficient to remove duplicate indices: Why?
	* leading to seg fault in `insertReceivedParticles()`
* **?bug?** (someties) duplicates in SPH send entries
	* leading to seg fault in `insertReceivedParticles()`
* Hilbert keys: `getTreeLevel()`
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

| id   | description                                       | Function                         | type     | size                | Location | Intersection |
| ---- | ------------------------------------------------- | -------------------------------- | -------- | ------------------- | -------- | ------------ |
| 1    | mutex/lock                                        | `computeBoundingBox()`           | int      | 1                   | device   | -            |
| 2    | mark particles with correspondent process         | `markParticlesProcess()`         | int      | `numParticlesLocal` | device   | 3            |
| 3    | sorting particle array and copying back           | `sortArray()` & `copyArray()`    | float    | `numParticlesLocal` | device   | 2            |
| 4    | send lengths                                      | -                                | int      | `numProcesses`      | host     | 5            |
| 5    | receive lengths                                   | -                                | int      | `numProcesses`      | host     | 4            |
| 6    | sorting lowest domain list array and copying back | `sortArray()` & `copyArray()`    | float    | domain list length  | device   | -            |
| 7    | buffer send indices for parallel grav. force      | `symbolicForce()`                | int      | (up to) `numPLocal` | device   | 8            |
| 8    | buffer removing duplicates in sent indices        | `removeDuplicates()`             | int      | (up to) `numPLocal` | device   | 7            |
| 9    | collecting values to be (temporarily) sent        | `collectValues()`                | float    | (up to) `numPLocal` | device   | 8            |
| 10   | sph: already inserted                             | `particles2Send()`               | int      | `numProcesses`      | device   | 11, 12       |
| 11   | sph: send count                                   | `particles2Send()`               | int      | `numProcesses`      | device   | 10, 12       |
| 12   | sph: send indices for parallel sph force(s)       | `particles2Send()`               | int      | (up to) `numPLocal` | device   | 10, 11       |
| 13   | sph: collecting values to be (temp.) sent         | `collectSendIndices()`           | int      | (up to) `numPLocal` | device   | 12           |
| 14   | sph: collecting send entries                      | `collectSendEntries()`, `exchangeParticleEntry()` | float    | (up to) `numPLocal` | device   | 13 (or 12)   |
| 15   | key histogram y-axis                              | `createKeyHistRanges()`, `keyHistCounter()`, `calculateNewRange()` | int      | bins (up to 10000)  | device   | 16 |
| 16   | key histogram x-axis                              | `keyHistCounter()`, `calculateNewRange()`                          | key type | bins (up to 10000)  | device   | 15 |
| 17   | buffers for file I/O                              | `particles2file()`               | -        | `numParticlesLocal`  | host (device for keys) |  |

* is `13` really necessary? Or is it possible to remove by changing `SPH::Kernel::Launch::collectSendEntrie()`?
      


                       
 