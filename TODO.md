# TODO

## 2021-10-29

* make preprocessor directives to constants 
	* e.g. `theta`
* `nnl` not as function parameter, but as attribute of particle struct 
* Integrator
	* decouple sml integration (see Integrator problems...) 
* SI-units
	* change initial conditions 
	* within Gravity 
	* ...?
* introduce `GRAVITY_SIM` and `SPH_SIM` in order to optimize
	* memory usage
	* performance
	* compilation 
		* `#find ./src -type f -name "*.cu" -not -path "./src/gravity/*"`
		* `#find ./src -type f -name "*.cu" -not -path "*/gravity/*"`
		* `#find . -type d \( -path ./src/sph -o -path ./src/gravity -o -path ./dir3 \) -prune -o -name '*.cu' -print`
		* `#find . -type d \( -name sph -o -name gravity -o -name dir3 \) -prune -o -name '*.cu' -print` 
* **[FIXED]** code not running (anymore) for `DIM = 2` (at *build domain tree*...)
* restructure *gravity/*
	* for parallel and serial functions 

	
_____

# TODO

_________

## 2021-11-31

* adaptive time-step for `explicit_euler` with (switch?)
	* which criterions?
	* Move `device_predictor_corrector_euler::Shared` and `BlockShared` to *device_rhs.cu* ?!
* energy and angular momentum calculation
* continue `SAFETY_LEVEL` using `cudaAssert(...)` and `cudaTerminate(...)`
* test Boss-Bodenheimer again
* Kepler: donut shape (for 3D) ?!
* Master thesis ...

## 2021-11-30

### Now

* adaptive time-step for `explicit_euler` (?!)
	* how to accomplish this for gravity only simulation?! 
* command line/config file arguments
	* **[FINISHED]** bins for dynamic load balancing
	* **[FINISHED]** load balancing in config file? (+ load balancing interval)
	* ...
* energy/angular momentum calculation (**[FINISHED]** with flag)
* center of mass calculation (**[FINISHED]** with flag)
* **[FINISHED]** reduce amount of needed memory: `numParticles = (int)(m.size() * 0.7);` (with flag)
	* `numNodes` as function for `numParticles`
* **[FINISHED]** Introduce compiler directive `SAFETY_LEVEL`
	* `SAFETY_LEVEL 0`: almost no safety measures
	* `SAFETY_LEVEL 1`: most relevant/important safety measures
	* `SAFETY_LEVEL 2`: more safety measures, including assertions
	* `SAFETY_LEVEL 3`: many security measures, including all assertions
	* `cudaAssert(...)`: printing warning and/or terminating in dependence of `SAFETY_LEVEL`
	* `cudaTerminate(...)`: terminating from within CUDA kernel
* **[FINISHED]**  Introduce verbosity via variable
	* `verbosity = 0`: almost no verbosity
	* `verbosity = 1`: verbosity regarding control flow and most important information (like number of particles)
	* `verbosity = 2`: debug information

> Special flag for omitting TIME output


### Potentially 

* small encounter handling via smoothing length
* Some kind of key class to enable more tree levels
* `nnl` not as function parameter, but as attribute of particle struct 

### Later

* **particle entries in dependence of**
	* `GRAVITY_SIM`
	* `SPH_SIM` 
* use *libconfig.hpp* instead of *libconfig.h*
* compilation optimization
	* Gravity and SPH
	* Allow compilation with `g++`, thus without MPI 
* revise buffer handling
* serial functionality
* writing summary file
* documentation
* application to define compiler directives/generate `parameter.h`
* restart simulation
* read particle file from root and broadcast

### Postprocessing

* performance analytics
* ...

_________

# Notes

## Commands

* introduce `GRAVITY_SIM` and `SPH_SIM` in order to optimize
	* memory usage
	* performance
	* compilation 
		* `#find ./src -type f -name "*.cu" -not -path "./src/gravity/*"`
		* `#find ./src -type f -name "*.cu" -not -path "*/gravity/*"`
		* `#find . -type d \( -path ./src/sph -o -path ./src/gravity -o -path ./dir3 \) -prune -o -name '*.cu' -print`
		* `#find . -type d \( -name sph -o -name gravity -o -name dir3 \) -prune -o -name '*.cu' -print` 
