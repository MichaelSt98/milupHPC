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