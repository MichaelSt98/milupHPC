# Config

* for compile time settings refer to [README](../README.md) or rather [include/parameter.h](../include/parameter.h)

## Config file/Runtime settings

* see e.g. [config.info](config.info)
	* or [plummer_config.info](plummer_config.info) for the Plummer test case
	* or [sedov_config.info](sedov_config.info) for the Sedov test case
* based on the boost property tree: [INFO format](https://www.boost.org/doc/libs/1_46_1/doc/html/boost_propertytree/parsers.html#boost_propertytree.parsers.info_parser)

basically looking like this

```
key1 value1
key2
{
   key3 value3
   {
      key4 "value4 with spaces"
   }
   key5 value5
}
```

<details>
 <summary>
   Config file for milupHPC
 </summary>
 
```
; IO RELATED
; ------------------------------------------------------
; output directory (will be created if it does not exist)
directory bb/

; outputRank (-1 corresponds to all)
outputRank -1
; omit logType::TIME for standard output
omitTime true
; create log file (including warnings, errors, ...)
log true
; create performance log
performanceLog true
; write particles to be sent to h5 file
particlesSent2H5 true


; INTEGRATOR RELATED
; ------------------------------------------------------
; integrator selection
; explicit euler [0], predictor-corrector euler [1], leapfrog [2]
integrator 1
; initial time step
timeStep 1e-4
; max time step allowed
maxTimeStep 1e-4
; end time for simulation
;timeEnd 6e-2

; SIMULATION RELATED
; ------------------------------------------------------
; space-filling curve selection
; lebesgue [0], hilbert [1]
sfc 1

; theta-criterion for Barnes-Hut (approximative gravity)
theta 0.5
; smoothing parameter for gravitational forces
smoothing 2.56e+20

; SPH smoothing kernel selection
; spiky [0], cubic spline [1], wendlandc2 [3], wendlandc4 [4], wendlandc6 [5]
smoothingKernel 1

; remove particles (corresponding to some criterion)
removeParticles true
; spherically [0], cubic [1]
removeParticlesCriterion 0
; allowed distance to center (0, 0, 0)
removeParticlesDimension 3.6e14

; execute load balancing
loadBalancing false
; interval for executing load balancing (every Nth step)
loadBalancingInterval 1

; how much memory to allocate (1.0 -> all particles can in principle be on one process)
particleMemoryContingent 1.0

; calculate angular momentum (and save to output file)
calculateAngularMomentum true
; calculate (total) energy (and save to output file)
calculateEnergy true
; calculate center of mass (and save to output file)
calculateCenterOfMass false

; IMPLEMENTATION SELECTION
; ------------------------------------------------------
; force version for gravity (use [2])
; burtscher [0], burtscher without presorting [1], miluphcuda with presorting [2],
; miluphcuda without presorting [3], miluphcuda shared memory (experimental) [4]
gravityForceVersion 0
; fixed radius NN version for SPH (use [0])
; normal [0], brute-force [1], shared-memory [2], within-box [3]
sphFixedRadiusNNVersion 3
```
</details>

## Material config file

* see e.g. [material.cfg](material.cfg)
	* or [sedov_material.cfg](sedov_material.cfg) for the Sedov test case
* based on [libconfig](http://hyperrealm.github.io/libconfig/) including the [documentation](http://hyperrealm.github.io/libconfig/libconfig_manual.html)

<details>
 <summary>
   Material config file for milupHPC
 </summary>
 
```
materials = (
{
    ID = 0
    name = "IsothermalGas"
    #sml = 1e12
    sml = 5.2e11
    interactions = 50
    artificial_viscosity = { alpha = 1.0; beta = 2.0; };
    eos = {
        type = 3
    };
}
);

...
```
 
</details>

