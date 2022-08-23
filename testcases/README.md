# Test cases

* [Plummer test case](plummer/) as gravity-only test case ([README](plummer/README.md))
* [Taylor-von Neumann-Sedov blast wave test case](sedov/) as SPH-only test case ([README](sedov/README.md))

<details>
 <summary>
   Plummer
 </summary>

* refer to [this testcase](plummer/) ([README](plummer/README.md))
* Plummer model: four GPUs with dynamic load balancing every 10th step (top: lebesgue, bottom: hilbert)

<img src="../documents/4proc_plummer_dynamic.gif" alt="Plummer sample" width="100%"/>

The Plummer model is a gravity only test case, the distribution is stable over time enabling the validation as shown in the following picture.

<img src="../documents/figures/long_pl_N10000000_sfc1D_np4_mass_quantiles.png" alt="Plummer sample" width="100%"/>

</details>


<details>
 <summary>
   Taylor–von Neumann–Sedov blast wave
 </summary>

* refer to [this testcase](sedov/) ([README](sedov/README.md))
* Sedov explosion: one and two GPUs

<img src="../documents/sedov_sample_movie.gif" alt="Sedov sample" width="100%"/>

The density in dependence of the radius for t = 0.06 and the semi-analytical solution as comparison.

<img src="../documents/figures/sedov_N171_sfc1D_np8_density.png" alt="Sedov sample" width="100%"/>

</details>
