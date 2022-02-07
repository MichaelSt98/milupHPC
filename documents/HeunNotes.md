
```cpp
class ParticleHandler {

real *d_rhs_x;
real *d_x; // cudaMalloc()
Particles *d_particles;

d_particles.x 

	set_pointer(IntegratedParticles *integratedParticles) {
		d_rhs_x = integratedParticles->x;
	}

}

kernel<<<>>>(particleHandler->d_particles);

particleHandler->d_x;

kernel(Particles *particles) {
	particles->x[i] = 0;
}
```

* $\Delta t \leq C \frac{h}{c + 1.2 (\alpha_{\nu} c + \beta_{\nu} \mu_{max})}$

* $\Delta t \leq \sqrt{\frac{h}{|\vec{a}|}}$

* $\Delta t \leq \begin{cases}
     a \frac{|f| + f_{min}}{|df|} & |df| > 0 \\
     \Delta t_{max} & |df| = 0 \\
    \end{cases}$ where $a < 1$
    
* $\Delta t \cdot v_{max} < \frac{h}{2} \, \Leftrightarrow \Delta t < \frac{h}{2 v_{max}}$

