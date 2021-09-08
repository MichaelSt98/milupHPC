* cudaFree

```
(gpuErrorcheck\(cudaFree\()(.*?)(\)\)\;)
Test string: gpuErrorcheck(cudaFree(d_uid));
Substitution: cuda::free($2);
Result: cuda::free(d_uid);
```

* cudaMemset

```
(gpuErrorcheck\(cudaMemset\()(.*?\,)(.*?\,)(.*?[0-9])(.*?)(\)\)\;)
Test string: gpuErrorcheck(cudaMemset(d_numParticles, numParticles, 2*sizeof(integer)));
Substitution: cuda::set($2$3$4);
Result: cuda::set(d_numParticles, numParticles, 2);


(gpuErrorcheck\(cudaMemset\()(.*?\,)(.*?\,)(.*?)(\)\)\;)
Test string: gpuErrorcheck(cudaMemset(d_numParticles, numParticles, sizeof(integer)));
Substitution: cuda::set($2$3 1);
Result: cuda::set(d_numParticles, numParticles, 1);
```

* cudaMalloc

```
(gpuErrorcheck\(cudaMalloc\(\(void\*\*\)\&)(.*?\,)(.*)(\*.*?\)\))
Test string: gpuErrorcheck(cudaMalloc((void**)&d_keys, numParticlesLocal*sizeof(keyType)));
Substitution: cuda::malloc($2$3
Result: cuda::malloc(d_keys, numParticlesLocal);

(gpuErrorcheck\(cudaMalloc\(\&)(.*?\,)(.*)(\*.*?\)\))
Test string: gpuErrorcheck(cudaMalloc(&d_keys, numParticlesLocal*sizeof(keyType)));
Substitution: cuda::malloc($2$3
Result: cuda::malloc(d_keys, numParticlesLocal);
```

* cudaMemcpy

```
(gpuErrorcheck\(cudaMemcpy\()(.*?)(,)(.*)(,)(.*?)(\*)(.*)
Test string: gpuErrorcheck(cudaMemcpy(d_mass,  h_mass,  numParticles*sizeof(real), cudaMemcpyHostToDevice));
Substitution: cuda::copy($4, $2, To::device);
Result: cuda::copy(  h_mass, d_mass, To::device);
```