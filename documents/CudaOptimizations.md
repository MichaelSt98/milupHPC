# Cuda Optimizations

_____

* 400 ms
* 
_____

* Resources
	* [CoffeeBeforeArch: CUDA Crash Course: GPU Performance Optimizations Part 1](https://www.youtube.com/watch?v=b8ESCws3_1s&t=56s) 
	* [NVIDIA: CUDA OPTIMIZATION TIPS, TRICKS AND TECHNIQUES](https://on-demand.gputechconf.com/gtc/2017/presentation/s7122-stephen-jones-cuda-optimization-tips-tricks-and-techniques.pdf)

## Generally

* avoid global memory read/write
	* utitilize/abuse L1 cache 
* avoid thread divergence
* avoid thread synchronization
* thread Block should be a multiple of number of SMs
* pin memory
* **streams and concurrency**
	* overlap copy and computing 

## Temporary variables 

* Caching variables which are updated/read frequently (e.g. within loop)
	* to possibly hit L1 cache
	* to reduce global memory read/write

e.g.: 

```cpp
for (int i=0; i<N; i++) {
	tmp += a[row * N + i] * b[i * N + col];
}
c[row * N + col] = tmp;
```

instead of

```cpp
for (int i=0; i<N; i++) {
	c[row * N + col] += a[row * N + i] * b[i * N + col];
}
```

<details>
<summary>Code sample</summary>

```cpp
__global__ void matrixMul(const int *a, const int *b, int *c, int N) {
  // Compute each thread's global row and column index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Iterate over row, and down column
  c[row * N + col] = 0;
  for (int k = 0; k < N; k++) {
    // Accumulate results for a single element
    c[row * N + col] += a[row * N + k] * b[k * N + col];
  }
}

matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N);
```

</details>  

## Coalesced access

* Reduce request(s) from (global) memory by aligning arrays

## Prefetching data

* prefetch data to allow coalesced global memory read

e.g.:

```cpp
for (int i=0; i<N; i+= 4) {
	int4 a_tmp = reinterpret_cast<int4*>(&a[row * N + i][0]);
	
	tmp += a_tmp.x * b[i * N + col];
	tmp += a_tmp.y * b[(i + 1) * N + col];
	tmp += a_tmp.z * b[(i + 2) * N + col];
	tmp += a_tmp.w * b[(i + 3) * N + col];
}
``` 


## Unrolling

* compiler explicitly unrolls loop in order to reduce loop overhead

e.g.:


```cpp
#pragma unroll
for (int i=0; i<N; i+= 4) {
	int4 a_tmp = reinterpret_cast<int4*>(&a[row * N + i][0]);
	
	tmp += a_tmp.x * b[i * N + col];
	tmp += a_tmp.y * b[(i + 1) * N + col];
	tmp += a_tmp.z * b[(i + 2) * N + col];
	tmp += a_tmp.w * b[(i + 3) * N + col];
}
``` 


## Shared/Tiled memory

* **Idea:** guarantee something is in the cache in order to have fewer related stalls
* **Solution:** Shared memory
	* User-managed L1-cache
	* Private per-threadblock
* it is possibly necessary to batch data in order to fit into L1 cache

<details>
<summary>Code sample</summary>

```cpp
__global__ void matrixMul(const int *a, const int *b, int *c) {
  // Compute each thread's global row and column index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Statically allocated shared memory
  __shared__ int s_a[SHMEM_SIZE];
  __shared__ int s_b[SHMEM_SIZE];

  // Accumulate in temporary variable
  int tmp = 0;

  // Sweep tile across matrix
  for (int i = 0; i < N; i += blockDim.x) {
    // Load in elements for this tile
    s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * N + i + threadIdx.x];
    s_b[threadIdx.y * blockDim.x + threadIdx.x] =
        b[i * N + threadIdx.y * N + col];

    // Wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // Do matrix multiplication on the small matrix
    for (int j = 0; j < blockDim.x; j++) {
      tmp +=
          s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
    }

    // Wait for all threads to finish using current tiles before loading in new
    // ones
    __syncthreads();
  }

  // Write back results
  c[row * N + col] = tmp;
}

...
matrixMul<<<blocks, threads>>>(d_a, d_b, d_c);
...
 
```

</details>  

 	
