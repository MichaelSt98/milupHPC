# Debugging

**Basic debugging:**

* using GDB: `mpirun -np <num processes> xterm -e gdb bin/runner -x initPipe.gdb`
	* see [debug_gdb.sh](debug_gdb.sh) 
* using LLDB: `mpirun -np <num processes> xterm -e lldb ./bin/runner -s initPipe.lldb`
	* see [debug_lldb.sh](debug_lldb.sh) 
* using CUDA-GDB: `mpirun -np <num processes> xterm -e cuda-gdb bin/runner`
	* see [cuda_debug.sh](cuda_debug.sh) 

______

Further, there are other and more sophisticated tools available for debugging MPI and/or CUDA programs. Among others, there are the following tools.

______

## MPI Tools

### ARM DDT

* [ARM DDT Debugger](https://www.arm.com/products/development-tools/server-and-hpc/forge/ddt)

Arm DDT is the number one server and HPC debugger in research, industry, and academia for software engineers and scientists developing C++, C, Fortran parallel and threaded applications on CPUs, GPUs, Intel and Arm. Arm DDT is trusted as a powerful tool for automatic detection of memory bugs and divergent behavior to achieve lightning-fast performance at all scales.

### Valgrind

* [Valgrind instrumentation framework for building dynamic analysis tools](https://www.valgrind.org/)

Valgrind is an instrumentation framework for building dynamic analysis tools. There are Valgrind tools that can automatically detect many memory management and threading bugs, and profile your programs in detail. You can also use Valgrind to build new tools.

### Intel Inspector

* [Intel Inspector](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/inspector.html)

Memory errors and nondeterministic threading errors are difficult to find without the right tool. Intel® Inspector is designed to find these errors. It is a dynamic memory and threading error debugger for C, C++, and Fortran applications that run on Windows* and Linux* operating systems.

### Oracle Developer Studio Thread Anaylzer

* [Oracle Thread Analyzer](https://www.oracle.com/application-development/technologies/developerstudio.html)

### MPI Runtime Correctness Analysis

* [MUST - MPI Runtime Correctness Analysis](https://www.i12.rwth-aachen.de/go/id/nrbe)

MUST detects usage errors of the Message Passing Interface (MPI) and reports them to the user. As MPI calls are complex and usage errors common, this functionality is extremely helpful for application developers that want to develop correct MPI applications. This includes errors that already manifest – segmentation faults or incorrect results – as well as many errors that are not visible to the application developer or do not manifest on a certain system or MPI implementation.

______

## CUDA Tools

* [NVIDIA: Debugging solutions](https://developer.nvidia.com/debugging-solutions)

### CUDA-MEMCHECK

* [NVIDIA: CUDA-MEMCHECK](https://docs.nvidia.com/cuda/cuda-memcheck/index.html)

Accurately identifying the source and cause of memory access errors can be frustrating and time-consuming.  CUDA-MEMCHECK detects these errors in your GPU code and allows you to locate them quickly.  CUDA-MEMCHECK also reports runtime execution errors, identifying situations that could otherwise result in an “unspecified launch failure” error when your application is running.

### CUDA-GDB

* [NVIDIA: CUDA-GDB](https://docs.nvidia.com/cuda/cuda-gdb/index.html)

When developing massively parallel applications on the GPU, you need a debugger capable of handling thousands of threads running simultaneously on each GPU in the system.  CUDA-GDB delivers a seamless debugging experience that allows you to debug both the CPU and GPU portions of your application simultaneously.

### Nsight

* [NVIDIA: Nsight](https://developer.nvidia.com/nsight-systems)

and the [NVIDIA Nsight Visual Studio Edition](https://developer.nvidia.com/nsight-visual-studio-edition)

NVIDIA® Nsight™ Systems is a system-wide performance analysis tool designed to visualize an application’s algorithms, help you identify the largest opportunities to optimize, and tune to scale efficiently across any quantity or size of CPUs and GPUs; from large server to our smallest SoC.

### Arm Forge

* [NVIDIA: Arm Forge](https://developer.nvidia.com/allinea-ddt)

Supporting the latest toolkits from NVIDIA, Arm DDT provides one of the most comprehensive debuggers of CPU and GPU applications available.
Relied on by software developers on CUDA-enabled laptops and workstations through to the world’s largest hybrid GPU/CPU supercomputers, Arm DDT is designed to help you tackle software bugs in CPU/GPU applications easily.

### TotalView

* [NVIDIA: TotalView](https://developer.nvidia.com/totalview-debugger)

TotalView is the leading dynamic analysis and debugging tool designed to handle complex CPU and GPU based multi-threaded, multi-process and multi-node cluster applications. It enables developers to analyze their C, C++, Fortran and mixed-language Python applications in order to quickly diagnose and fix bugs. Using TotalView’s powerful reverse debugging, memory debugging and advanced debugging technologies, developers are able to reduce development cycles and amount of time it takes to find and fix difficult bugs in their complex codes. TotalView supports the latest CUDA SDK’s, NVIDIA GPU hardware, Linux x86-64, Arm64, and OpenPower platforms and applications utilizing MPI and OpenMP technologies.
