#!/usr/bin/env bash

mpirun -np 2 xterm -e cuda-gdb bin/runner #-x debug/initPipe.cuda
#mpirun -np 2 -host nv1,nv2 -x DISPLAY=host.nvidia.com:0 xterm -e /usr/local/cuda-10.1/bin/cuda-gdb bin/runner