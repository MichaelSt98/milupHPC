#!/usr/bin/env bash

mpirun -np 2 xterm -e gdb bin/runner -x debug/initPipe.gdb
