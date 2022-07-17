#!/usr/bin/env bash

mpirun -np 1 xterm -e lldb ./bin/runner -s debug/initPipe.lldb
