#!/bin/bash

mpirun -np 2 bin/runner -n 200 -f plummer/initial_plummer/plN4096seed2723515521.h5 -C plummer/pl_N4096_sfc0F_np2/config.info

./plummer/pl_N4096_sfc0F_np2/postprocess.sh
