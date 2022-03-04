#!/bin/bash

mpirun -np 4 bin/runner -n 200 -f plummer/initial_plummer/plN4096seed2723515521.h5 -C plummer/pl_N4096_sfc0D_np4/config.info

./plummer/pl_N4096_sfc0D_np4/postprocess.sh
