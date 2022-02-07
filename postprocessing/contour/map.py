#!/usr/bin/python3

import os
import sys

def main():
    N = 284
    
    for i in range(1,N+1):
        fname = 'gas.'+str(i).zfill(4)
        print("Mapping", fname);
        os.system('./map_sph_to_grid -f '+fname+' -N 2144822 -x -3.2e14 -X 3.2e14 -y -3.2e14 -Y 3.2e14 -z -3.2e14 -Z 3.2e14 -g 128')

if __name__ == '__main__':
    main()
