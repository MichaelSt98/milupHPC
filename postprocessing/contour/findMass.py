open#!/usr/bin/python3

#finds the mass inside a specified cube
fname = 'gas.0284'
xmin = 0 
xmax = 8e13
ymin = 4e13
ymax = 1.4e14
zmin = -5e12
zmax = 5e12

mass = 0
with open(fname, 'r') as f:
    for line in f:
        data = line.split()
        x = float(data[0])
        y = float(data[1])
        z = float(data[2])
        if x > xmin and x < xmax and y > ymin and y < ymax and z > zmin and z < zmax:
            mass += float(data[6])


print('Masse:', mass)
