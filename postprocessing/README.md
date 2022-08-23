# Postprocessing

**Scripts for postprocessing, visualization, ...**

* See also [H5Renderer](../H5Renderer/) for basic rendering ([README](../H5Renderer/README.md))


## Snippets

* reading the particle output files

```python
import h5py
import numpy as np

h5file = "sample.h5"
f = h5py.File(h5file, 'r')

mass = f["m"][:]
pos = f["x"][:] # x, y, z component
vel = f["v"][:] # x, y, z component
COM = f["COM"][:] # x, y, z component
key = f["key"][:] # SFC key
proc = f["proc"][:]
time = f["time"][:]
# ...
```

* reading the performance files

```python
import h5py
import numpy as np

h5file = "sample.h5"
f = h5py.File(h5file, 'r')
keys = f['time'].keys()

for key in keys:
	# e.g. average for each entry/key
	elapsed = np.array(f["time/{}".format(key)][:])
	elapsed_max = [np.array(elem).max() for elem in elapsed]
	mean_elapsed = np.array(elapsed_max).mean()
	print("{}: {} ms".format(key, mean_elapsed))
```


## Paraview

* [ParaView](https://www.paraview.org/)

ParaView is an open-source, multi-platform data analysis and visualization application. ParaView users can quickly build visualizations to analyze their data using qualitative and quantitative techniques. The data exploration can be done interactively in 3D or programmatically using ParaView’s batch processing capabilities.

Create a XDMF file in order to facilitate the data access for Paraview:

```python
#!/usr/bin/env python3

import h5py
import sys
import argparse

parser = argparse.ArgumentParser(description='Generate xdmf file from .h5 miluph output files for paraview postprocessing.\n Open the generated .xdmf file with paraview.')

parser.add_argument('--output', help='output file name', default='paraview.xdmf')
parser.add_argument('--dim', help='dimension', default='3')
parser.add_argument('--input_files', nargs='+', help='input file names', default='none')

args = parser.parse_args()

if len(sys.argv) < 3:
    print("Try --help to get a usage.")
    sys.exit(0);

def write_xdmf_header(fh):
    header = """
<Xdmf>
<Domain Name="MSI">
<Grid Name="CellTime" GridType="Collection" CollectionType="Temporal">
"""
    fh.write(header)
    
def write_xdmf_footer(fh):
    footer = """
</Grid>
</Domain>
</Xdmf>
"""
    fh.write(footer)
    
xdmfh = open(args.output, 'w')

# write header of xdmf file
write_xdmf_header(xdmfh)

# now process input files
# scalar floats only 
wanted_attributes = ['rho', 'p', 'sml', 'noi', 'e']

for hfile in args.input_files:
    print("Processing %s " % hfile)
    try: 
        f = h5py.File(hfile, 'r')
    except IOError:
        print("Cannot open %s, exiting." % hfile)
        sys.exit(1)

    current_time = f['time'][...]
    # write current time entry
    xdmfh.write('<Grid Name="particles" GridType="Uniform">\n')
    xdmfh.write('<Time Value="%s" />\n' % current_time[0])
    mylen = len(f['x'])
    xdmfh.write('<Topology TopologyType="Polyvertex" NodesPerElement="%s"></Topology>\n' % mylen)
    if args.dim == '3':
        xdmfh.write('<Geometry GeometryType="XYZ">\n')
        xdmfh.write('<DataItem DataType="Float" Dimensions="%s 3" Format="HDF">\n' % mylen)
    else:
        xdmfh.write('<Geometry GeometryType="XY">\n')
        xdmfh.write('<DataItem DataType="Float" Dimensions="%s 2" Format="HDF">\n' % mylen)
    xdmfh.write('%s:/x\n' % hfile) 
    xdmfh.write('</DataItem>\n')
    xdmfh.write('</Geometry>\n')

    # velocities
    xdmfh.write('<Attribute AttributeType="Vector" Centor="Node" Name="velocity">\n')
    xdmfh.write('<DataItem DataType="Float" Dimensions="%s %s" Format="HDF">\n' % (mylen, args.dim))
    xdmfh.write('%s:/v\n' % hfile)
    xdmfh.write('</DataItem>\n')
    xdmfh.write('</Attribute>\n')

    # wanted_attributes
    for myattr in wanted_attributes:
        datatype = "Float"
        if myattr in ['number_of_interactions', 'material_type']:
            datatype = "Integer"
        xdmfh.write('<Attribute AttributeType="Scalar" Centor="Node" Name="%s">\n' % myattr)
        xdmfh.write('<DataItem DataType="%s" Dimensions="%s 1" Format="HDF">\n' % (datatype, mylen))
        xdmfh.write('%s:/%s\n' % (hfile, myattr))
        xdmfh.write('</DataItem>\n')
        xdmfh.write('</Attribute>\n')
    xdmfh.write('</Grid>\n')
    f.close()

# write footnote of xdmf file
write_xdmf_footer(xdmfh)
```

### Remote Paraview

The information presented here is based on [https://docs.paraview.org/en/latest/ReferenceManual/parallelDataVisualization.html](https://docs.paraview.org/en/latest/ReferenceManual/parallelDataVisualization.html)

1. install locally paraview on your laptop and on the server with the identical version.
2. run the binary pvserver on the server side (it's in the same directory as the paraview binary), e.g.  
~/sw/paraview/ParaView-5.8.0-MPI-Linux-Python3.7-64bit/bin/pvserver --force-offscreen-rendering
IMPORTANT: log in to the server via ssh without X forwarding (ssh +X cpt-kamino.am10.uni-tuebingen.de)
3. start the client on your laptop, go to File->connect and add cpt-kamino to the server list
4. if the client connection is successful, you can choose File->open and open the file on the server. the rendering will be done on the server and the client will display the final picture.


## Custom scripts

* [RhsPerformance.py](RhsPerformance.py): get average execution time for each part of the simulation
	* usage: `./RhsPerformance -d <HDF5 performance file>`  
* [PerformanceComparison.py](PerformanceComparison.py): compare average execution time for each part of the simulation
	* usage: `./RhsPerformance -a <HDF5 performance file 1> -b <HDF5 performance file 2>`  
* [PlotPlummer.py](PlotPlummer.py): plot mass quantiles for the plummer test case
	* usage: e.g. `./PlotPlummer.py -Q -d <input data directory> -o <output directory>` 
* [PlotSedov.py](PlotSedov.py): plot density, pressure and internal energy in dependence of the radius for the sedov test case, including the semi-analytical solution
	* usage: e.g.  `./PlotSedov.py -i <input HDF5 particle file> -o <output directory> -a -p 1`
	* for applying script for all output files: e.g. `find <directory with input files> -type f -name 'ts*.h5' -print0 | parallel -0 -j 1 ./PlotSedov.py -i {} -o <output directory> -a -p 1 \;`
		* combining those to a video: `ffmpeg -r 24 -i <input directory>/ts%06d.h5.png -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -pix_fmt yuv420p -vcodec libx264 -y -an <output directory>/evolution.mp4`
* [PlotBB.py](PlotBB.py): plot density evolution (for the isothermal collapse test case)
	* usage: `PlotBB.py -d <input directory> -o <output directory>`
* [GetMinMaxMean.py](GetMinMaxMean.py): calculate min, max, mean for each entry and each output file and write/summarize to CSV file  
	* usage: `./GetMinMaxMean.py -i <input directory> -o <output directory>`
* [PlotMinMaxMean.py](PlotMinMaxMean.py): take min, max, mean CSV file and plot for each entry
	* usage: `./PlotMinMaxMean.py -i <input directory>/min_max_mean.csv -o <output directory> -a` 
	* requires previous execution of `PlotMinMaxMean.py`
* [VisualizeDistribution.py](VisualizeDistribution.py): visualize a particle distribution with Matplotlib 3D
* [VisualizeDomains_2D.py](VisualizeDomains_2D.py): visualize domain borders or rather domains for 2D simulations
* [VisualizeDomains_3D.py](VisualizeDomains_3D.py): visualize domain borders or rather domains for 3D simulations

