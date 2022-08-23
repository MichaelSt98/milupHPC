# H5Renderer

* compile via `make` using Makefile
* adjust settings in *h5renderer.info* (image height, system size, ...)
* execute binary `bin/h5renderer`
	* e.g. `./H5Renderer/bin/h5renderer -c <H5Renderer config file> -i <input directory with HDF5 files> -o <output directory>`

```
Create images from HDF5 files.
Usage:
  h5renderer [OPTION...]

  -c, --config arg  Path to config file (default: h5renderer.info)
  -v, --verbose     More printouts for debugging
  -i, --input arg   Read files from given path (otherwise from config file) 
                    (default: -)
  -o, --output arg  Write result files to given path (default: ./output)
  -x, --mark        Mark particles as + instead of points
  -z, --zoom arg    Zoom in factor to show more details (default: 1)
  -h, --help        Show this help
```

* for creating a mp4 from the images: `./H5Renderer/createMP4From <input directory with images>` (based on ffmpeg)
* for creating a gif from the mp4: `./createGifFromMP4`

<details>
<summary>
./createMP4From
</summary>

```
ffmpeg -pattern_type glob -i "$1*.ppm" -vcodec libx264 -s 1024x512 -pix_fmt yuv420p $1movie.mp4
```

</details>


## Implementation details

```c
src/include
├── Color.h // Color definitions
├── ConfigParser.h/.cpp // Config file parser
├── H5Renderer.h/.cpp // actual Renderer
├── Logger.h/.cpp // Logger
├── Particle.h // Particle struct/class
└── cxxopts.h // command line parsing
```

* particle data loaded from H5/HDF5 files
	* including positions (`/x`), keys (`/keys`) and (`/ranges`)
* particles mapped to pixels according to their position and colored according to their process assignment (via key and ranges) 	