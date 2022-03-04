* 31^3 particles (ca. 30k)
* 61^3 particles (ca. 200k (226k))
* 81^3 particles (ca. 500k (531k))
* 101^3 particles (ca. 1M)
* 126^3 particles (ca. 2M)
* 171^3 particles (ca. 5M)

* energy smoothed over sml, thus always same amount of particles with higher initial energy:
	* particles sedov: 515
	* thus, for different particle numbers different radius

* `find . -type f -name '*.h5' -print0 | parallel -0 -j8 ./plot_sedov.py {} \;`
* `ffmpeg -r 24 -i ts%06d.h5.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p -vcodec libx264 -y -an evolution.mp4` 
