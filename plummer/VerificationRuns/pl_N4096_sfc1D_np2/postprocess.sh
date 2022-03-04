export OMP_NUM_THREADS=1
./H5Renderer/bin/h5renderer -c H5Renderer/h5renderer.info -i plummer/pl_N4096_sfc1D_np2/ -o plummer/pl_N4096_sfc1D_np2/ > /dev/null
yes | ./H5Renderer/createMP4From plummer/pl_N4096_sfc1D_np2/ &> /dev/null

./postprocessing/PlotPlummer.py -Q -d plummer/pl_N4096_sfc1D_np2/ -o plummer/pl_N4096_sfc1D_np2/

./postprocessing/GetMinMaxMean.py -i plummer/pl_N4096_sfc1D_np2/ -o plummer/pl_N4096_sfc1D_np2/
./postprocessing/PlotMinMaxMean.py -i plummer/pl_N4096_sfc1D_np2/min_max_mean.csv -o plummer/pl_N4096_sfc1D_np2/ -a

./postprocessing/Performance.py -f plummer/pl_N4096_sfc1D_np2/log/performance.h5 -d plummer/pl_N4096_sfc1D_np2/ -s 0
