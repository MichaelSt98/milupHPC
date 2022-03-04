export OMP_NUM_THREADS=1
./H5Renderer/bin/h5renderer -c H5Renderer/h5renderer.info -i plummer/pl_N4096_sfc1D_np4/ -o plummer/pl_N4096_sfc1D_np4/ > /dev/null
yes | ./H5Renderer/createMP4From plummer/pl_N4096_sfc1D_np4/ &> /dev/null

./postprocessing/PlotPlummer.py -Q -d plummer/pl_N4096_sfc1D_np4/ -o plummer/pl_N4096_sfc1D_np4/

./postprocessing/GetMinMaxMean.py -i plummer/pl_N4096_sfc1D_np4/ -o plummer/pl_N4096_sfc1D_np4/
./postprocessing/PlotMinMaxMean.py -i plummer/pl_N4096_sfc1D_np4/min_max_mean.csv -o plummer/pl_N4096_sfc1D_np4/ -a

./postprocessing/Performance.py -f plummer/pl_N4096_sfc1D_np4/log/performance.h5 -d plummer/pl_N4096_sfc1D_np4/ -s 0
