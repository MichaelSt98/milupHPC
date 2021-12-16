export OMP_NUM_THREADS=1
./H5Renderer/bin/h5renderer -c H5Renderer/h5renderer.info -i plummer/pl_N4096_sfc0F_np1/ -o plummer/pl_N4096_sfc0F_np1/ > /dev/null
yes | ./H5Renderer/createMP4From plummer/pl_N4096_sfc0F_np1/ &> /dev/null

./postprocessing/PlotPlummer.py -Q -d plummer/pl_N4096_sfc0F_np1/ -o plummer/pl_N4096_sfc0F_np1/

./postprocessing/GetMinMaxMean.py -i plummer/pl_N4096_sfc0F_np1/ -o plummer/pl_N4096_sfc0F_np1/
./postprocessing/PlotMinMaxMean.py -i plummer/pl_N4096_sfc0F_np1/min_max_mean.csv -o plummer/pl_N4096_sfc0F_np1/ -a

./postprocessing/Performance.py -f plummer/pl_N4096_sfc0F_np1/log/performance.h5 -d plummer/pl_N4096_sfc0F_np1/ -s 0
