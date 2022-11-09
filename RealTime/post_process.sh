#!/usr/bin/env bash
source activate /home/wchapman/anaconda3/envs/post_process

##find most recent forecast
rm -rf /glade/work/wchapman/ARcnn_LRP_DEMO/RealTime/dump_forecast/*

# !!!!!this is the directory of the new wwrf  cf files: !!!!alter  for comet!!!
cd /glade/scratch/wchapman/WWRF_cf

newFORE=$(ls -td -- */ | head -n 1 | cut -d'/' -f1)
# !!!!!this is the directory of the new wwrf  cf files: !!!!alter  for comet!!!
dataREP="/glade/scratch/wchapman/WWRF_cf/"
totFIL="$dataREP$newFORE"

echo "RUNNING FORECAST "$newFORE

cd /glade/work/wchapman/ARcnn_LRP_DEMO/RealTime/
cp -r $totFIL /glade/work/wchapman/ARcnn_LRP_DEMO/RealTime/dump_forecast


cd /glade/work/wchapman/ARcnn_LRP_DEMO/RealTime/

#start post-processing.
Basepath="/glade/work/wchapman/ARcnn_LRP_DEMO/RealTime/"
Fpath=$(ls /glade/work/wchapman/ARcnn_LRP_DEMO/RealTime/dump_forecast | xargs -n 1 basename)

echo "REGRIDDING: "$Fpath

##########
#rename files block
##########
#.....
count=0
for file in ${Basepath}dump_forecast/${Fpath}/*d01*.nc; do
  nummy=`printf %03d $count`
  echo cp "$file" "${file/.nc/_F"$nummy".nc}"
  (( count++ ))
done

##########
##########
#F000 block
ERAGrid="/glade/work/wchapman/ARcnn_LRP_DEMO/RealTime/regrid_folder/ERA5Grid.txt"
startpath="${Basepath}dump_forecast/${Fpath}/*_F000.nc" #this can be changed to _F000.nc
endpath="${Basepath}/regrid_folder/IVT_ERAGrid_F000.nc"
/glade/u/apps/ch/opt/cdo/2.0.1/gnu/10.1.0/bin/cdo remapbil,$ERAGrid $startpath $endpath
# load anaconda environment tfp ... however this is done on COMET
python /glade/work/wchapman/ARcnn_LRP_DEMO/RealTime/python_scripts/F000.py
#this saves nc file in ./output_files

#F003 block
startpath="${Basepath}dump_forecast/${Fpath}/*_F003.nc" #this can be changed to _F003.nc
endpath="${Basepath}/regrid_folder/IVT_ERAGrid_F003.nc"
/glade/u/apps/ch/opt/cdo/2.0.1/gnu/10.1.0/bin/cdo remapbil,$ERAGrid $startpath $endpath
# load anaconda environment tfp ... however this is done on COMET
python /glade/work/wchapman/ARcnn_LRP_DEMO/RealTime/python_scripts/F003.py
#this saves nc file in ./output_files

#F006 block
startpath="${Basepath}dump_forecast/${Fpath}/*_F006.nc" #this can be changed to _F006.nc
endpath="${Basepath}/regrid_folder/IVT_ERAGrid_F006.nc"
/glade/u/apps/ch/opt/cdo/2.0.1/gnu/10.1.0/bin/cdo remapbil,$ERAGrid $startpath $endpath
# load anaconda environment tfp ... however this is done on COMET
python /glade/work/wchapman/ARcnn_LRP_DEMO/RealTime/python_scripts/F006.py
#this saves nc file in ./output_files

#rest of the blocks ....as above....
#.........

#F003 block: 
#MerrGrid="/data/downloaded/Forecasts/Machine_Learning/Post_Process_GFS/regrid_folder/MerraGrid.txt"
#startpath="${Basepath}dump_forecast/${Fpath}/*_F003.nc"
#endpath="${Basepath}/regrid_folder/IVT_MerrGrid_F003.nc"
#/apps/cdo-1.8.2_gnu_fortran_4.8.5-11/bin/cdo remapcon,$MerrGrid $startpath $endpath
##create forecast: 
##/home/wchapman/anaconda3/envs/post_process/bin/python /data/downloaded/Forecasts/Machine_Learning/Post_Process_GFS/python_scripts/F003.py
#/apps/Anaconda/anaconda3/bin/python /data/downloaded/Forecasts/Machine_Learning/Post_Process_GFS/python_scripts/F003.py
##remove IVTfile 
#rm $endpath

##F006 block: 
#MerrGrid="/data/downloaded/Forecasts/Machine_Learning/Post_Process_GFS/regrid_folder/MerraGrid.txt"
#startpath="${Basepath}dump_forecast/${Fpath}/*_F006.nc"
#endpath="${Basepath}/regrid_folder/IVT_MerrGrid_F006.nc"
#/apps/cdo-1.8.2_gnu_fortran_4.8.5-11/bin/cdo remapcon,$MerrGrid $startpath $endpath
##create forecast: 
##/home/wchapman/anaconda3/envs/post_process/bin/python /data/downloaded/Forecasts/Machine_Learning/Post_Process_GFS/python_scripts/F006.py
#/apps/Anaconda/anaconda3/bin/python /data/downloaded/Forecasts/Machine_Learning/Post_Process_GFS/python_scripts/F006.py
##remove IVTfile 
#rm $endpath



#export NCARG_ROOT="/apps/NCL/ncl-6.6.2"
#/apps/NCL/ncl-6.6.2/bin/ncl /data/downloaded/Forecasts/Machine_Learning/Post_Process_GFS/plot_ivt.ncl

#export NCARG_ROOT="/apps/NCL/ncl-6.6.2"
#/apps/NCL/ncl-6.6.2/bin/ncl /data/downloaded/Forecasts/Machine_Learning/Post_Process_GFS/plot_ivt_gfs.ncl
##/apps/NCL/ncl-6.6.2/bin/ncl /data/downloaded/Forecasts/Machine_Learning/Post_Process_GFS/plot_ivt_gefs.ncl

#scp /data/downloaded/Forecasts/Machine_Learning/Post_Process_GFS/figures/*.png cw3e@cw3e-web.ucsd.edu:/var/www/cw3e/htdocs/images/GFS_ARcnn


