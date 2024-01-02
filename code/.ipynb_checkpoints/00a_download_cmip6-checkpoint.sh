#!/bin/bash

#PBS -l nodes=1:ppn=1
#PBS -l walltime=12:00:00
#PBS -l pmem=10gb
#PBS -A kaf26_c_g_sc_default
#PBS -j oe
#PBS -o download_cmip6_ssp.out
#PBS -l feature=rhel7

echo "Job started on `hostname` at `date`"

# Go to the correct place
cd $PBS_O_WORKDIR

# Activate env
micromamba activate acccmip6-2023-06

# Download via acccmip6: https://github.com/TaufiqHassan/acccmip6

export STORAGE_DIR=/gpfs/group/kaf26/default/public/CMIP6

## Run through models

# SSPs
mkdir $STORAGE_DIR/ACCESS-CM2/
mkdir $STORAGE_DIR/ACCESS-CM2/ssp245/
acccmip6 -o D -m ACCESS-CM2 -e ssp245 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3 -skip _gr_ -yr 85 -dir $STORAGE_DIR/ACCESS-CM2/ssp245/ &> $STORAGE_DIR/ACCESS-CM2/ssp245/download.out
mkdir $STORAGE_DIR/ACCESS-CM2/ssp370/
acccmip6 -o D -m ACCESS-CM2 -e ssp370 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3 -skip _gr_ -yr 85 -dir $STORAGE_DIR/ACCESS-CM2/ssp370/ &> $STORAGE_DIR/ACCESS-CM2/ssp370/download.out
mkdir $STORAGE_DIR/ACCESS-CM2/ssp585/
acccmip6 -o D -m ACCESS-CM2 -e ssp585 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3 -skip _gr_ -yr 85 -dir $STORAGE_DIR/ACCESS-CM2/ssp585/ &> $STORAGE_DIR/ACCESS-CM2/ssp585/download.out

mkdir $STORAGE_DIR/ACCESS-ESM1-5/
mkdir $STORAGE_DIR/ACCESS-ESM1-5/ssp245/
acccmip6 -o D -m ACCESS-ESM1-5 -e ssp245 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3,4,5 -skip _gr_ -yr 85 -dir $STORAGE_DIR/ACCESS-ESM1-5/ssp245/ &> $STORAGE_DIR/ACCESS-ESM1-5/ssp245/download.out
mkdir $STORAGE_DIR/ACCESS-ESM1-5/ssp370/
acccmip6 -o D -m ACCESS-ESM1-5 -e ssp370 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3,4,5 -skip _gr_ -yr 85 -dir $STORAGE_DIR/ACCESS-ESM1-5/ssp370/ &> $STORAGE_DIR/ACCESS-ESM1-5/ssp370/download.out
mkdir $STORAGE_DIR/ACCESS-ESM1-5/ssp585/
acccmip6 -o D -m ACCESS-ESM1-5 -e ssp585 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3,4,5 -skip _gr_ -yr 85 -dir $STORAGE_DIR/ACCESS-ESM1-5/ssp585/ &> $STORAGE_DIR/ACCESS-ESM1-5/ssp585/download.out

mkdir $STORAGE_DIR/AWI-CM-1-1-MR/
mkdir $STORAGE_DIR/AWI-CM-1-1-MR/ssp245/
acccmip6 -o D -m AWI-CM-1-1-MR -e ssp245 -v tas,tasmax,tasmin,pr -f day -rlzn 1 -skip _gr_ -yr 85 -dir $STORAGE_DIR/AWI-CM-1-1-MR/ssp245/ &> $STORAGE_DIR/AWI-CM-1-1-MR/ssp245/download.out
mkdir $STORAGE_DIR/AWI-CM-1-1-MR/ssp370/
acccmip6 -o D -m AWI-CM-1-1-MR -e ssp370 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3,4,5 -skip _gr_ -yr 85 -dir $STORAGE_DIR/AWI-CM-1-1-MR/ssp370/ &> $STORAGE_DIR/AWI-CM-1-1-MR/ssp370/download.out
mkdir $STORAGE_DIR/AWI-CM-1-1-MR/ssp585/
acccmip6 -o D -m AWI-CM-1-1-MR -e ssp585 -v tas,tasmax,tasmin,pr -f day -rlzn 1 -skip _gr_ -yr 85 -dir $STORAGE_DIR/AWI-CM-1-1-MR/ssp585/ &> $STORAGE_DIR/AWI-CM-1-1-MR/ssp585/download.out
 
mkdir $STORAGE_DIR/CanESM5/
mkdir $STORAGE_DIR/CanESM5/ssp245/
acccmip6 -o D -m CanESM5 -e ssp245 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3,4,5,6,7 -skip _gr_,1p2,1f2 -n esgf3.dkrz.de -dir $STORAGE_DIR/CanESM5/ssp245/ &> $STORAGE_DIR/CanESM5/ssp245/download.out
mkdir $STORAGE_DIR/CanESM5/ssp370/
acccmip6 -o D -m CanESM5 -e ssp370 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3,4,5,6,7 -skip _gr_,1p2,1f2 -n esgf3.dkrz.de -dir $STORAGE_DIR/CanESM5/ssp370/ &> $STORAGE_DIR/CanESM5/ssp370/download.out
mkdir $STORAGE_DIR/CanESM5/ssp585/
acccmip6 -o D -m CanESM5 -e ssp585 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3,4,5,6,7 -skip _gr_,1p2,1f2 -n esgf3.dkrz.de -yr 85 -dir $STORAGE_DIR/CanESM5/ssp585/ &> $STORAGE_DIR/CanESM5/ssp585/download.out

mkdir $STORAGE_DIR/EC-Earth3/
mkdir $STORAGE_DIR/EC-Earth3/ssp245/
acccmip6 -o D -m EC-Earth3 -e ssp245 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,4 -yr 85 -dir $STORAGE_DIR/EC-Earth3/ssp245/ -n esgf3.dkrz.de,esgf-data1.llnl.gov &> $STORAGE_DIR/EC-Earth3/ssp245/download.out
acccmip6 -o D -m EC-Earth3 -e ssp245 -v tas -f day -rlzn 2,4 -yr 85 -skip 1f2 -dir $STORAGE_DIR/EC-Earth3/ssp245/ -n esgf3.dkrz.de,esgf-data1.llnl.gov &> $STORAGE_DIR/EC-Earth3/ssp245/download.out
mkdir $STORAGE_DIR/EC-Earth3/ssp370/
acccmip6 -o D -m EC-Earth3 -e ssp370 -v tas,tasmax,tasmin,pr -f day -rlzn 1,4 -yr 85 -dir $STORAGE_DIR/EC-Earth3/ssp370/ -n esgf.ceda.ac.uk,esgf-cnr.hpc.cineca.it &> $STORAGE_DIR/EC-Earth3/ssp370/download.out
mkdir $STORAGE_DIR/EC-Earth3/ssp585/
acccmip6 -o D -m EC-Earth3 -e ssp585 -v tas,tasmax,tasmin,pr -f day -rlzn 1,3,4 -yr 85 -dir $STORAGE_DIR/EC-Earth3/ssp585/ -n esgf-data1.llnl.gov &> $STORAGE_DIR/EC-Earth3/ssp585/download.out

mkdir $STORAGE_DIR/EC-Earth3-Veg/
mkdir $STORAGE_DIR/EC-Earth3-Veg/ssp245/
acccmip6 -o D -m EC-Earth3-Veg -e ssp245 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3,4,5 -dir $STORAGE_DIR/EC-Earth3-Veg/ssp245/ -yr 85 -n esg-dn1.nsc.liu.se,esgf-cnr.hpc.cineca.it,esgf.bsc,esgf-data1.llnl.gov,esgf-data04.diasjp.net &> $STORAGE_DIR/EC-Earth3-Veg/ssp245/download.out
mkdir $STORAGE_DIR/EC-Earth3-Veg/ssp370/
acccmip6 -o D -m EC-Earth3-Veg -e ssp370 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3,4 -dir $STORAGE_DIR/EC-Earth3-Veg/ssp370/ -yr 85 -n esg-dn1.nsc.liu.se,esgf-cnr.hpc.cineca.it,esgf.bsc,esgf-data1.llnl.gov,esgf-data04.diasjp.net &> $STORAGE_DIR/EC-Earth3-Veg/ssp370/download.out
mkdir $STORAGE_DIR/EC-Earth3-Veg/ssp585/
acccmip6 -o D -m EC-Earth3-Veg -e ssp585 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3,4 -dir $STORAGE_DIR/EC-Earth3-Veg/ssp585/ -yr 85 -n esg-dn1.nsc.liu.se,esgf-cnr.hpc.cineca.it,esgf.bsc.es,esgf-data04.diasjp.net &> $STORAGE_DIR/EC-Earth3-Veg/ssp585/download.out

mkdir $STORAGE_DIR/FGOALS-g3/
mkdir $STORAGE_DIR/FGOALS-g3/ssp245/
acccmip6 -o D -m FGOALS-g3 -e ssp245 -v tas,tasmax,tasmin,pr -f day -rlzn 1,3,4 -skip _gr_ -yr 85 -dir $STORAGE_DIR/FGOALS-g3/ssp245/ &> $STORAGE_DIR/FGOALS-g3/ssp245/download.out
mkdir $STORAGE_DIR/FGOALS-g3/ssp370/
acccmip6 -o D -m FGOALS-g3 -e ssp370 -v tas,tasmax,tasmin,pr -f day -rlzn 1,3,4,5 -skip _gr_ -yr 85 -dir $STORAGE_DIR/FGOALS-g3/ssp370/ &> $STORAGE_DIR/FGOALS-g3/ssp370/download.out
mkdir $STORAGE_DIR/FGOALS-g3/ssp585/
acccmip6 -o D -m FGOALS-g3 -e ssp585 -v tas,tasmax,tasmin,pr -f day -rlzn 1,3,4 -skip _gr_ -yr 85 -dir $STORAGE_DIR/FGOALS-g3/ssp585/ &> $STORAGE_DIR/FGOALS-g3/ssp585/download.out

mkdir $STORAGE_DIR/HadGEM3-GC31-LL/
mkdir $STORAGE_DIR/HadGEM3-GC31-LL/ssp245/
acccmip6 -o D -m HadGEM3-GC31-LL -e ssp245 -v tas,tasmax,tasmin,pr -f day -rlzn 1 -skip _gr_ -yr 85 -dir $STORAGE_DIR/HadGEM3-GC31-LL/ssp245/ &> $STORAGE_DIR/HadGEM3-GC31-LL/ssp245/download.out
mkdir $STORAGE_DIR/HadGEM3-GC31-LL/ssp585/
acccmip6 -o D -m HadGEM3-GC31-LL -e ssp585 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3 -skip _gr_ -yr 85 -dir $STORAGE_DIR/HadGEM3-GC31-LL/ssp585/ &> $STORAGE_DIR/HadGEM3-GC31-LL/ssp585/download.out

mkdir $STORAGE_DIR/HadGEM3-GC31-MM/
mkdir $STORAGE_DIR/HadGEM3-GC31-MM/ssp585/
acccmip6 -o D -m HadGEM3-GC31-MM -e ssp585 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2 -skip _gr_ -yr 85 -dir $STORAGE_DIR/HadGEM3-GC31-MM/ssp585/ &> $STORAGE_DIR/HadGEM3-GC31-MM/ssp585/download.out

mkdir $STORAGE_DIR/INM-CM5-0/
mkdir $STORAGE_DIR/INM-CM5-0/ssp245/
acccmip6 -o D -m INM-CM5-0 -e ssp245 -v tas,tasmax,tasmin,pr -f day -rlzn 1 -skip _gr_ -yr 85 -dir $STORAGE_DIR/INM-CM5-0/ssp245/ &> $STORAGE_DIR/INM-CM5-0/ssp245/download.out
mkdir $STORAGE_DIR/INM-CM5-0/ssp370/
acccmip6 -o D -m INM-CM5-0 -e ssp370 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3,4,5 -skip _gr_ -yr 85 -dir $STORAGE_DIR/INM-CM5-0/ssp370/ &> $STORAGE_DIR/INM-CM5-0/ssp370/download.out
mkdir $STORAGE_DIR/INM-CM5-0/ssp585/
acccmip6 -o D -m INM-CM5-0 -e ssp585 -v tas,tasmax,tasmin,pr -f day -rlzn 1 -skip _gr_ -yr 85 -dir $STORAGE_DIR/INM-CM5-0/ssp585/ &> $STORAGE_DIR/INM-CM5-0/ssp585/download.out

mkdir $STORAGE_DIR/IPSL-CM6A-LR/
mkdir $STORAGE_DIR/IPSL-CM6A-LR/ssp245/
acccmip6 -o D -m IPSL-CM6A-LR -e ssp245 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3,4,5 -yr 85 -dir $STORAGE_DIR/IPSL-CM6A-LR/ssp245/ -n esgf.ceda.ac.uk &> $STORAGE_DIR/IPSL-CM6A-LR/ssp245/download.out
mkdir $STORAGE_DIR/IPSL-CM6A-LR/ssp370/
acccmip6 -o D -m IPSL-CM6A-LR -e ssp370 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3,4,5,6,7,8,9,10 -yr 85 -dir $STORAGE_DIR/IPSL-CM6A-LR/ssp370/ -n esgf.ceda.ac.uk &> $STORAGE_DIR/IPSL-CM6A-LR/ssp370/download.out
mkdir $STORAGE_DIR/IPSL-CM6A-LR/ssp585/
acccmip6 -o D -m IPSL-CM6A-LR -e ssp585 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3,4 -yr 85 -dir $STORAGE_DIR/IPSL-CM6A-LR/ssp585/ -n esgf.ceda.ac.uk &> $STORAGE_DIR/IPSL-CM6A-LR/ssp585/download.out

mkdir $STORAGE_DIR/KACE-1-0-G/
mkdir $STORAGE_DIR/KACE-1-0-G/ssp245/
acccmip6 -o D -m KACE-1-0-G -e ssp245 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3 -yr 85 -dir $STORAGE_DIR/KACE-1-0-G/ssp245/ &> $STORAGE_DIR/KACE-1-0-G/ssp245/download.out
mkdir $STORAGE_DIR/KACE-1-0-G/ssp370/
acccmip6 -o D -m KACE-1-0-G -e ssp370 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3 -yr 85 -dir $STORAGE_DIR/KACE-1-0-G/ssp370/ &> $STORAGE_DIR/KACE-1-0-G/ssp370/download.out
mkdir $STORAGE_DIR/KACE-1-0-G/ssp585/
acccmip6 -o D -m KACE-1-0-G -e ssp585 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3 -yr 85 -dir $STORAGE_DIR/KACE-1-0-G/ssp585/ &> $STORAGE_DIR/KACE-1-0-G/ssp585/download.out

mkdir $STORAGE_DIR/MIROC6/
mkdir $STORAGE_DIR/MIROC6/ssp245/
acccmip6 -o D -m MIROC6 -e ssp245 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3 -skip _gr_ -yr 85 -dir $STORAGE_DIR/MIROC6/ssp245/ &> $STORAGE_DIR/MIROC6/ssp245/download.out
mkdir $STORAGE_DIR/MIROC6/ssp370/
acccmip6 -o D -m MIROC6 -e ssp370 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3 -skip _gr_ -yr 85 -dir $STORAGE_DIR/MIROC6/ssp370/ &> $STORAGE_DIR/MIROC6/ssp370/download.out
mkdir $STORAGE_DIR/MIROC6/ssp585/
acccmip6 -o D -m MIROC6 -e ssp585 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3,4,5 -skip _gr_ -yr 85 -dir $STORAGE_DIR/MIROC6/ssp585/ &> $STORAGE_DIR/MIROC6/ssp585/download.out

mkdir $STORAGE_DIR/MPI-ESM1-2-HR/
mkdir $STORAGE_DIR/MPI-ESM1-2-HR/ssp245/
acccmip6 -o D -m MPI-ESM1-2-HR -e ssp245 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2 -skip _gr_ -yr 85 -dir $STORAGE_DIR/MPI-ESM1-2-HR/ssp245/ &> $STORAGE_DIR/MPI-ESM1-2-HR/ssp245/download.out
mkdir $STORAGE_DIR/MPI-ESM1-2-HR/ssp370/
acccmip6 -o D -m MPI-ESM1-2-HR -e ssp370 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3,4,5,6,7,8,9,10 -skip _gr_ -yr 85 -dir $STORAGE_DIR/MPI-ESM1-2-HR/ssp370/ &> $STORAGE_DIR/MPI-ESM1-2-HR/ssp370/download.out
mkdir $STORAGE_DIR/MPI-ESM1-2-HR/ssp585/
acccmip6 -o D -m MPI-ESM1-2-HR -e ssp585 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2 -skip _gr_ -yr 85 -dir $STORAGE_DIR/MPI-ESM1-2-HR/ssp585/ &> $STORAGE_DIR/MPI-ESM1-2-HR/ssp585/download.out

mkdir $STORAGE_DIR/MPI-ESM1-2-LR/
mkdir $STORAGE_DIR/MPI-ESM1-2-LR/ssp245/
acccmip6 -o D -m MPI-ESM1-2-LR -e ssp245 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3,4,5,6,7,8,10 -skip _gr_ -yr 85 -dir $STORAGE_DIR/MPI-ESM1-2-LR/ssp245/ &> $STORAGE_DIR/MPI-ESM1-2-LR/ssp245/download.out
mkdir $STORAGE_DIR/MPI-ESM1-2-LR/ssp370/
acccmip6 -o D -m MPI-ESM1-2-LR -e ssp370 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3,4,5,7,8,10 -skip _gr_ -yr 85 -dir $STORAGE_DIR/MPI-ESM1-2-LR/ssp370/ &> $STORAGE_DIR/MPI-ESM1-2-LR/ssp370/download.out
mkdir $STORAGE_DIR/MPI-ESM1-2-LR/ssp585/
acccmip6 -o D -m MPI-ESM1-2-LR -e ssp585 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3,4,5,6,7,8,10 -skip _gr_ -yr 85 -dir $STORAGE_DIR/MPI-ESM1-2-LR/ssp585/ &> $STORAGE_DIR/MPI-ESM1-2-LR/ssp585/download.out

mkdir $STORAGE_DIR/MRI-ESM2-0/
mkdir $STORAGE_DIR/MRI-ESM2-0/ssp245/
acccmip6 -o D -m MRI-ESM2-0 -e ssp245 -v tas,tasmax,tasmin,pr -f day -rlzn 1 -skip _gr_ -yr 85 -dir $STORAGE_DIR/MRI-ESM2-0/ssp245/ &> $STORAGE_DIR/MRI-ESM2-0/ssp245/download.out
mkdir $STORAGE_DIR/MRI-ESM2-0/ssp370/
acccmip6 -o D -m MRI-ESM2-0 -e ssp370 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3,4,5 -skip _gr_ -yr 85 -dir $STORAGE_DIR/MRI-ESM2-0/ssp370/ &> $STORAGE_DIR/MRI-ESM2-0/ssp370/download.out
mkdir $STORAGE_DIR/MRI-ESM2-0/ssp585/
acccmip6 -o D -m MRI-ESM2-0 -e ssp585 -v tas,tasmax,tasmin,pr -f day -rlzn 1 -skip _gr_ -yr 85 -dir $STORAGE_DIR/MRI-ESM2-0/ssp585/ &> $STORAGE_DIR/MRI-ESM2-0/ssp585/download.out

mkdir $STORAGE_DIR/NorESM2-LM/
mkdir $STORAGE_DIR/NorESM2-LM/ssp245/
acccmip6 -o D -m NorESM2-LM -e ssp245 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3 -skip _gr_,1f2 -yr 85 -dir $STORAGE_DIR/NorESM2-LM/ssp245/ &> $STORAGE_DIR/NorESM2-LM/ssp245/download.out
mkdir $STORAGE_DIR/NorESM2-LM/ssp370/
acccmip6 -o D -m NorESM2-LM -e ssp370 -v tas,tasmax,tasmin,pr -f day -rlzn 1 -skip _gr_ -yr 85 -dir $STORAGE_DIR/NorESM2-LM/ssp370/ &> $STORAGE_DIR/NorESM2-LM/ssp370/download.out
mkdir $STORAGE_DIR/NorESM2-LM/ssp585/
acccmip6 -o D -m NorESM2-LM -e ssp585 -v tas,tasmax,tasmin,pr -f day -rlzn 1 -skip _gr_ -yr 85 -dir $STORAGE_DIR/NorESM2-LM/ssp585/ &> $STORAGE_DIR/NorESM2-LM/ssp585/download.out

mkdir $STORAGE_DIR/NorESM2-MM/
mkdir $STORAGE_DIR/NorESM2-MM/ssp245/
acccmip6 -o D -m NorESM2-MM -e ssp245 -v tas,tasmax,tasmin,pr -f day -rlzn 1,2 -skip _gr_ -yr 85 -dir $STORAGE_DIR/NorESM2-MM/ssp245/ &> $STORAGE_DIR/NorESM2-MM/ssp245/download.out
mkdir $STORAGE_DIR/NorESM2-MM/ssp370/
acccmip6 -o D -m NorESM2-MM -e ssp370 -v tas,tasmax,tasmin,pr -f day -rlzn 1 -skip _gr_,1f2 -yr 85 -dir $STORAGE_DIR/NorESM2-MM/ssp370/ &> $STORAGE_DIR/NorESM2-MM/ssp370/download.out
mkdir $STORAGE_DIR/NorESM2-MM/ssp585/
acccmip6 -o D -m NorESM2-MM -e ssp585 -v tas,tasmax,tasmin,pr -f day -rlzn 1 -skip _gr_,1f2 -yr 85 -dir $STORAGE_DIR/NorESM2-MM/ssp585/ &> $STORAGE_DIR/NorESM2-MM/ssp585/download.out

## Historical
mkdir $STORAGE_DIR/ACCESS-CM2/historical
acccmip6 -o D -m ACCESS-CM2 -e historical -v tas,tasmax,tasmin,pr -f day -skip _gr_ -rlzn 1,2,3 -dir $STORAGE_DIR/ACCESS-CM2/historical/ &> $STORAGE_DIR/ACCESS-CM2/historical/download.out

mkdir $STORAGE_DIR/ACCESS-ESM1-5/historical
acccmip6 -o D -m ACCESS-ESM1-5 -e historical -v tas,tasmax,tasmin,pr -f day -skip _gr_ -rlzn 1,2,3,4,5 -dir $STORAGE_DIR/ACCESS-ESM1-5/historical/ &> $STORAGE_DIR/ACCESS-ESM1-5/historical/download.out

mkdir $STORAGE_DIR/AWI-CM-1-1-MR/historical
acccmip6 -o D -m AWI-CM-1-1-MR -e historical -v tas,tasmax,tasmin,pr -f day -skip _gr_ -rlzn 1,2,3,4,5 -dir $STORAGE_DIR/AWI-CM-1-1-MR/historical/ &> $STORAGE_DIR/AWI-CM-1-1-MR/historical/download.out

mkdir $STORAGE_DIR/CanESM5/historical
acccmip6 -o D -m CanESM5 -e historical -v tas,tasmax,tasmin,pr -f day -skip _gr_,1p2,1f2 -rlzn 1,2,3,4,5,6,7 -dir $STORAGE_DIR/CanESM5/historical/ &> $STORAGE_DIR/CanESM5/historical/download.out

mkdir $STORAGE_DIR/EC-Earth3/historical
acccmip6 -o D -m EC-Earth3 -e historical -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3,4 -n esgf3.dkrz.de -yr -65 -dir $STORAGE_DIR/EC-Earth3/historical/ &> $STORAGE_DIR/EC-Earth3/historical/download.out
acccmip6 -o D -m EC-Earth3 -e historical -v tasmin,tas -f day -rlzn 3 -yr -65 -n esgf.bsc.es -dir $STORAGE_DIR/EC-Earth3/historical/ &> $STORAGE_DIR/EC-Earth3/historical/download.out

mkdir $STORAGE_DIR/EC-Earth3-Veg/historical
acccmip6 -o D -m EC-Earth3-Veg -e historical -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3,4,5 -n esgf-data1.llnl.gov,esg-dn1.nsc.liu.se -dir $STORAGE_DIR/EC-Earth3-Veg/historical/ &> $STORAGE_DIR/EC-Earth3-Veg/historical/download.out

mkdir $STORAGE_DIR/FGOALS-g3/historical
acccmip6 -o D -m FGOALS-g3 -e historical -v tas,tasmax,tasmin,pr -f day -skip _gr_ -rlzn 1,3,4,5 -dir $STORAGE_DIR/FGOALS-g3/historical/ &> $STORAGE_DIR/FGOALS-g3/historical/download.out

mkdir $STORAGE_DIR/HadGEM3-GC31-LL/historical
acccmip6 -o D -m HadGEM3-GC31-LL -e historical -v tas,tasmax,tasmin,pr -f day -skip _gr_ -rlzn 1,2,3 -dir $STORAGE_DIR/HadGEM3-GC31-LL/historical/ &> $STORAGE_DIR/HadGEM3-GC31-LL/historical/download.out

mkdir $STORAGE_DIR/HadGEM3-GC31-MM/historical
acccmip6 -o D -m HadGEM3-GC31-MM -e historical -v tas,tasmax,tasmin,pr -f day -skip _gr_ -rlzn 1,2 -dir $STORAGE_DIR/HadGEM3-GC31-MM/historical/ &> $STORAGE_DIR/HadGEM3-GC31-MM/historical/download.out

mkdir $STORAGE_DIR/INM-CM5-0/historical
acccmip6 -o D -m INM-CM5-0 -e historical -v tas,tasmax,tasmin,pr -f day -skip _gr_ -rlzn 1,2,3,4,5 -dir $STORAGE_DIR/INM-CM5-0/historical/ &> $STORAGE_DIR/INM-CM5-0/historical/download.out

mkdir $STORAGE_DIR/IPSL-CM6A-LR/historical
acccmip6 -o D -m IPSL-CM6A-LR -e historical -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3,4,5,6,7,8,9,10 -dir $STORAGE_DIR/IPSL-CM6A-LR/historical/ &> $STORAGE_DIR/IPSL-CM6A-LR/historical/download.out

mkdir $STORAGE_DIR/KACE-1-0-G/historical
acccmip6 -o D -m KACE-1-0-G -e historical -v tas,tasmax,tasmin,pr -f day -rlzn 1,2,3 -dir $STORAGE_DIR/KACE-1-0-G/historical/ &> $STORAGE_DIR/KACE-1-0-G/historical/download.out

mkdir $STORAGE_DIR/MIROC6/historical
acccmip6 -o D -m MIROC6 -e historical -v tas,tasmax,tasmin,pr -f day -skip _gr_ -rlzn 1,2,3,4,5 -dir $STORAGE_DIR/MIROC6/historical/ &> $STORAGE_DIR/MIROC6/historical/download.out

mkdir $STORAGE_DIR/MPI-ESM1-2-HR/historical
acccmip6 -o D -m MPI-ESM1-2-HR -e historical -v tas,tasmax,tasmin,pr -f day -skip _gr_ -rlzn 1,2,3,4,5,6,7,8,9,10 -dir $STORAGE_DIR/MPI-ESM1-2-HR/historical/ &> $STORAGE_DIR/MPI-ESM1-2-HR/historical/download.out

mkdir $STORAGE_DIR/MPI-ESM1-2-LR/historical
acccmip6 -o D -m MPI-ESM1-2-LR -e historical -v tas,tasmax,tasmin,pr -f day -skip _gr_ -rlzn 1,2,3,4,5,6,7,8,10 -dir $STORAGE_DIR/MPI-ESM1-2-LR/historical/ &> $STORAGE_DIR/MPI-ESM1-2-LR/historical/download.out

mkdir $STORAGE_DIR/MRI-ESM2-0/historical
acccmip6 -o D -m MRI-ESM2-0 -e historical -v tas,tasmax,tasmin,pr -f day -skip _gr_ -rlzn 1,2,3,4,5 -dir $STORAGE_DIR/MRI-ESM2-0/historical/ &> $STORAGE_DIR/MRI-ESM2-0/historical/download.out

mkdir $STORAGE_DIR/NorESM2-LM/historical
acccmip6 -o D -m NorESM2-LM -e historical -v tas,tasmax,tasmin,pr -f day -skip _gr_ -rlzn 1,2,3 -dir $STORAGE_DIR/NorESM2-LM/historical/ &> $STORAGE_DIR/NorESM2-LM/historical/download.out

mkdir $STORAGE_DIR/NorESM2-MM/historical
acccmip6 -o D -m NorESM2-MM -e historical -v tas,tasmax,tasmin,pr -f day -skip _gr_ -rlzn 1,2 -dir $STORAGE_DIR/NorESM2-MM/historical/ &> $STORAGE_DIR/NorESM2-MM/historical/download.out


echo "Job Ended at `date`"