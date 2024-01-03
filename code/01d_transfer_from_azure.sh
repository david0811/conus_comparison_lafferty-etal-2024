#!/bin/bash

# make sure executable is in path
export PATH=$PATH:/storage/home/dcl5300/azcopy_linux_amd64_10.22.1

# Transfer CIL-GDPCIR metrics from MPC to local storage
OUT_PATH="/storage/group/pches/default/users/dcl5300/conus_comparison_lafferty-etal-2024/metrics/CIL-GDPCIR" # where to store metrics

# CIL-GDPCIR 
azcopy copy "https://mpctransfer.blob.core.windows.net/conus-comparison/" "${OUT_PATH}/" --recursive
