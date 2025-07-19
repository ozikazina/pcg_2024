#!/bin/bash
#SBATCH --job-name=
#SBATCH --partition=qgpu
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --account=

. ./loadModules.sh
cd release
make
nsys profile --force-overwrite=true --trace=cuda --output=nsysreport ./nbody4 131072 0.01 100 512 10 4096 128 "generated/131072.dat" out.dat