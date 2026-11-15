#!/bin/bash
# Blue-noise inputs of the library benchmark: Lloyd iterations (centroidal Voronoi tessellation, as the blue noise of
# Ray et al. 2018 and Basselin et al. 2021) from the uniform point files, in the periodic unit box.
#   sbatch -o <log> bench/libraries/make_bluenoise_a100.sh <build dir> <data dir> [iterations]
#SBATCH -J libbench_blue
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -euo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0
build="$1"
data="$2"
iterations="${3:-100}"
for n in 2097152 16777216; do
    "$build/library_bench" lloyd "$data/points/uniform_${n}.pts" "$iterations" "$data/points/bluenoise_${n}.pts"
done
sha256sum "$data"/points/bluenoise_*.pts
