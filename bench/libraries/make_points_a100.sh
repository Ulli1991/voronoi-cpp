#!/bin/bash
# The shared inputs of the library benchmark: point files of the voronoi-cpp generators (src/test/support.hpp,
# fixed seeds) and one moving-mesh trajectory. Uses the GPU only for the slot order of the trajectory.
#   sbatch -o /raven/u/uli/libbench-data/logs/points-%j.out bench/libraries/make_points_a100.sh <build dir> <data dir>
#SBATCH -J libbench_points
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
mkdir -p "$data/points"
for n in 2097152 16777216; do
    for kind in jittered uniform clustered halo; do
        "$build/library_bench" points "$n" "$kind" "$data/points/${kind}_${n}.pts"
    done
done
# 10 steps of 0.03 mean spacings of the bench flow, in slot order (as voronoi_bench step).
"$build/library_bench" trajectory 2097152 jittered 0.03 10 "$data/points/jittered_2097152_step0.03.trj"
sha256sum "$data"/points/*.pts "$data"/points/*.trj
