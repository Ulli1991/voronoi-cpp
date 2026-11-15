#!/bin/bash
# The shared inputs of the library benchmark: point files of the voronoi-cpp generators (src/test/support.hpp,
# fixed seeds) and moving-mesh trajectories. Uses the GPU only for the slot order of the trajectory.
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
# KINDS selects inputs; fccsolid has 4 l³ atoms, so it runs at l = 64 and 128 unit cells (2^20 and 2^23 atoms).
for kind in ${KINDS:-jittered uniform clustered halo cosmicweb surface fccsolid}; do
    sizes="2097152 16777216"
    [ "$kind" = fccsolid ] && sizes="1048576 8388608"
    for n in $sizes; do
        "$build/library_bench" points "$n" "$kind" "$data/points/${kind}_${n}.pts"
    done
done
# 10 steps of 0.03 mean spacings of the bench flow, in slot order (as voronoi_bench step).
for kind in ${STEP_KINDS:-jittered cosmicweb surface fccsolid}; do
    n=2097152
    [ "$kind" = fccsolid ] && n=1048576
    "$build/library_bench" trajectory $n "$kind" 0.03 10 "$data/points/${kind}_${n}_step0.03.trj"
done
sha256sum "$data"/points/*.pts "$data"/points/*.trj
