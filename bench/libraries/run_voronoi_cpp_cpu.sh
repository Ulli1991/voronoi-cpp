#!/bin/bash
# voronoi-cpp (Kokkos OpenMP backend) in the library benchmark on one full CPU node, at 72 threads and 1 thread:
# warp rebuild and exact start on the 2.1M point files (16.8M at 72 threads), then the moving-mesh trajectory.
#   sbatch -o bench/libraries/results/voronoi-cpp-cpu-%j.jsonl -e <log> bench/libraries/run_voronoi_cpp_cpu.sh <build> <data>
#SBATCH -J libbench_vcpp_cpu
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=02:00:00
set -euo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/15
build="$1"
data="$2"
export LIBBENCH_COMMIT="$(git -C "$SLURM_SUBMIT_DIR" rev-parse --short HEAD)$(git -C "$SLURM_SUBMIT_DIR" diff --quiet HEAD -- src bench || echo +local)"
export OMP_PROC_BIND=spread OMP_PLACES=cores
lscpu | grep -E "Model name|MHz" >&2
for threads in 72 1; do
    export OMP_NUM_THREADS=$threads
    sizes="2097152"
    [ "$threads" = 72 ] && sizes="2097152 16777216"
    for n in $sizes; do
        for kind in jittered uniform clustered halo; do
            for path in warp exact; do
                "$build/library_bench" build "$data/points/${kind}_${n}.pts" --path=$path --repeats=3 \
                    --volumes="$data/volumes/voronoi-cpp-${path}-cpu${threads}_${kind}_${n}.f64"
            done
        done
    done
    "$build/library_bench" step "$data/points/jittered_2097152_step0.03.trj" \
        --volumes="$data/volumes/voronoi-cpp-step-cpu${threads}_jittered_2097152_step10.f64"
done
