#!/bin/bash
# votess on one Raven general node, CPU path (device::cpu, its std::thread workers; driver built with icpx -fsycl,
# build_votess.sh): tesellate(), and the same plus our double volumes from its neighbour lists (OpenMP), at 72 and 1
# threads on every point file (1 untimed run for peak memory and volumes + 3 timed per variant; volumes written from
# the 72-thread run). Records (JSON lines) on stdout.
#   sbatch -o bench/libraries/results/votess-cpu-%j.jsonl -e /raven/u/uli/libbench-data/logs/votess-cpu-%j.err \
#          bench/libraries/run_votess_cpu.sh /raven/u/uli/libbench-data   (env SIZES, KINDS, THREADS select a subset)
#SBATCH -J libbench_votess
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=10:00:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load intel/2025.3
data="${1:-/raven/u/uli/libbench-data}"
V=/raven/u/uli/libbench-src/votess
# No OpenMP binding: with OMP_PROC_BIND=spread the OpenMP runtime pins the main thread to one core and votess's
# std::thread workers inherit that mask (72 threads ran at 1-thread speed, 71 s at 2.1M, job 32230805; geogram the same).
export OMP_PROC_BIND=false OMP_DYNAMIC=false
export VOTESS_VERSION="f0f805f (main 2025-07-15), icpx 2025.3 -fsycl, CPU path"
echo "# host $(hostname), nproc --all $(nproc --all), affinity $(taskset -cp $$)" >&2
declare -A shell=([bluenoise]=4 [jittered]=4 [uniform]=4 [clustered]=8 [halo]=8)
for n in ${SIZES:-2097152 16777216}; do
    for kind in ${KINDS:-jittered uniform clustered halo bluenoise}; do
        for t in ${THREADS:-72 1}; do
            [ "$t" = 1 ] && [ "$n" != 2097152 ] && continue  # single-core runs at 128^3 only
            export OMP_NUM_THREADS=$t
            vol=""
            [ "$t" = 72 ] && vol="--volumes=$data/volumes/votess-cpu_${kind}_${n}.f64"
            echo "# $(date +%T) $kind $n threads $t" >&2
            "$V/install/bin/votess_icpx" "$data/points/${kind}_${n}.pts" --device=cpu --threads=$t --shell=${shell[$kind]} \
                --repeats=3 $vol || echo "# FAILED $kind $n threads $t: exit $?" >&2
        done
    done
done
