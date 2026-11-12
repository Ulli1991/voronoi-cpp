#!/bin/bash
# geogram in the library benchmark on one Raven general node: every point file at 72 and 1 threads,
#   pdel              PeriodicDelaunay3d, native periodic, Delaunay only
#   pdel-cells        PeriodicDelaunay3d + Voronoi cells (volumes, centroids, faces with neighbour and area)
#   pdel-nonper       ParallelDelaunay3d (PDEL) on the points padded with periodic images, Delaunay only
#   pdel-nonper-dual  the same + our dual volumes
# each 1 untimed run (peak memory, volumes) + 3 timed. Records (JSON lines) on stdout.
#   sbatch -o bench/libraries/results/geogram-cpu-%j.jsonl -e /raven/u/uli/libbench-data/logs/geogram-cpu-%j.err \
#       bench/libraries/run_geogram_cpu.sh [sizes] [inputs] [threads]
#SBATCH -J libbench_geogram
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=08:00:00
set -euo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14
root=/raven/u/uli/libbench-src/geogram
data=/raven/u/uli/libbench-data
sizes="${1:-2097152 16777216}"
inputs="${2:-jittered uniform clustered halo}"
threadlist="${3:-72 1}"
export GEOGRAM_COMMIT="$(git -C "$root/src" rev-parse --short=12 HEAD)"
# No OpenMP binding: with OMP_PROC_BIND=spread geogram's threads ran on one core (18 threads 20.8 s against 2.6 s
# unbound, test job 32226520), so the library's own thread pool is left to the scheduler on the exclusive node.
export OMP_PROC_BIND=false OMP_DYNAMIC=false
echo "# host $(hostname), nproc $(nproc), $(lscpu | grep -E '^(Model name|Thread\(s\) per core|CPU\(s\)):' | tr -s ' ' | paste -sd';')" >&2
for n in $sizes; do
    # Padding for the non-periodic variant: 6 mean spacings (checked per cell in the untimed run).
    width=$(awk -v n=$n 'BEGIN { printf "%.6f", 6.0 / exp(log(n) / 3) }')
    for kind in $inputs; do
        pts="$data/points/${kind}_${n}.pts"
        [ -f "$pts" ] || { echo "# missing $pts" >&2; continue; }
        for t in $threadlist; do
            export OMP_NUM_THREADS=$t
            for v in pdel pdel-cells pdel-nonper pdel-nonper-dual; do
                vol=""
                if [ "$t" = 72 ]; then
                    case $v in
                        pdel-cells) vol="--volumes=$data/volumes/geogram-pdel-cpu_${kind}_${n}.f64" ;;
                        pdel-nonper-dual) vol="--volumes=$data/volumes/geogram-pdel-nonper-cpu_${kind}_${n}.f64" ;;
                    esac
                fi
                echo "# $(date +%T) $v $kind $n threads $t" >&2
                "$root/geogram_bench" "$pts" --variant=$v --threads=$t --repeats=3 --width=$width $vol ||
                    echo "# FAILED $v $kind $n threads $t: exit $?" >&2
            done
        done
    done
done
