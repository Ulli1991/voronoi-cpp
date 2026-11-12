#!/bin/bash
# CGAL in the library benchmark on one full Raven general node (2x Xeon Platinum 8360Y, 72 cores): per point file,
# Periodic_3_Delaunay_triangulation_3 (serial) and the padded Delaunay_triangulation_3 with Parallel_tag at 72 and
# 1 threads, each "Delaunay" and "Delaunay + faces (ours)", 1 untimed run (peak memory, volumes) + 3 timed;
# then the P3DT moving mesh on the trajectory. Records (JSON lines) on stdout.
#   sbatch -o bench/libraries/results/cgal-cpu-%j.jsonl -e /raven/u/uli/libbench-data/logs/cgal-cpu-%j.err \
#       bench/libraries/run_cgal_cpu.sh [sizes] [inputs]
#SBATCH -J libbench_cgal
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=12:00:00
set -euo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14
# No OpenMP binding: it would pin the main thread, and the TBB workers created after it, to one core.
export OMP_PROC_BIND=false
bin=/raven/u/uli/libbench-src/cgal/bin/cgal_bench
data=/raven/u/uli/libbench-data
sizes="${1:-2097152 16777216}"
inputs="${2:-jittered uniform clustered halo bluenoise}"
lscpu | grep -E "Model name|^CPU\(s\)" >&2
for n in $sizes; do
    for kind in $inputs; do
        pts="$data/points/${kind}_${n}.pts"
        [ -f "$pts" ] || { echo "# missing $pts" >&2; continue; }
        echo "# $(date +%T) p3dt $kind $n" >&2
        "$bin" p3dt "$pts" --repeats=3 --volumes="$data/volumes/cgal-p3dt-cpu_${kind}_${n}.f64" || echo "# p3dt $kind $n failed: $?" >&2
        for threads in 72 1; do
            echo "# $(date +%T) dt3 $threads threads $kind $n" >&2
            vol=""
            [ "$threads" = 72 ] && vol="--volumes=$data/volumes/cgal-dt3-par-cpu_${kind}_${n}.f64"
            "$bin" dt3 "$pts" --threads=$threads --repeats=3 $vol || echo "# dt3 $threads $kind $n failed: $?" >&2
        done
    done
done
if [ "${STEP:-1}" = 1 ]; then
    echo "# $(date +%T) step" >&2
    "$bin" step "$data/points/jittered_2097152_step0.03.trj" --steps=10 --budget=600 \
        --volumes="$data/volumes/cgal-p3dt-cpu_jittered_2097152"
fi
echo "# $(date +%T) done" >&2
