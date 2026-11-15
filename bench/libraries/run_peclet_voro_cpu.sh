#!/bin/bash
# peclet-voro on one full Raven CPU node (Kokkos OpenMP): as run_peclet_voro_a100.sh at 72 threads (both sizes, both
# build variants, the trajectory steps), and the cold build at 1 thread at 2097152 only (16777216 at one thread
# would take about 5 min per run, 25 min per input: left out).
#   sbatch -o bench/libraries/results/peclet-voro-cpu-%j.jsonl -e /raven/u/uli/libbench-data/logs/peclet-voro-cpu-%j.err \
#          bench/libraries/run_peclet_voro_cpu.sh /raven/u/uli/libbench-data
#SBATCH -J libbench_peclet_cpu
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=02:00:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14
data="${1:-/raven/u/uli/libbench-data}"
bin=/raven/u/uli/libbench-src/peclet-voro/install/bin/peclet_voro_openmp
export PECLET_VORO_VERSION="v1.0.4-1-gd7371c9 (d7371c9), Kokkos 5.1.1" OMP_PROC_BIND=spread OMP_PLACES=cores
lscpu | grep "Model name" >&2
export OMP_NUM_THREADS=72
for n in ${SIZES:-2097152 16777216}; do
    for kind in ${KINDS:-jittered uniform clustered halo bluenoise}; do
        "$bin" build "$data/points/${kind}_${n}.pts" --variant=plain --repeats=3 \
            --volumes="$data/volumes/peclet-voro-cpu_${kind}_${n}.f64" || echo "# FAILED plain $kind $n" >&2
        "$bin" build "$data/points/${kind}_${n}.pts" --variant=resident --repeats=3 || echo "# FAILED resident $kind $n" >&2
    done
done
if [ -z "${SKIP_STEP:-}" ]; then
    "$bin" step "$data/points/jittered_2097152_step0.03.trj" \
        --volumes="$data/volumes/peclet-voro-cpu_jittered_2097152_step10.f64" || echo "# FAILED step" >&2
fi
export OMP_NUM_THREADS=1
for kind in ${KINDS:-jittered uniform clustered halo bluenoise}; do
    [ -f "$data/points/${kind}_2097152.pts" ] || continue
    "$bin" build "$data/points/${kind}_2097152.pts" --variant=plain --repeats=3 || echo "# FAILED 1 thread $kind" >&2
done
