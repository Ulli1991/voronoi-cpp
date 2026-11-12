#!/bin/bash
# voro++ in the library benchmark on one full general node: master (serial, container_periodic) at 1 thread, and the
# multithreaded dev branch (Lu, Lazar, Rycroft 2023) at 72 and 1 threads, on every point file; 1 untimed run (peak
# memory, volumes) + 3 timed. Records (JSON lines) on stdout.
#   sbatch -o bench/libraries/results/voropp-cpu-%j.jsonl -e /raven/u/uli/libbench-data/logs/voropp-cpu-%j.err \
#       bench/libraries/run_voropp_cpu.sh [data dir]
#SBATCH -J libbench_voropp
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
set -euo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14
data="${1:-/raven/u/uli/libbench-data}"
bin=/raven/u/uli/libbench-src/voropp/install/bin
export OMP_PROC_BIND=spread OMP_PLACES=cores
lscpu | grep -E 'Model name|^CPU\(s\)|MHz' >&2
for n in ${SIZES:-2097152 16777216}; do
    for kind in ${KINDS:-jittered uniform clustered halo bluenoise}; do
        pts="$data/points/${kind}_${n}.pts"
        # voro++ master's container_periodic exits on two points closer than 1e-5 (check_duplicate, common.cc).
        OMP_NUM_THREADS=1 "$bin/voropp_bench" "$pts" --volumes="$data/volumes/voropp-cpu_${kind}_${n}.f64" ||
            echo "# FAILED voropp serial $kind $n: exit $?" >&2
        OMP_NUM_THREADS=72 "$bin/voropp_mt_bench" "$pts" --threads=72 \
            --volumes="$data/volumes/voropp-mt-cpu_${kind}_${n}.f64" || echo "# FAILED voropp mt 72 $kind $n: exit $?" >&2
        OMP_NUM_THREADS=1 "$bin/voropp_mt_bench" "$pts" --threads=1 || echo "# FAILED voropp mt 1 $kind $n: exit $?" >&2
    done
done
