#!/bin/bash
# gDel3D in the library benchmark on one A100: per point file (2097152 points; jittered, uniform, clustered, halo) the
# variants "Delaunay" and "Delaunay + dual volumes (ours)", 1 untimed run + 3 timed each. 16777216 is attempted once
# (jittered) to record the failure: gDel3D stores a tetrahedron index in 27 bits of TetOpp (index << 5), so at most
# 2^26 tetrahedra; it allocates 8.5 x points (16.8M + images: about 190M).
#   sbatch -o bench/libraries/results/gdel3d-a100-%j.jsonl -e /raven/u/uli/libbench-data/logs/gdel3d-a100-%j.err \
#          bench/libraries/run_gdel3d_a100.sh /raven/u/uli/libbench-data [extra driver options, e.g. --width=10]
#SBATCH -J libbench_gdel3d
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/13 cuda/12.6
data="$1"; shift
bin=/raven/u/uli/libbench-src/gdel3d/install/bin/gdel3d
export OMP_NUM_THREADS=18 OMP_PROC_BIND=spread OMP_PLACES=cores
export LIBBENCH_LIB_VERSION="a429be8 + gdel3d.patch"
clocks() { nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu --format=csv,noheader | sed 's/^/# clocks: /' >&2; }
nvidia-smi -L >&2
clocks
for kind in ${KINDS:-jittered uniform clustered halo bluenoise}; do
    "$bin" "$data/points/${kind}_2097152.pts" --repeats=3 --volumes="$data/volumes/gdel3d-a100_${kind}_2097152.f64" "$@" \
        || echo "# gdel3d ${kind} 2097152 failed: exit $?" >&2
    clocks
done
if [ -z "${SKIP_16M:-}" ]; then
    "$bin" "$data/points/jittered_16777216.pts" --repeats=1 "$@" || echo "# gdel3d jittered 16777216 failed: exit $?" >&2
    clocks
fi
