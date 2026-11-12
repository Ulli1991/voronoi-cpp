#!/bin/bash
# Oracle volumes of the sampled cells of every point file (cached next to it), then the check of the given volumes.
#   sbatch -o bench/libraries/results/accuracy-<name>-%j.jsonl -e <log> bench/libraries/oracle_cpu.sh <data> [volumes...]
# With no volumes: fills the caches of all point files and the trajectory's last frame.
#SBATCH -J libbench_oracle
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=04:00:00
set -euo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14
export OMP_NUM_THREADS=72 OMP_PROC_BIND=spread OMP_PLACES=cores
data="$1"
shift
check=/raven/u/uli/libbench-src/oracle_check
if [ $# -eq 0 ]; then
    for p in "$data"/points/*_2097152.pts "$data/points/jittered_2097152_step0.03.trj@10" "$data"/points/*_16777216.pts; do
        "$check" "$p" 1000
    done
    exit 0
fi
# Volumes named <library>-<device>_<input>_<n>[_step<k>].f64 (k: the trajectory frame): grouped by their point file.
for v in "$@"; do
    base=$(basename "$v" .f64)
    rest=${base#*_}
    case "$rest" in
        *_step[0-9]*) p="$data/points/${rest%_step*}_step0.03.trj@${rest##*_step}" ;;
        *) p="$data/points/${rest}.pts" ;;
    esac
    "$check" "$p" 1000 "$v" || echo "# oracle_check failed on $v: exit $?" >&2
done
