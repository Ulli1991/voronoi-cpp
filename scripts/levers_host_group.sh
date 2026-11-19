#!/bin/bash
# A mesh_cases group of a CI build (sanitizers on) at 1 and 2 OpenMP threads, <repeats> times each; prints the cases
# that fail and the first sanitizer report:
#   sbatch -o build-jobs/logs/%x-%j.out scripts/levers_host_group.sh <ci build dir> <group> <repeats>
#SBATCH -J lev_group
#SBATCH -p interactive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:55:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/15
export OMP_PROC_BIND=false UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ASAN_OPTIONS=detect_leaks=1
dir="$1" group="$2" repeats="${3:-3}"
log="$dir/group-${SLURM_JOB_ID:-0}.log"
for threads in 1 2; do
    for ((r = 1; r <= repeats; ++r)); do
        OMP_NUM_THREADS=$threads "$dir/mesh_cases_test" "$group" >"$log" 2>&1
        rc=$?
        echo "threads $threads repeat $r exit $rc: last case $(grep '^case ' "$log" | tail -1)"
        ((rc)) && grep -m1 -A12 "runtime error" "$log"
    done
done
