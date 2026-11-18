#!/bin/bash
# ctest of a CI build (scripts/ci_host_general.sh, sanitizers on), as CI runs it (8 tests at a time, OpenMP at 2 threads),
# <repeats> times, for failures that come and go between runs:
#   sbatch -o build-jobs/logs/%x-%j.out scripts/levers_ctest_repeat.sh <ci build dir> <repeats>
#SBATCH -J lev_ctest_rep
#SBATCH -p general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:30:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/15 cmake
export OMP_NUM_THREADS=2 OMP_PROC_BIND=false UBSAN_OPTIONS=halt_on_error=1 ASAN_OPTIONS=detect_leaks=1
dir="$1" repeats="${2:-3}"
for ((r = 1; r <= repeats; ++r)); do
    echo "== repeat $r"
    ctest --test-dir "$dir" --output-on-failure --parallel 8 2>&1 | grep -E "runtime error|tests passed|\(Failed\)|Timeout"
done
