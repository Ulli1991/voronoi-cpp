#!/bin/bash
# One host test program of a CI build (scripts/ci_host_general.sh) with its arguments, OpenMP at 2 threads, UBSan halting:
#   sbatch -o build-jobs/logs/%x-%j.out scripts/levers_host_case.sh <program> [args...]
#SBATCH -J lev_case
#SBATCH -p interactive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/15
export OMP_NUM_THREADS=2 OMP_PROC_BIND=false UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ASAN_OPTIONS=detect_leaks=1
"$@" 2>&1 | tail -60
echo "exit ${PIPESTATUS[0]}"
