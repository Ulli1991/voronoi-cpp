#!/bin/bash
# CI's host build with sanitizers on (gcc 15, Kokkos OpenMP) of a checkout, then its ctest as CI runs it, <repeats> times:
#   sbatch -o build-jobs/logs/%x-%j.out scripts/levers_ci_on_repeat.sh <checkout> <repeats>
#SBATCH -J lev_ci_on
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
src="$1" repeats="${2:-2}"
dir="$src/build-jobs/ci-ON"
cmake -S "$src" -B "$dir" -DCMAKE_BUILD_TYPE=Release -DVORONOI_SANITIZERS=ON -DKokkos_DIR=/u/uli/kokkos-gcc15/lib64/cmake/Kokkos >/dev/null
cmake --build "$dir" --parallel 72 2>&1 | grep -E " error" | head
echo "built $(git -C "$src" log --oneline -1 | cut -c1-60)"
for ((r = 1; r <= repeats; ++r)); do
    echo "== repeat $r"
    ctest --test-dir "$dir" --output-on-failure --parallel 8 2>&1 | grep -E "runtime error|tests passed|\(Failed\)|Timeout"
done
