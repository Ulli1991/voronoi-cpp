#!/bin/bash
# CI's host steps on one general node: comment rules, format of changed lines, build with sanitizers off and on, ctest,
# clang-tidy 18 on the given sources (host Kokkos 5.0.99 OpenMP at 2 threads, where CI builds a Serial Kokkos 5.0.2):
#   sbatch -o build-jobs/%x-%j.out scripts/ci_host_general.sh <base ref> <source>...
#SBATCH -J ci_host
#SBATCH -p general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=120G
#SBATCH --time=03:00:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/15 cmake
base="$1"
shift
export OMP_NUM_THREADS=2 OMP_PROC_BIND=false UBSAN_OPTIONS=halt_on_error=1 ASAN_OPTIONS=detect_leaks=1
python3 scripts/check_comments.py && echo "comments: ok"
PATH=/mpcdf/soft/SLE_15/packages/x86_64/clang/18.1.8/bin:$PATH python3 scripts/check_format.py "$base" && echo "format: ok"
for san in OFF ON; do
    dir=build-jobs/ci-$san
    cmake -S . -B $dir -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DVORONOI_SANITIZERS=$san \
        -DKokkos_DIR=/u/uli/kokkos-gcc15/lib64/cmake/Kokkos >/dev/null
    cmake --build $dir --parallel 36 2>&1 | grep -E "error|warning: " | head -20
    echo "== ctest sanitizers $san"
    ctest --test-dir $dir --output-on-failure --parallel 8 2>&1 | tail -30
done
echo "== clang-tidy 18"
for source in "$@"; do
    /mpcdf/soft/SLE_15/packages/x86_64/clang/18.1.8/bin/clang-tidy -p build-jobs/ci-OFF "$source" 2>&1 |
        grep -E "warning:|error:" | grep -v "Kokkos\|/u/uli/kokkos" | head -40
    echo "clang-tidy done: $source"
done
