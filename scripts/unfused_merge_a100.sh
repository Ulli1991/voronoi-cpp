#!/bin/bash
# unfused-on-main after merging origin/main (fbe6a0c), on one gpu1 node, from the cpu-overlap repository root:
#   sbatch -o build-overlap/jobs/%x-%j.out scripts/unfused_merge_a100.sh <unfused checkout>
# CUDA and OpenMP (gcc 15) builds, A100 ctest (repro.serial: CUDA against Serial), CUDA against OpenMP per cell
# (voronoi_bench --dump-cells, 128³, uniform and cosmicweb), this branch's modes_test again.
#SBATCH -J unfused_merge
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=02:00:00
set -uo pipefail
source /etc/profile.d/modules.sh
here="$(pwd)"
src=$(realpath "$1")
out="$here/build-overlap/jobs/$SLURM_JOB_ID"
mkdir -p "$out/dump"
hostname
module purge; module load gcc/14 cuda/13.0 cmake
cmake -S "$src" -B "$out/cuda" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/u/uli/kokkos-cuda \
    -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper -DCMAKE_CXX_EXTENSIONS=OFF >"$out/cuda.configure.log" 2>&1
cmake --build "$out/cuda" -j16 >"$out/cuda.build.log" 2>&1
echo "build cuda exit $? errors $(grep -c ' error' "$out/cuda.build.log")"
export OMP_PROC_BIND=spread OMP_PLACES=cores OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ctest --test-dir "$out/cuda" --output-on-failure >"$out/ctest_cuda.log" 2>&1
echo "ctest cuda: $(grep -E 'tests passed|tests failed' "$out/ctest_cuda.log")"
grep -E "^ *[0-9]+ - " "$out/ctest_cuda.log"
"$out/cuda/mesh_cases_test" repro.serial >"$out/repro_serial.log" 2>&1
echo "repro.serial on CUDA exit $?: $(grep -c '^ *ok' "$out/repro_serial.log") ok, $(grep -c '^ *FAIL' "$out/repro_serial.log") FAIL"
grep -E "FAIL|skip" "$out/repro_serial.log" | head
for kind in uniform cosmicweb; do
    "$out/cuda/voronoi_bench" step 2097152 "$kind" 0.1 3 --dump-cells="$out/dump/cuda-$kind" >"$out/dump/cuda-$kind.log" 2>&1
done
"$here/build-overlap/jobs/32413666/branch-both/modes_test" >"$out/modes.log" 2>&1
echo "modes_test (cpu-overlap) exit $?: $(grep -c '^ *ok' "$out/modes.log") ok, $(grep -c '^ *FAIL' "$out/modes.log") FAIL"
module purge; module load gcc/15 cmake
cmake -S "$src" -B "$out/omp" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/u/uli/kokkos-gcc15 >"$out/omp.configure.log" 2>&1
cmake --build "$out/omp" --target voronoi_bench -j16 >"$out/omp.build.log" 2>&1
echo "build omp exit $?"
for kind in uniform cosmicweb; do
    "$out/omp/voronoi_bench" step 2097152 "$kind" 0.1 3 --dump-cells="$out/dump/omp-$kind" >"$out/dump/omp-$kind.log" 2>&1
done
for kind in uniform cosmicweb; do
    if cmp -s "$out/dump/cuda-$kind.geometry.npy" "$out/dump/omp-$kind.geometry.npy" &&
        cmp -s "$out/dump/cuda-$kind.neighbours.npy" "$out/dump/omp-$kind.neighbours.npy"; then
        echo "$kind: CUDA and OpenMP dumps identical"
    else
        echo "$kind: CUDA and OpenMP dumps DIFFER"
    fi
done
