#!/bin/bash
#SBATCH --job-name=voronoi-reference-profile
#SBATCH --partition=gpu1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:15:00
#SBATCH --output=build-table-validation/reference-profile-%j.out
# Ask before submission. All outputs are isolated by job ID.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
profile_root="$PWD/build-roofline/${SLURM_JOB_ID:?}"
gcc_gpu=/mpcdf/soft/SLE_15/packages/x86_64/gcc/14.1.0
cuda_root=/mpcdf/soft/SLE_15/packages/x86_64/cuda/13.0.1
export PATH="$cuda_root/bin:$gcc_gpu/bin:$PATH"
export LD_LIBRARY_PATH="$gcc_gpu/lib64:$cuda_root/lib64:${LD_LIBRARY_PATH:-}"
export NVCC_WRAPPER_DEFAULT_COMPILER="$gcc_gpu/bin/g++"
export OMP_NUM_THREADS=1
module load nsight_compute/2026
hostname
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.free --format=csv
ncu --version
cmake -S . -B "$profile_root/build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS=-lineinfo \
    -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper \
    -DKokkos_DIR=/u/uli/kokkos-cuda/lib64/cmake/Kokkos
cmake --build "$profile_root/build" --target construction_scaling -j 4
exe="$profile_root/build/construction_scaling"
# Ordinary throughput and fenced stage measurements are collected separately.
for side in 128 195; do
    for policy in relative_early relative_early_scratch_sort; do
        srun "$exe" "$side" 5 "$policy" "$profile_root/timings.csv" "$profile_root/search-stages.csv"
    done
done
# Profile one warmed, nonempty reference reconstruction kernel. Retry cost is
# included in ordinary timings, but this capture isolates the main kernel.
srun ncu --target-processes all --kernel-name-base demangled \
    --kernel-name 'regex:reconstruct_hydro<' --launch-skip 2 --launch-count 1 \
    --section LaunchStats --section Occupancy --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Tables --section WarpStateStats \
    --section SourceCounters --export "$profile_root/reference-128" \
    "$exe" 128 1 relative_early_scratch_sort "$profile_root/instrumented-only.csv"
ncu --import "$profile_root/reference-128.ncu-rep" --page raw > "$profile_root/reference-128-details.txt"
ncu --import "$profile_root/reference-128.ncu-rep" --page source > "$profile_root/reference-128-source.txt"
python3 - "$profile_root/reference-128-details.txt" <<'CHECK'
import pathlib
import sys
text = pathlib.Path(sys.argv[1]).read_text()
if 'reconstruct_hydro<' not in text or 'ParallelScan<' in text or '(!) nan' in text:
    raise SystemExit('Invalid reconstruction capture; inspect the report before using it')
if 'launch__registers_per_thread' not in text or 'local' not in text:
    raise SystemExit('Missing register/local-memory counters')
print('Confirmed main reconstruction capture')
CHECK
