#!/bin/bash
#SBATCH --job-name=voronoi-roofline
#SBATCH --partition=gpudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:15:00
#SBATCH --output=build-table-validation/roofline-%j.out
# Ask the user before submitting; Nsight Compute may need GPU performance counters.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
root="$PWD/build-roofline/${SLURM_JOB_ID:?}"
gcc_gpu=/mpcdf/soft/SLE_15/packages/x86_64/gcc/14.1.0
cuda_root=/mpcdf/soft/SLE_15/packages/x86_64/cuda/13.0.1
export PATH="$cuda_root/bin:$gcc_gpu/bin:$PATH"
export LD_LIBRARY_PATH="$gcc_gpu/lib64:$cuda_root/lib64:${LD_LIBRARY_PATH:-}"
export NVCC_WRAPPER_DEFAULT_COMPILER="$gcc_gpu/bin/g++"
export OMP_NUM_THREADS=1
module load nsight_compute/2026
ncu --version
cmake -S . -B "$root/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper \
  -DKokkos_DIR=/u/uli/kokkos-cuda/lib64/cmake/Kokkos
cmake --build "$root/build" --target construction_scaling -j 4
exe="$root/build/construction_scaling"
metrics=dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warps_eligible.avg.per_cycle_active,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
for policy in relative relative_early; do
  # Profile the first nonempty reconstruction kernel after setup/search and warmup.
  srun ncu --metrics "$metrics" --target-processes all --kernel-name-base demangled \
    --kernel-name 'regex:reconstruct_hydro<' --launch-skip 2 --launch-count 1 \
    --force-overwrite --export "$root/${policy}-32" "$exe" 32 1 "$policy" "$root/${policy}.csv"
done
srun ncu --import "$root/relative-32.ncu-rep" --page raw > "$root/relative-32-details.txt"
srun ncu --import "$root/relative_early-32.ncu-rep" --page raw > "$root/relative_early-32-details.txt"
python3 - "$root" <<'CHECK'
import pathlib, sys
for p in pathlib.Path(sys.argv[1]).glob('*-details.txt'):
    text = p.read_text()
    if 'reconstruct_hydro<' not in text or 'ParallelScan<' in text or '(!) nan' in text:
        raise SystemExit(f'Invalid reconstruction capture: {p}')
    if 'dram__bytes_read.sum' not in text:
        raise SystemExit(f'Missing hardware counters: {p}')
    print(f'Confirmed reconstruction capture: {p}')
CHECK
echo "Nsight Compute reports: $root"
