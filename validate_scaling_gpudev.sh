#!/bin/bash
#SBATCH --job-name=voronoi-scaling
#SBATCH --partition=gpudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:15:00
#SBATCH --output=build-table-validation/scaling-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
validation_root="$PWD/build-table-validation"
export LD_LIBRARY_PATH="/mpcdf/soft/SLE_15/packages/x86_64/gcc/15.1.0/lib64:/mpcdf/soft/SLE_15/packages/x86_64/cuda/13.0.1/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
for side in 16 24 32; do
    srun --ntasks=1 "$validation_root/cpu/table_mesh_check" "$validation_root/cpu-faces-$side.csv" "$side"
    srun --ntasks=1 "$validation_root/gpu/table_mesh_check" "$validation_root/gpu-faces-$side.csv" "$side"
    python3 compare_faces.py "$validation_root/cpu-faces-$side.csv" "$validation_root/gpu-faces-$side.csv"
done
