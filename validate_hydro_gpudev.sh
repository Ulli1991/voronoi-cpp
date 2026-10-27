#!/bin/bash
#SBATCH --job-name=voronoi-hydro-profile
#SBATCH --partition=gpudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:15:00
#SBATCH --output=build-table-validation/hydro-%j.out

# Prepared for review. Submit only after the user approves this test job.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
validation_root="$PWD/build-hydro-validation/${SLURM_JOB_ID:?}"
gcc_cpu=/mpcdf/soft/SLE_15/packages/x86_64/gcc/15.1.0
gcc_gpu=/mpcdf/soft/SLE_15/packages/x86_64/gcc/14.1.0
cuda_root=/mpcdf/soft/SLE_15/packages/x86_64/cuda/13.0.1
export PATH="$cuda_root/bin:$gcc_gpu/bin:$PATH"
export LD_LIBRARY_PATH="$gcc_cpu/lib64:$gcc_gpu/lib64:$cuda_root/lib64:${LD_LIBRARY_PATH:-}"
export NVCC_WRAPPER_DEFAULT_COMPILER="$gcc_gpu/bin/g++"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

hostname
nvidia-smi --query-gpu=name,driver_version --format=csv
cmake -S . -B "$validation_root/cpu" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="$gcc_cpu/bin/g++" \
    -DKokkos_DIR=/u/uli/kokkos-gcc15/lib64/cmake/Kokkos
cmake --build "$validation_root/cpu" -j 4
cmake -S . -B "$validation_root/gpu" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper \
    -DKokkos_DIR=/u/uli/kokkos-cuda/lib64/cmake/Kokkos
cmake --build "$validation_root/gpu" -j 4
for backend in cpu gpu; do
    srun --ntasks=1 "$validation_root/$backend/hydro_geometry_check"
done
for side in 12 16 32; do
    for backend in cpu gpu; do
        srun --ntasks=1 "$validation_root/$backend/table_mesh_check" \
            "$validation_root/$backend-faces-$side.csv" "$side"
    done
    python3 compare_faces.py "$validation_root/cpu-faces-$side.csv" "$validation_root/gpu-faces-$side.csv"
    python3 compare_faces.py "$validation_root/cpu-faces-$side.csv.hydro.csv" "$validation_root/gpu-faces-$side.csv.hydro.csv"
done
