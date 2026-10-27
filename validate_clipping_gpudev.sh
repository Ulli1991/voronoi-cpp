#!/bin/bash
#SBATCH --job-name=voronoi-vertex-cache
#SBATCH --partition=gpudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:15:00
#SBATCH --output=build-table-validation/clipping-%j.out

# Prepared for review; ask the user before submitting this job.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
validation_root="$PWD/build-clipping-validation/${SLURM_JOB_ID:?}"
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
cmake --build "$validation_root/cpu" --target table_mesh_check hydro_geometry_check -j 4
cmake -S . -B "$validation_root/gpu" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper \
    -DKokkos_DIR=/u/uli/kokkos-cuda/lib64/cmake/Kokkos \
    -DVORONOI_CUDA_RESOURCE_REPORT=ON
cmake --build "$validation_root/gpu" --target table_mesh_check hydro_geometry_check -j 4
for backend in cpu gpu; do
    srun --ntasks=1 "$validation_root/$backend/hydro_geometry_check"
done

run_case() {
    local distribution="$1" side="$2"
    for vertices in recomputed cached; do
        for backend in cpu gpu; do
            local sorter=insertion
            if [[ "$backend" == gpu ]]; then sorter=team_bitonic; fi
            srun --ntasks=1 "$validation_root/$backend/table_mesh_check" \
                "$validation_root/$backend-$distribution-$vertices-$side.csv" \
                "$side" "$sorter" "$vertices" "$distribution"
        done
        for suffix in '' .hydro.csv; do
            python3 compare_faces.py "$validation_root/cpu-$distribution-$vertices-$side.csv$suffix" \
                "$validation_root/gpu-$distribution-$vertices-$side.csv$suffix"
        done
    done
    for backend in cpu gpu; do
        for suffix in '' .hydro.csv; do
            python3 compare_faces.py "$validation_root/$backend-$distribution-recomputed-$side.csv$suffix" \
                "$validation_root/$backend-$distribution-cached-$side.csv$suffix"
        done
    done
}
run_case lattice 12
run_case disordered 12
run_case regular 12
run_case lattice 32
